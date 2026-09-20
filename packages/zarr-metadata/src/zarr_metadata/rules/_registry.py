"""Register rules by document type and extension entity.

`@document_rule` and `@entity_rule` register checks where they are
defined, so a rule cannot be written without joining the set it belongs
to. Both reject dependencies absent from the document type. Entity rules
are keyed by `(field, canonical_name)` and require a corresponding shape
validator.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._chunk_grid import ChunkGrid
from zarr_metadata.rules._engine import Rule
from zarr_metadata.rules._entity import configuration_mapping, entity_configuration
from zarr_metadata.rules._spec import ArrayParts, propagate
from zarr_metadata.v3._extension_points import ExtensionPointField, canonical_name

if TYPE_CHECKING:
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON

from zarr_metadata.v3._shape import (
    blocking_problems,
    entity_configuration_keys,
    entity_name,
    entity_required_configuration_keys,
    modelled_entities,
    validate_known_entity_metadata,
)

EntityCheck = Callable[
    [Mapping[str, object], Mapping[str, object], "ArrayParts | None"],
    "tuple[ValidationProblem, ...]",
]
"""An entity rule's check: `(configuration, document, incoming)` in, problems out.

`incoming` is what the entity receives — for a codec, the array parts as
transformed by every codec before it in the chain, or `None` where this
package can no longer say. A caller with no chain context passes `None`.
Rules that need it test for `None` and decline; rules that do not simply
ignore it.

Problems carry locations relative to the entity's `configuration`; the
dispatcher re-bases them onto the entity's position in the document.
"""


@dataclass(frozen=True, slots=True)
class EntityRule:
    """One composition check for a named extension entity.

    Identified by `(field, entity)`, never by name alone: names are
    unique only within an extension point, and `bytes` is both a core
    codec and a registered extension data type. Keying by name would
    make a rule written for one fire on the other.

    `requires` are *document* keys the check reads beyond the entity
    itself (e.g. `shape`), gating the rule exactly as `Rule.requires`
    does.

    `reads` are the *required* configuration members the check subscripts.
    A rule runs only when none of them has a shape problem of its own,
    which is what makes `configuration["level"]` safe: a required member
    that is absent or ill-typed is reported at `("configuration", member)`
    and stands the rule down, while the rest of the entity is still
    judged. Only required members may be declared here — an optional one
    can be absent with nothing reported, so subscripting it would raise
    out of a validator.

    `reads_optional` are modelled members the check tests for presence
    rather than subscripting (`"endian" not in configuration`). They gate
    the rule the same way; they are separate so that the subscript
    guarantee above stays true by construction.
    """

    field: str
    entity: str
    requires: frozenset[str]
    reads: frozenset[str]
    reads_optional: frozenset[str]
    check: EntityCheck


_DOCUMENT_RULES: Final[dict[str, list[Rule]]] = defaultdict(list)
_ENTITY_RULES: Final[dict[tuple[str, str], list[EntityRule]]] = defaultdict(list)
_DOCUMENT_KEYS: Final[dict[str, frozenset[str]]] = {}
_DISPATCHED_FIELDS: Final[set[str]] = set()


def register_document_type(
    document_type: str,
    standard_keys: frozenset[str],
    extension_keys: frozenset[str] = frozenset(),
) -> None:
    """Declare a document type's known keys, so `requires` can be checked.

    `extension_keys` names keys that are not part of the document's
    TypedDict but that this package nonetheless recognizes — the v3
    `consolidated_metadata` convention is the only one today. Requiring
    them to be declared here rather than exempting unknown keys wholesale
    keeps the typo check meaningful.
    """
    _DOCUMENT_KEYS[document_type] = standard_keys | extension_keys


def _validate_requires(document_type: str, requires: frozenset[str], what: str) -> None:
    known = _DOCUMENT_KEYS.get(document_type)
    if known is None:
        msg = f"unknown document type {document_type!r} registering {what}"
        raise LookupError(msg)
    unknown = requires - known
    if len(unknown) != 0:
        msg = (
            f"{what} requires {sorted(unknown)}, which {document_type} documents "
            f"do not have; such a rule could never fire"
        )
        raise ValueError(msg)


def document_rule(
    document_type: str, requires: frozenset[str]
) -> Callable[[Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]], Rule]:
    """Register a whole-document rule, returning the `Rule` it becomes.

    The decorated function is replaced by its `Rule`, so a rule cannot be
    defined without being registered, and referencing one by name yields
    the registered object rather than a copy.
    """

    def decorate(
        check: Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]],
    ) -> Rule:
        _validate_requires(document_type, requires, f"rule {check.__name__!r}")
        rule = Rule(requires=requires, check=check)
        _DOCUMENT_RULES[document_type].append(rule)
        return rule

    return decorate


def entity_rule(
    document_type: str,
    field: ExtensionPointField,
    entity: str,
    requires: frozenset[str] = frozenset(),
    reads: frozenset[str] = frozenset(),
    reads_optional: frozenset[str] = frozenset(),
) -> Callable[[EntityCheck], EntityRule]:
    """Register a rule about one named entity within `document_type`.

    The entity must already be shape-modelled in `zarr_metadata.v3._shape`:
    entity rules read configuration members by name, so they only run once
    the shape validator vouches those members exist and are typed. A rule
    registered for an unmodelled name would silently never fire, so that
    is refused here rather than discovered as a missing check later.
    """

    def decorate(check: EntityCheck) -> EntityRule:
        _validate_requires(document_type, requires, f"entity rule {check.__name__!r}")
        canonical_entity = canonical_name(field, entity)
        if (field, canonical_entity) not in modelled_entities():
            msg = (
                f"entity rule {check.__name__!r} targets {entity!r}, which has no shape "
                f"validator in zarr_metadata.v3._shape; such a rule could never fire"
            )
            raise ValueError(msg)
        modelled = entity_configuration_keys(field, entity) or frozenset()
        required = entity_required_configuration_keys(field, entity) or frozenset()
        unmodelled = (reads | reads_optional) - modelled
        if len(unmodelled) != 0:
            msg = (
                f"entity rule {check.__name__!r} declares {sorted(unmodelled)}, which "
                f"{entity!r} does not model; such a member can never carry a value to read"
            )
            raise ValueError(msg)
        optional = reads - required
        if len(optional) != 0:
            msg = (
                f"entity rule {check.__name__!r} declares reads={sorted(optional)}, which "
                f"{entity!r} does not require; an absent optional member is reported by "
                f"nothing, so subscripting it would raise out of a validator. Declare it as "
                f"reads_optional and test for presence instead."
            )
            raise ValueError(msg)
        rule = EntityRule(
            field=field,
            entity=entity,
            requires=requires,
            reads=reads,
            reads_optional=reads_optional,
            check=check,
        )
        _ENTITY_RULES[field, canonical_entity].append(rule)
        return rule

    return decorate


def document_rules(document_type: str) -> tuple[Rule, ...]:
    """Every rule registered for `document_type`, in definition order."""
    return tuple(_DOCUMENT_RULES[document_type])


def dispatched_fields() -> frozenset[str]:
    """Extension points that have a dispatcher, so their rules can run.

    An entity rule registered at a field with no dispatcher is accepted and
    then never fires — the silent-pass failure this module exists to
    prevent. Checking coverage at registration would depend on import
    order, so `tests/rules/test_registry.py` asserts it instead.
    """
    return frozenset(_DISPATCHED_FIELDS)


def registered_entities() -> frozenset[tuple[str, str]]:
    """Every `(field, canonical name)` that has at least one registered rule."""
    return frozenset(_ENTITY_RULES)


def run_entity_rules(
    field: ExtensionPointField,
    value: object,
    document: Mapping[str, object],
    loc: tuple[str | int, ...],
    incoming: ArrayParts | None = None,
) -> tuple[ValidationProblem, ...]:
    """Run the rules registered for whatever entity `value` names.

    Declines silently when `value` names nothing known, when its shape is
    broken in a way that makes its configuration uninterpretable (the
    shape rule owns that complaint), or when a rule's required document
    keys are absent. An `unknown_key` never declines — see
    `zarr_metadata.v3._shape.blocking_problems`.
    """
    name = entity_name(value)
    if name is None:
        return ()
    rules = _ENTITY_RULES.get((field, canonical_name(field, name)))
    if rules is None or len(rules) == 0:
        return ()
    verdict = validate_known_entity_metadata(field, value)
    if verdict is None:
        return ()
    blocking = blocking_problems(verdict)
    # Only two locations mean there is no configuration to read: the entity
    # itself, and `configuration` as a whole. Anything else is about one
    # member — including `must_understand`, which is part of the envelope
    # and says nothing about whether the configuration is readable.
    if any(problem.loc in ((), ("configuration",)) for problem in blocking):
        return ()
    unusable = frozenset(
        str(problem.loc[1])
        for problem in blocking
        if len(problem.loc) >= 2 and problem.loc[0] == "configuration"
    )
    configuration = configuration_mapping(value)
    if configuration is None:
        return ()
    problems: list[ValidationProblem] = []
    for rule in rules:
        if not rule.requires <= document.keys():
            continue
        if len((rule.reads | rule.reads_optional) & unusable) != 0:
            continue
        for found in rule.check(configuration, document, incoming):
            # A rule that reports at the entity itself (an empty loc) is
            # judging the whole entity, not a member of its configuration —
            # and a bare-string entity has no `configuration` node to point at.
            base = (*loc, "configuration") if len(found.loc) != 0 else loc
            problems.append(ValidationProblem((*base, *found.loc), found.message, found.kind))
    return tuple(problems)


def dispatch_field(
    field: ExtensionPointField,
) -> Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]:
    """A check that runs entity rules for the entity in `document[field]`."""

    def check(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
        return run_entity_rules(field, document[field], document, (field,))

    _DISPATCHED_FIELDS.add(field)
    check.__name__ = f"_dispatch_{field}_entity_rules"
    return check


def dispatch_field_sequence(
    field: ExtensionPointField,
) -> Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]:
    """A check that runs entity rules for every entity in `document[field]`."""

    def check(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
        entries = document[field]
        if not isinstance(entries, (list, tuple)):
            return ()
        sequence = cast("tuple[object, ...]", entries)
        return run_chain_rules(field, sequence, document, (field,), chain_initial_spec(document))

    _DISPATCHED_FIELDS.add(field)
    check.__name__ = f"_dispatch_{field}_entity_rules"
    return check


def run_chain_rules(
    field: ExtensionPointField,
    codecs: Sequence[object],
    document: Mapping[str, object],
    loc: tuple[str | int, ...],
    initial: ArrayParts | None,
) -> tuple[ValidationProblem, ...]:
    """Run entity rules over a codec chain, propagating the array spec.

    Each codec's rules receive the spec that codec actually receives —
    the array as transformed by everything before it. Shared by the
    top-level `codecs` dispatcher and by sharding, whose inner pipelines
    are chains that start from the inner chunk.
    """
    problems: list[ValidationProblem] = []
    for index, entry, incoming in propagate(
        codecs, initial, lambda codec: entity_configuration(field, codec)
    ):
        problems.extend(run_entity_rules(field, entry, document, (*loc, index), incoming))
    return tuple(problems)


def chain_initial_spec(document: Mapping[str, object]) -> ArrayParts | None:
    """What enters a document's top-level codec chain.

    The parts a chunk pipeline encodes are the chunks of the document's
    chunk grid. A `data_type` that is not a metadata field has been
    rejected structurally already, but it costs only itself: the grid is
    still readable, and the geometry rules should still report what they
    can rather than making the reader fix one fault to discover the rest.
    """
    data_type = document.get("data_type")
    grid = ChunkGrid.of(document.get("chunk_grid"), document.get("shape"))
    if entity_name(data_type) is None:
        return ArrayParts(grid, None)
    return ArrayParts(grid, cast("ZarrV3MetadataFieldJSON", data_type))


__all__ = [
    "EntityCheck",
    "EntityRule",
    "chain_initial_spec",
    "dispatched_fields",
    "document_rule",
    "document_rules",
    "entity_rule",
    "registered_entities",
    "run_chain_rules",
    "run_entity_rules",
]
