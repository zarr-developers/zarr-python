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

from zarr_metadata.rules._engine import Rule, as_string_mapping, prefixed
from zarr_metadata.rules._spec import NOTHING_KNOWN, ArraySpec, initial_spec, propagate
from zarr_metadata.v3._extension_points import CHUNK_GRID, ExtensionPointField, canonical_name
from zarr_metadata.v3._shape import (
    blocking_problems,
    entity_name,
    modelled_entities,
    validate_known_entity_metadata,
)

if TYPE_CHECKING:
    from zarr_metadata.model._validation import ValidationProblem

EntityCheck = Callable[
    [Mapping[str, object], Mapping[str, object], "ArraySpec"],
    "tuple[ValidationProblem, ...]",
]
"""An entity rule's check: `(configuration, document, incoming)` in, problems out.

`incoming` is the `ArraySpec` the entity receives — for a codec, the
array as transformed by every codec before it in the chain. Fields this
package cannot determine are `None`; a caller with no chain context
passes `NOTHING_KNOWN`. Rules that need a field test it `is None` and
decline; rules that do not simply ignore the spec.

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
    """

    field: str
    entity: str
    requires: frozenset[str]
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
        rule = EntityRule(field=field, entity=entity, requires=requires, check=check)
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
    incoming: ArraySpec = NOTHING_KNOWN,
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
    # Entity rules read configuration members by name, so they may only run
    # once the shape validator vouches those members exist and are typed.
    configuration = entity_configuration(field, value)
    if configuration is None:
        return ()
    problems: list[ValidationProblem] = []
    for rule in rules:
        if not rule.requires <= document.keys():
            continue
        problems.extend(
            prefixed((*loc, "configuration"), rule.check(configuration, document, incoming))
        )
    return tuple(problems)


def entity_configuration(field: ExtensionPointField, value: object) -> Mapping[str, object] | None:
    """`value`'s configuration if its modelled fields are usable, else None.

    Shared by the dispatchers and by rules that reach across entities
    (sharding's nested pipelines). `unknown_key` problems do not make an
    entity unusable; anything else does.
    """
    verdict = validate_known_entity_metadata(field, value)
    if verdict is None or len(blocking_problems(verdict)) != 0:
        return None
    mapping = as_string_mapping(value)
    if mapping is None:
        # Bare-string metadata is the canonical spelling for entities whose
        # configuration is optional. Rules still need a real mapping to run
        # against, especially when they judge a missing optional member.
        return {} if isinstance(value, str) else None
    if "configuration" not in mapping:
        return {}
    return as_string_mapping(mapping["configuration"])


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
    initial: ArraySpec,
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


def chain_initial_spec(document: Mapping[str, object]) -> ArraySpec:
    """The spec entering a document's top-level codec chain.

    The array a chunk pipeline encodes is one chunk: shape from a regular
    grid this package can read (None otherwise), data type from the
    document. Non-positive chunk extents yield None for the shape — the
    grid's own values rule owns that complaint, and geometry against a
    zero extent is noise on top of it.
    """
    from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID_NAME

    grid = document.get("chunk_grid")
    chunk_shape: tuple[int, ...] | None = None
    if entity_name(grid) == REGULAR_CHUNK_GRID_NAME:
        configuration = entity_configuration(CHUNK_GRID, grid)
        extents = configuration.get("chunk_shape") if configuration is not None else None
        if isinstance(extents, tuple):
            values = cast("tuple[object, ...]", extents)
            if all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in values):
                chunk_shape = cast("tuple[int, ...]", values)
    return initial_spec(document, chunk_shape)


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
