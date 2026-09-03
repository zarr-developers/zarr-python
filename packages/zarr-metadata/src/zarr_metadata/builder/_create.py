"""One-shot factories for metadata document TypedDicts.

Each factory copies and normalizes its input, then applies structural and
composition validation. Invalid input raises one
`MetadataValidationError`. V3 array and group factories accept extension
fields through `extensions=` for compatibility with type checkers that do
not support PEP 728.

"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from typing_extensions import Unpack

from zarr_metadata.model._group import ZarrV2ConsolidatedMetadata
from zarr_metadata.model._validation import (
    ARRAY_METADATA_STANDARD_KEYS_V3,
    GROUP_METADATA_STANDARD_KEYS_V3,
    MetadataValidationError,
    ValidationProblem,
    arrays_to_tuples,
    parse_array_metadata_v2,
    parse_array_metadata_v3,
    parse_group_metadata_v2,
    parse_group_metadata_v3,
    validate_consolidated_metadata_v3,
)
from zarr_metadata.rules import (
    ZARR_V2_ARRAY_RULES,
    ZARR_V3_ARRAY_RULES,
    ZARR_V3_GROUP_RULES,
    run_rules,
    validate_array_metadata_v2,
    validate_group_metadata_v2,
)
from zarr_metadata.rules._v3_group import consolidated_entries_problems

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from zarr_metadata.v2.array import ZarrV2ArrayMetadataJSON, ZarrV2ZArrayJSON
    from zarr_metadata.v2.consolidated import ZarrV2ConsolidatedMetadataJSON
    from zarr_metadata.v2.group import ZarrV2GroupMetadataJSON, ZarrV2ZGroupJSON
    from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON, ZarrV3ExtensionField
    from zarr_metadata.v3.consolidated import ZarrV3ConsolidatedMetadataJSON
    from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON


def _merged_with_extensions(
    kwargs: Mapping[str, object],
    extensions: Mapping[str, ZarrV3ExtensionField] | None,
    standard_keys: AbstractSet[str],
) -> tuple[dict[str, object], list[ValidationProblem]]:
    """`kwargs` plus `extensions`, refusing extension names that shadow standard keys.

    A colliding name is reported and *not* merged, so a later structural or
    semantic pass judges the standard field the caller actually passed
    rather than a value smuggled in through the extension hatch.
    """
    document = dict(kwargs)
    problems: list[ValidationProblem] = []
    for name, value in (extensions or {}).items():
        if name in standard_keys:
            problems.append(
                ValidationProblem(
                    (name,),
                    f"{name!r} is a standard metadata key; pass it as a keyword argument",
                    "invalid_value",
                )
            )
        else:
            document[name] = value
    return document, problems


def _normalized(document: Mapping[str, object]) -> dict[str, object]:
    """A deep copy of `document` with JSON arrays materialized as tuples.

    Deep-copying first means the returned document shares no mutable state
    with the caller's arguments: mutating an input after the factory
    returns cannot alter the validated result.
    """
    return cast("dict[str, object]", arrays_to_tuples(copy.deepcopy(dict(document))))


def _raise_if_problems(problems: Sequence[ValidationProblem]) -> None:
    if len(problems) != 0:
        raise MetadataValidationError(problems)


def _reject_attributes(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """Problems for an `attributes` key in a strict on-disk v2 document.

    The strict `.zarray` / `.zgroup` shapes exclude `attributes` (it lives
    in the sibling `.zattrs` file). The signature enforces that statically
    at keyword call sites; this is the runtime backstop for `**`-splatted
    and untyped callers, without which the merged-form parser would accept
    the key and the returned value would not be the type it claims.
    """
    if "attributes" not in document:
        return ()
    return (
        ValidationProblem(
            ("attributes",),
            "'attributes' is not part of the on-disk document (it belongs to the "
            "sibling .zattrs file); use the merged-form factory instead",
            "invalid_value",
        ),
    )


def create_zarr_v3_array_metadata_json(
    *,
    extensions: Mapping[str, ZarrV3ExtensionField] | None = None,
    **kwargs: Unpack[ZarrV3ArrayMetadataJSON],
) -> ZarrV3ArrayMetadataJSON:
    """A validated v3 array metadata document (the `zarr.json` content for an array).

    Required keys are enforced statically by the signature; at runtime the
    document is checked structurally (via the model layer's parser) and
    semantically (via `ZARR_V3_ARRAY_RULES`), and every problem from both
    passes is raised together in one `MetadataValidationError`. Extension
    fields go in `extensions`; names that shadow standard keys are rejected.
    """
    document, problems = _merged_with_extensions(
        kwargs, extensions, ARRAY_METADATA_STANDARD_KEYS_V3
    )
    normalized = _normalized(document)
    parsed: ZarrV3ArrayMetadataJSON | None = None
    try:
        parsed = parse_array_metadata_v3(normalized)
    except MetadataValidationError as error:
        problems.extend(error.problems)
    problems.extend(run_rules(ZARR_V3_ARRAY_RULES, normalized))
    _raise_if_problems(problems)
    assert parsed is not None
    return parsed


def create_zarr_v3_group_metadata_json(
    *,
    extensions: Mapping[str, ZarrV3ExtensionField] | None = None,
    **kwargs: Unpack[ZarrV3GroupMetadataJSON],
) -> ZarrV3GroupMetadataJSON:
    """A validated v3 group metadata document (the `zarr.json` content for a group).

    Extension fields go in `extensions`; names that shadow standard keys
    are rejected. The composition rules recurse into inline consolidated
    metadata, so an embedded child document invalid under its own rules
    is reported here, at its path.
    """
    document, problems = _merged_with_extensions(
        kwargs, extensions, GROUP_METADATA_STANDARD_KEYS_V3
    )
    normalized = _normalized(document)
    parsed: ZarrV3GroupMetadataJSON | None = None
    try:
        parsed = parse_group_metadata_v3(normalized)
    except MetadataValidationError as error:
        problems.extend(error.problems)
    problems.extend(run_rules(ZARR_V3_GROUP_RULES, normalized))
    _raise_if_problems(problems)
    assert parsed is not None
    return parsed


def create_zarr_v3_consolidated_metadata_json(
    **kwargs: Unpack[ZarrV3ConsolidatedMetadataJSON],
) -> ZarrV3ConsolidatedMetadataJSON:
    """A validated v3 inline consolidated metadata object.

    This is the value embedded in a v3 group document under the
    `consolidated_metadata` key, not a store document of its own.
    """
    normalized = _normalized(kwargs)
    _raise_if_problems(
        validate_consolidated_metadata_v3(normalized) + consolidated_entries_problems(normalized)
    )
    return cast("ZarrV3ConsolidatedMetadataJSON", normalized)


def create_zarr_v2_array_metadata_json(
    **kwargs: Unpack[ZarrV2ArrayMetadataJSON],
) -> ZarrV2ArrayMetadataJSON:
    """A validated v2 array metadata document, in-memory merged form.

    Models `.zarray` plus the sibling `.zattrs` folded in as `attributes`.
    For the strict on-disk `.zarray` shape use `create_zarr_v2_zarray_json`.
    """
    normalized = _normalized(kwargs)
    parsed: ZarrV2ArrayMetadataJSON | None = None
    problems: list[ValidationProblem] = []
    try:
        parsed = parse_array_metadata_v2(normalized)
    except MetadataValidationError as error:
        problems.extend(error.problems)
    problems.extend(run_rules(ZARR_V2_ARRAY_RULES, normalized))
    _raise_if_problems(problems)
    assert parsed is not None
    return parsed


def create_zarr_v2_group_metadata_json(
    **kwargs: Unpack[ZarrV2GroupMetadataJSON],
) -> ZarrV2GroupMetadataJSON:
    """A validated v2 group metadata document, in-memory merged form.

    Models `.zgroup` plus the sibling `.zattrs` folded in as `attributes`.
    For the strict on-disk `.zgroup` shape use `create_zarr_v2_zgroup_json`.
    """
    parsed: ZarrV2GroupMetadataJSON | None = None
    problems: list[ValidationProblem] = []
    try:
        parsed = parse_group_metadata_v2(_normalized(kwargs))
    except MetadataValidationError as error:
        problems.extend(error.problems)
    _raise_if_problems(problems)
    assert parsed is not None
    return parsed


def create_zarr_v2_zarray_json(**kwargs: Unpack[ZarrV2ZArrayJSON]) -> ZarrV2ZArrayJSON:
    """A validated on-disk `.zarray` document (strict form, no `attributes`).

    Structurally checked with the merged-form parser plus a runtime
    rejection of `attributes`: the strict shape is the merged shape minus
    `attributes`, and the runtime check holds for callers the signature's
    static exclusion cannot see (`**`-splatted mappings, untyped code).
    """
    normalized = _normalized(kwargs)
    problems: list[ValidationProblem] = list(_reject_attributes(normalized))
    parsed: ZarrV2ArrayMetadataJSON | None = None
    try:
        parsed = parse_array_metadata_v2(normalized)
    except MetadataValidationError as error:
        problems.extend(error.problems)
    problems.extend(run_rules(ZARR_V2_ARRAY_RULES, normalized))
    _raise_if_problems(problems)
    assert parsed is not None
    return cast("ZarrV2ZArrayJSON", parsed)


def create_zarr_v2_zgroup_json(**kwargs: Unpack[ZarrV2ZGroupJSON]) -> ZarrV2ZGroupJSON:
    """A validated on-disk `.zgroup` document (strict form, no `attributes`).

    Structurally checked with the merged-form parser plus a runtime
    rejection of `attributes`: the strict shape is the merged shape minus
    `attributes`, and the runtime check holds for callers the signature's
    static exclusion cannot see (`**`-splatted mappings, untyped code).
    """
    normalized = _normalized(kwargs)
    problems: list[ValidationProblem] = list(_reject_attributes(normalized))
    parsed: ZarrV2GroupMetadataJSON | None = None
    try:
        parsed = parse_group_metadata_v2(normalized)
    except MetadataValidationError as error:
        problems.extend(error.problems)
    _raise_if_problems(problems)
    assert parsed is not None
    return cast("ZarrV2ZGroupJSON", parsed)


def _validate_v2_consolidated_envelope(
    document: Mapping[str, object],
) -> tuple[ValidationProblem, ...]:
    """Every reason `document` is not a `.zmetadata` envelope.

    The envelope itself is the model layer's judgment; on top of it, each
    entry's path suffix selects the strict on-disk document shape that its
    value must satisfy.
    """
    try:
        ZarrV2ConsolidatedMetadata.from_json(document)
    except MetadataValidationError as error:
        return error.problems
    problems: list[ValidationProblem] = []
    for key, entry in cast("Mapping[str, object]", document["metadata"]).items():
        if not isinstance(entry, Mapping):
            problems.append(
                ValidationProblem(("metadata", key), "expected a JSON object", "invalid_type")
            )
            continue
        entry_mapping = cast("Mapping[str, object]", entry)
        if key.endswith(".zarray"):
            nested = validate_array_metadata_v2(entry_mapping) + _reject_attributes(entry_mapping)
        elif key.endswith(".zgroup"):
            nested = validate_group_metadata_v2(entry_mapping) + _reject_attributes(entry_mapping)
        elif key.endswith(".zattrs"):
            nested = ()
        else:
            nested = (
                ValidationProblem(
                    (),
                    "expected a v2 metadata file suffix: .zarray, .zgroup, or .zattrs",
                    "invalid_value",
                ),
            )
        problems.extend(
            ValidationProblem(("metadata", key, *found.loc), found.message, found.kind)
            for found in nested
        )
    return tuple(problems)


def create_zarr_v2_consolidated_metadata_json(
    **kwargs: Unpack[ZarrV2ConsolidatedMetadataJSON],
) -> ZarrV2ConsolidatedMetadataJSON:
    """A validated `.zmetadata` consolidated metadata document.

    The runtime pass checks the envelope and validates each nested value
    against the strict document shape selected by its path suffix. This is
    the runtime backstop for callers the signature's static enforcement
    cannot see (`**`-splatted mappings, untyped code).
    """
    normalized = _normalized(kwargs)
    _raise_if_problems(_validate_v2_consolidated_envelope(normalized))
    return cast("ZarrV2ConsolidatedMetadataJSON", normalized)


__all__ = [
    "create_zarr_v2_array_metadata_json",
    "create_zarr_v2_consolidated_metadata_json",
    "create_zarr_v2_group_metadata_json",
    "create_zarr_v2_zarray_json",
    "create_zarr_v2_zgroup_json",
    "create_zarr_v3_array_metadata_json",
    "create_zarr_v3_consolidated_metadata_json",
    "create_zarr_v3_group_metadata_json",
]
