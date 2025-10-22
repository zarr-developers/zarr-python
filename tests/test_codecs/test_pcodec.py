import numpy as np
import pytest

from tests.test_codecs.conftest import BaseTestCodec
from tests.test_codecs.test_numcodecs import EXPECTED_WARNING_STR, compare_json_dicts
from zarr.codecs.numcodecs import _NumcodecsArrayBytesCodec, _NumcodecsCodec
from zarr.codecs.numcodecs.pcodec import PCodec
from zarr.core.array import create_array
from zarr.core.common import CodecJSON, CodecJSON_V2, ZarrFormat
from zarr.errors import ZarrUserWarning

pytest.importorskip("pcodec")


class TestPCodec(BaseTestCodec):
    test_cls = PCodec
    valid_json_v2 = (
        {
            "id": "pcodec",
            "level": 8,
            "mode_spec": "auto",
            "delta_spec": "auto",
            "paging_spec": "equal_pages_up_to",
            "delta_encoding_order": 1,
            "equal_pages_up_to": 262144,
        },
    )
    valid_json_v3 = (
        {
            "name": "pcodec",
            "configuration": {
                "level": 8,
                "mode_spec": "auto",
                "delta_spec": "auto",
                "paging_spec": "equal_pages_up_to",
                "delta_encoding_order": 1,
                "equal_pages_up_to": 262144,
            },
        },
        {
            "name": "numcodecs.pcodec",
            "configuration": {
                "level": 8,
                "mode_spec": "auto",
                "delta_spec": "auto",
                "paging_spec": "equal_pages_up_to",
                "delta_encoding_order": 1,
                "equal_pages_up_to": 262144,
            },
        },
    )


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
@pytest.mark.parametrize(("codec", "expected"), [(PCodec(level=8, delta_encoding_order=1),)])
def test_json_roundtrip_default_config(
    codec: _NumcodecsCodec, expected: CodecJSON_V2, zarr_format: ZarrFormat
) -> None:
    """Test JSON serialization and roundtrip for all codecs with various configurations."""
    # Helper function to compare dictionaries with potential numpy arrays

    # Test serialization
    expected_transformed: CodecJSON
    if zarr_format == 3:
        expected_transformed = {
            "name": expected["id"],
            "configuration": {k: v for k, v in expected.items() if k != "id"},
        }
    else:
        expected_transformed = expected

    json_output = codec.to_json(zarr_format=zarr_format)
    assert compare_json_dicts(json_output, expected_transformed), (
        f"JSON mismatch: {json_output} != {expected_transformed}"
    )

    codec_from_json = type(codec).from_json(json_output)

    original_config = codec.codec_config
    roundtrip_config = codec_from_json.codec_config

    assert compare_json_dicts(roundtrip_config, original_config), (
        f"Roundtrip config mismatch: {roundtrip_config} != {original_config}"
    )


def test_generic_bytes_codec(codec_class: type[_NumcodecsArrayBytesCodec]) -> None:
    codec_class = PCodec

    data = np.arange(0, 256, dtype="float32").reshape((16, 16))

    with pytest.warns(ZarrUserWarning, match=EXPECTED_WARNING_STR):
        a = create_array(
            {},
            shape=data.shape,
            chunks=(16, 16),
            dtype=data.dtype,
            fill_value=0,
            serializer=codec_class(),
        )

    a[:, :] = data.copy()  # type: ignore[index]
    np.testing.assert_array_equal(data, a[:, :])  # type: ignore[index]
