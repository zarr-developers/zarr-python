from __future__ import annotations

import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs
from zarr.codecs.numcodecs.jenkins_lookup3 import check_json_v2, check_json_v3


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestJenkinsLookup3Codec(BaseTestCodec):
    test_cls = _numcodecs.JenkinsLookup3
    valid_json_v2 = ({"id": "jenkins_lookup3", "initval": 0, "prefix": None},)
    valid_json_v3 = (
        {
            "name": "jenkins_lookup3",
            "configuration": {"initval": 0, "prefix": None},
        },
        {
            "name": "numcodecs.jenkins_lookup3",
            "configuration": {"initval": 0, "prefix": None},
        },
    )

    @staticmethod
    def check_json_v2(data: object) -> bool:
        return check_json_v2(data)

    @staticmethod
    def check_json_v3(data: object) -> bool:
        return check_json_v3(data)


def test_v3_json_alias() -> None:
    from zarr.codecs import numcodecs as _numcodecs

    """
    Test that the default JSON output of the legacy numcodecs.zarr3.JenkinsLookup3 codec is readable, even if it's
    underspecified.
    """
    assert _numcodecs.JenkinsLookup3.from_json(
        {"name": "numcodecs.jenkins_lookup3", "configuration": {}}
    ) == _numcodecs.JenkinsLookup3(initval=0, prefix=None)
