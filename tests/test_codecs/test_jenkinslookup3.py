import pytest

from tests.test_codecs.conftest import BaseTestCodec
from zarr.codecs import numcodecs as _numcodecs


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning")
class TestJenkinsLookup3Codec(BaseTestCodec):
    test_cls = _numcodecs.JenkinsLookup3
    valid_json_v2 = ({"id": "jenkins_lookup3", "initval": 0, "prefix": None},)
    valid_json_v3 = (
        {
            "name": "jenkins_lookup3",
            "configuration": {"initval": 0, "prefix": None},
        },
    )
