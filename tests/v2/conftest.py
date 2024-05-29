import pytest
import pathlib


@pytest.fixture(params=[str, pathlib.Path])
def path_type(request):
    return request.param


@pytest.fixture
def project_root(request):
    return request.config.rootpath
