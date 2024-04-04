import pytest
import pathlib


@pytest.fixture(params=[str, pathlib.Path])
def path_type(request):
    return request.param
