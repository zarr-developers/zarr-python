from pathlib import Path

import pytest


@pytest.fixture(params=[str, Path])
def path_type(request):
    return request.param
