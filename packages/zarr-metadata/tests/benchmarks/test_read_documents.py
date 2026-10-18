"""Each benchmark case reads: the script does not rot as the package changes."""

from __future__ import annotations

import pytest

from tests.benchmarks.read_documents import CASES


@pytest.mark.parametrize(
    ("name", "read"), [(name, read) for name, read, _ in CASES], ids=[name for name, _, _ in CASES]
)
def test_each_case_reads(name: str, read: object) -> None:
    assert callable(read)
    assert read() is not None
