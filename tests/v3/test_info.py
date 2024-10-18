import textwrap

from zarr._info import GroupInfo


def test_group_info_repr() -> None:
    info = GroupInfo(name="a", store_type="MemoryStore", read_only=False)
    result = repr(info)
    expected = textwrap.dedent("""\
        Name        : a
        Type        : Group
        Read-only   : False
        Store type  : MemoryStore""")
    assert result == expected


def test_group_info_complete() -> None:
    info = GroupInfo(name="a", store_type="MemoryStore", read_only=False, count_arrays=10, count_groups=4, count_members=14)
    result = repr(info)
    expected = textwrap.dedent("""\
        Name        : a
        Type        : Group
        Read-only   : False
        Store type  : MemoryStore
        No. members : 14
        No. arrays  : 10
        No. groups  : 4""")
    assert result == expected

