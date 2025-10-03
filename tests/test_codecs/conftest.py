from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from zarr.abc.codec import Codec
    from zarr.core.common import CodecJSON_V2, CodecJSON_V3


class BaseTestCodec:
    """
    A base class for testing codec classes.

    Attributes
    ----------

    test_cls : type[Codec]
        The codec class being tested
    valid_json_v2 : ClassVar[tuple[CodecJSON_V2, ...]]
        A tuple of valid JSON representations for Zarr format version 2.
    valid_json_v2 : ClassVar[tuple[CodecJSON_V2, ...]]
        A tuple of valid JSON representations for Zarr format version 2.
    """

    test_cls: type[Codec]
    valid_json_v2: ClassVar[tuple[CodecJSON_V2 | object, ...]]
    valid_json_v3: ClassVar[tuple[CodecJSON_V3 | object, ...]]

    @staticmethod
    def check_json_v2(data: object) -> bool:
        raise NotImplementedError

    @staticmethod
    def check_json_v3(data: object) -> bool:
        raise NotImplementedError

    def test_from_json_v2(self, valid_json_v2: CodecJSON_V2) -> None:
        """
        Test that the codec generated from valid JSON generates a JSON representation that generates
        the same codec
        """
        codec = self.test_cls.from_json(valid_json_v2)
        assert codec.from_json(codec.to_json(zarr_format=2)) == codec
        assert self.check_json_v2(codec.to_json(zarr_format=2))

    def test_from_json_v3(self, valid_json_v3: CodecJSON_V3) -> None:
        """
        Test that the codec generated from valid JSON generates a JSON representation that generates
        the same codec
        """
        codec = self.test_cls.from_json(valid_json_v3)
        assert codec.from_json(codec.to_json(zarr_format=3)) == codec
        assert self.check_json_v3(codec.to_json(zarr_format=3))


def pytest_generate_tests(metafunc: Any) -> None:
    """
    This is a pytest hook to parametrize class-scoped fixtures.

    This hook allows us to define class-scoped fixtures as class attributes and then
    generate the parametrize calls for pytest. This allows the fixtures to be
    reused across multiple tests within the same class.

    For example, if you had a regular pytest class like this:

    class TestClass:
       @pytest.mark.parametrize("param_a", [1, 2, 3])
        def test_method(self, param_a):
            ...

    Child classes inheriting from ``TestClass`` would not be able to override the ``param_a`` fixture

    this implementation of ``pytest_generate_tests`` allows you to define class-scoped fixtures as
    class attributes, which allows the following to work:

    class TestExample:
        param_a = [1, 2, 3]

        def test_example(self, param_a):
            ...

    # this class will have its test_example method parametrized with the values of TestB.param_a
    class TestB(TestExample):
        param_a = [1, 2, 100, 10]

    """
    # Iterate over all the fixtures defined in the class
    # and parametrize them with the values defined in the class
    # This allows us to define class-scoped fixtures as class attributes
    # and then generate the parametrize calls for pytest
    for fixture_name in metafunc.fixturenames:
        if hasattr(metafunc.cls, fixture_name):
            params = getattr(metafunc.cls, fixture_name)
            # Only parametrize if params is a tuple or list, not a function/method
            if isinstance(params, (tuple, list)):
                metafunc.parametrize(fixture_name, params, scope="class", ids=str)
