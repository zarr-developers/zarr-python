import json
from collections.abc import MutableMapping


class StoreComparer(MutableMapping):
    """
    Compare two store implementations, and make sure to do the same operation on
    both stores.

    The operation from the first store are always considered as reference and
    the will make sure the second store will return the same value, or raise
    the same exception where relevant.

    This should have minimal impact on API, but can as some generators are
    reified and sorted to make sure they are identical.
    """

    def __init__(self, reference, tested):
        self.reference = reference
        self.tested = tested

    def __getitem__(self, key):
        try:
            k1 = self.reference[key]
        except Exception as e1:
            try:
                k2 = self.tested[key]
                assert False, "should raise, got {} for {}".format(k2, key)
            except Exception as e2:
                raise
                if not isinstance(e2, type(e1)):
                    raise AssertionError("Expecting {type(e1)} got {type(e2)}") from e2
            raise
        k2 = self.tested[key]
        if key.endswith((".zgroup", ".zarray")):
            j1, j2 = json.loads(k1.decode()), json.loads(k2.decode())
            assert j1 == j2, "{} != {}".format(j1, j2)
        else:
            assert k2 == k1, "{} != {}\n missing: {},\n extra:{}".format(
                k1, k2, k1 - k2, k2 - k1
            )
        return k1

    def __setitem__(self, key, value):
        # todo : not quite happy about casting here, maybe we shoudl stay strict ?
        from numcodecs.compat import ensure_bytes

        value = ensure_bytes(value)
        try:
            self.reference[key] = value
        except Exception as e:
            try:
                self.tested[key] = value
            except Exception as e2:
                assert isinstance(e, type(e2))
        try:
            self.tested[key] = value
        except Exception as e:
            raise
            assert False, "should not raise, got {}".format(e)

    def keys(self):
        try:
            k1 = list(sorted(self.reference.keys()))
        except Exception as e1:
            try:
                k2 = self.tested.keys()
                assert False, "should raise"
            except Exception as e2:
                assert isinstance(e2, type(e1))
            raise
        k2 = sorted(self.tested.keys())
        assert k2 == k1, "got {};\n expecting {}\n missing: {},\n extra:{}".format(
            k1, k2, set(k1) - set(k2), set(k2) - set(k1)
        )
        return k1

    def __delitem__(self, key):
        try:
            del self.reference[key]
        except Exception as e1:
            try:
                del self.tested[key]
                assert False, "should raise"
            except Exception as e2:
                assert isinstance(e2, type(e1))
            raise
        del self.tested[key]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def __contains__(self, key):
        return key in self.reference
