import unittest
from zarr.creation import array


class TestArrayMissingKeys(unittest.TestCase):
    def test_raises_on_missing_key(self):
        a = array(range(4), chunks=2)

        # configure raise on missing chunk
        a.set_options(fill_missing_chunk=False)

        # pop first chunk
        a.chunk_store.pop("0")

        # read from missing chunk and make sure fill-value is returned
        assert a.fill_value == a[0]
        assert a.fill_value == a[1]

        # read from avaible chunk w/o error        
        assert 2 = a[2]
        assert 3 = a[3]

        # reading missing chunk should raise
        with self.assertRaises(KeyError):
            a[0]

        with self.assertRaises(KeyError):
            a[:2]
