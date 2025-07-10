from zarr.registry import get_codec

def test():
    c = get_codec('gzip', {"level": 1})
