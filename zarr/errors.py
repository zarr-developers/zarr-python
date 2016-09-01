# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from zarr.compat import PY2


if PY2:  # pragma: no cover

    class PermissionError(Exception):
        pass

else:

    PermissionError = PermissionError


class MetadataError(Exception):
    pass
