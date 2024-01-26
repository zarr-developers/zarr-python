from __future__ import annotations

from abc import abstractproperty, ABC
from collections.abc import MutableMapping
from typing import Dict, Any


class BaseGroup(ABC):
    @abstractproperty
    def attrs(self) -> Dict[str, Any]:
        """User-defined attributes."""
        ...

    @abstractproperty
    def info(self) -> Any:  # TODO: type this later
        """Return diagnostic information about the group."""
        ...


class AsynchronousGroup(BaseGroup):
    pass
    # TODO: (considering the following api)
    # store_path (rename to path?)
    # nchildren - number of child groups + arrays
    # children (async iterator)
    # contains - check if child exists
    # getitem - get child
    # group_keys (async iterator)
    # groups (async iterator)
    # array_keys (async iterator)
    # arrays (async iterator)
    # visit
    # visitkeys
    # visitvalues
    # tree
    # create_group
    # require_group
    # create_groups
    # require_groups
    # create_dataset
    # require_dataset
    # create
    # empty
    # zeros
    # ones
    # full
    # array
    # empty_like
    # zeros_like
    # ones_like
    # full_like
    # move


class SynchronousGroup(BaseGroup, MutableMapping):
    # TODO - think about if we want to keep the MutableMapping abstraction or
    pass
    # store_path (rename to path?)
    # __enter__
    # __exit__
    # group_keys
    # groups
    # array_keys
    # arrays
    # visit
    # visitkeys
    # visitvalues
    # visititems
    # tree
    # create_group
    # require_group
    # create_groups
    # require_groups
    # create_dataset
    # require_dataset
    # create
    # empty
    # zeros
    # ones
    # full
    # array
    # empty_like
    # zeros_like
    # ones_like
    # full_like
    # move
