Groups (``zarr.hierarchy``)
===========================
.. module:: zarr.hierarchy

.. autofunction:: group
.. autofunction:: open_group

.. autoclass:: Group

    .. automethod:: __len__
    .. automethod:: __iter__
    .. automethod:: __contains__
    .. automethod:: __getitem__
    .. automethod:: __enter__
    .. automethod:: __exit__
    .. automethod:: group_keys
    .. automethod:: groups
    .. automethod:: array_keys
    .. automethod:: arrays
    .. automethod:: visit
    .. automethod:: visitkeys
    .. automethod:: visitvalues
    .. automethod:: visititems
    .. automethod:: tree
    .. automethod:: create_group
    .. automethod:: require_group
    .. automethod:: create_groups
    .. automethod:: require_groups
    .. automethod:: create_dataset
    .. automethod:: require_dataset
    .. automethod:: create
    .. automethod:: empty
    .. automethod:: zeros
    .. automethod:: ones
    .. automethod:: full
    .. automethod:: array
    .. automethod:: empty_like
    .. automethod:: zeros_like
    .. automethod:: ones_like
    .. automethod:: full_like
    .. automethod:: move