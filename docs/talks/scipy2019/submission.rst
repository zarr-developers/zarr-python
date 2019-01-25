Zarr - scalable storage of tensor data for use in parallel and distributed computations
=======================================================================================

SciPy 2019 submission.


Short summary
-------------

(target: 100 words)

@@TODO


Abstract
--------

(target: 500 words)

Background
~~~~~~~~~~

Across a broad range of scientific disciplines, data are naturally
represented and stored as N-dimensional typed arrays, also known as
tensors. The volume of data being generated is outstripping our
ability to analyse it, and scientific communities are looking for ways
to leverage modern multi-core CPUs and distributed computing
platforms, including cloud computing. Retrieval and storage of data is
often the major bottleneck, and new approaches to data storage are
needed to accelerate distributed computations and enable them to scale
on a variety of platforms.

Methods
~~~~~~~

We have designed a new storage format and protocol for tensor data
[1_], and have released an open source Python implementation [2_,
3_]. Our approach builds on data storage concepts from HDF5 [4_],
particularly chunking and compression, and hierarchical organisation
of datasets. Key design goals include: platform-independent and as
simple as possible to implement in other programming languages;
support for multiple concurrent readers or writers; support for a
variety of parallel computing environments, from multi-threaded
execution on a single CPU to multi-process execution across a
multi-node cluster; pluggable storage subsystem with support for cloud
object stores; pluggable encoding subsystem with support for a variety
of modern compressors.

Results
~~~~~~~

We illustrate the use of Zarr with examples from several different
scientific domains. Zarr is being used within the Pangeo project [5_],
which is building a community platform for big data geoscience. The
Pangeo community have converted several existing datasets to Zarr,
including @@TODO, and have demonstrated their use in computations
including @@TODO. Within the MalariaGEN project [6_], Zarr is used to
store genome variation data from next-generation sequencing of natural
populations of malaria parasites and mosquitoes (see, e.g., [7_]), and
these data are used as input to analyses of the evolution of these
organisms in response to selective pressure from anti-malarial drugs
and insecticides. @@TODO another example.

Conclusions
~~~~~~~~~~~

Zarr is generating interest from potential users across a wide range
of scientific domains, and work is ongoing to establish a community
process for further development of the specifications and
implementations in other programming languages. Zarr has recently been
presented in several venues, including a webinar for the ESIP
Federation tech dive series [8_], and @@TODO @rabernat's recent talk
[9_].


References
~~~~~~~~~~

.. _1: https://zarr.readthedocs.io/en/stable/spec/v2.html
.. _2: https://github.com/zarr-developers/zarr
.. _3: https://github.com/zarr-developers/numcodecs
.. _4: https://www.hdfgroup.org/solutions/hdf5/
.. _5: https://pangeo.io/
.. _6: https://www.malariagen.net/
.. _7: http://alimanfoo.github.io/2016/09/21/genotype-compression-benchmark.html
.. _8: http://wiki.esipfed.org/index.php/Interoperability_and_Technology/Tech_Dive_Webinar_Series#8_March.2C_2018:_.22Zarr:_A_simple.2C_open.2C_scalable_solution_for_big_NetCDF.2FHDF_data_on_the_Cloud.22:_Alistair_Miles.2C_University_of_Oxford.
.. _9: @@TODO


Authors
-------

@@TODO
