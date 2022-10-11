Zarr - scalable storage of tensor data for use in parallel and distributed computing
====================================================================================

SciPy 2019 submission.


Short summary
-------------

Many scientific problems involve computing over large N-dimensional
typed arrays of data, and reading or writing data is often the major
bottleneck limiting speed or scalability. The Zarr project is
developing a simple, scalable approach to storage of such data in a
way that is compatible with a range of approaches to distributed and
parallel computing. We describe the Zarr protocol and data storage
format, and the current state of implementations for various
programming languages including Python. We also describe current uses
of Zarr in malaria genomics, the Human Cell Atlas, and the Pangeo
project.


Abstract
--------

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
of datasets. Key design goals include: a simple protocol and format
that can be implemented in other programming languages; support for
multiple concurrent readers or writers; support for a variety of
parallel computing environments, from multi-threaded execution on a
single CPU to multi-process execution across a multi-node cluster;
pluggable storage subsystem with support for file systems, key-value
databases and cloud object stores; pluggable encoding subsystem with
support for a variety of modern compressors.

Results
~~~~~~~

We illustrate the use of Zarr with examples from several scientific
domains. Zarr is being used within the Pangeo project [5_], which is
building a community platform for big data geoscience. The Pangeo
community have converted a number of existing climate modelling and
satellite observation datasets to Zarr [6_], and have demonstrated
their use in computations using HPC and cloud computing
environments. Within the MalariaGEN project [7_], Zarr is used to
store genome variation data from next-generation sequencing of natural
populations of malaria parasites and mosquitoes [8_] and these data
are used as input to analyses of the evolution of these organisms in
response to selective pressure from anti-malarial drugs and
insecticides. Zarr is being used within the Human Cell Atlas (HCA)
project [9_], which is building a reference atlas of healthy human
cell types. This project hopes to leverage this information to better
understand the dysregulation of cellular states that underly human
disease. The Human Cell Atlas uses Zarr as the output data format
because it enables the project to easily generate matrices containing
user-selected subsets of cells.

Conclusions
~~~~~~~~~~~

Zarr is generating interest across a range of scientific domains, and
work is ongoing to establish a community process to support further
development of the specifications and implementations in other
programming languages [10_, 11_, 12_] and building interoperability
with a similar project called N5 [13_]. Other packages within the
PyData ecosystem, notably Dask [14_], Xarray [15_] and Intake [16_],
have added capability to read and write Zarr, and together these
packages provide a compelling solution for large scale data science
using Python [17_]. Zarr has recently been presented in several
venues, including a webinar for the ESIP Federation tech dive series
[18_], and a talk at the AGU Fall Meeting 2018 [19_].


References
~~~~~~~~~~

.. _1: https://zarr.readthedocs.io/en/stable/spec/v2.html
.. _2: https://github.com/zarr-developers/zarr-python
.. _3: https://github.com/zarr-developers/numcodecs
.. _4: https://www.hdfgroup.org/solutions/hdf5/
.. _5: https://pangeo.io/
.. _6: https://pangeo.io/catalog.html
.. _7: https://www.malariagen.net/
.. _8: http://alimanfoo.github.io/2016/09/21/genotype-compression-benchmark.html
.. _9: https://www.humancellatlas.org/
.. _10: https://github.com/constantinpape/z5
.. _11: https://github.com/lasersonlab/ndarray.scala
.. _12: https://github.com/meggart/ZarrNative.jl
.. _13: https://github.com/saalfeldlab/n5
.. _14: http://docs.dask.org/en/latest/array-creation.html
.. _15: http://xarray.pydata.org/en/stable/io.html
.. _16: https://github.com/ContinuumIO/intake-xarray
.. _17: http://matthewrocklin.com/blog/work/2018/01/22/pangeo-2
.. _18: http://wiki.esipfed.org/index.php/Interoperability_and_Technology/Tech_Dive_Webinar_Series#8_March.2C_2018:_.22Zarr:_A_simple.2C_open.2C_scalable_solution_for_big_NetCDF.2FHDF_data_on_the_Cloud.22:_Alistair_Miles.2C_University_of_Oxford.
.. _19: https://agu.confex.com/agu/fm18/meetingapp.cgi/Paper/390015


Authors
-------

Project contributors are listed in alphabetical order by surname.

* `Ryan Abernathey <https://github.com/rabernat>`_, Columbia University
* `Stephan Balmer <https://github.com/sbalmer>`_, Meteotest
* `Ambrose Carr <https://github.com/ambrosejcarr>`_, Chan Zuckerberg Initiative
* `Tim Crone <https://github.com/tjcrone>`_, Columbia University
* `Martin Durant <https://github.com/martindurant>`_, Anaconda, inc.
* `Jan Funke <https://github.com/funkey>`_, HHMI Janelia
* `Darren Gallagher <https://github.com/dazzag24>`_, Satavia
* `Fabian Gans <https://github.com/meggart>`_, Max Planck Institute for Biogeochemistry
* `Shikhar Goenka <https://github.com/shikharsg>`_, Satavia
* `Joe Hamman <https://github.com/jhamman>`_, NCAR
* `Stephan Hoyer <https://github.com/shoyer>`_, Google
* `Jerome Kelleher <https://github.com/jeromekelleher>`_, University of Oxford
* `John Kirkham <https://github.com/jakirkham>`_, HHMI Janelia
* `Alistair Miles <https://github.com/alimanfoo>`_, University of Oxford
* `Josh Moore <https://github.com/joshmoore>`_, University of Dundee
* `Charles Noyes <https://github.com/CSNoyes>`_, University of Southern California
* `Tarik Onalan <https://github.com/onalant>`_
* `Constantin Pape <https://github.com/constantinpape>`_, University of Heidelberg
* `Zain Patel <https://github.com/mzjp2>`_, University of Cambridge
* `Matthew Rocklin <https://github.com/mrocklin>`_, NVIDIA
* `Stephan Saafeld <https://github.com/axtimwalde>`_, HHMI Janelia
* `Vincent Schut <https://github.com/vincentschut>`_, Satelligence
* `Justin Swaney <https://github.com/jmswaney>`_, MIT
* `Ryan Williams <https://github.com/ryan-williams>`_, Chan Zuckerberg Initiative
