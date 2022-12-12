.. zarr documentation main file, created by
   sphinx-quickstart on Mon May  2 21:40:09 2016.

Zarr
====

Zarr is a format for the storage of chunked, compressed, N-dimensional arrays
inspired by `HDF5 <https://www.hdfgroup.org/HDF5/>`_, `h5py
<https://www.h5py.org/>`_ and `bcolz <https://bcolz.readthedocs.io/>`_.

The project is fiscally sponsored by `NumFOCUS <https://numfocus.org/>`_, a US
501(c)(3) public charity, and development is supported by the
`MRC Centre for Genomics and Global Health <https://www.cggh.org>`_
and the `Chan Zuckerberg Initiative <https://chanzuckerberg.com/>`_.

These documents describe the Zarr Python implementation. More information
about the Zarr format can be found on the `main website <https://zarr.dev>`_.

Highlights
----------

 * Create N-dimensional arrays with any NumPy dtype.
 * Chunk arrays along any dimension.
 * Compress and/or filter chunks using any NumCodecs_ codec.
 * Store arrays in memory, on disk, inside a Zip file, on S3, ...
 * Read an array concurrently from multiple threads or processes.
 * Write to an array concurrently from multiple threads or processes.
 * Organize arrays into hierarchies via groups.

Contributing
------------

Feedback and bug reports are very welcome, please get in touch via
the `GitHub issue tracker <https://github.com/zarr-developers/zarr-python/issues>`_. See
:doc:`contributing` for further information about contributing to Zarr.

Projects using Zarr
-------------------

If you are using Zarr, we would `love to hear about it
<https://github.com/zarr-developers/community/issues/19>`_.

Acknowledgments
---------------

The following people have contributed to the development of Zarr by contributing code,
documentation, code reviews, comments and/or ideas:

:user:`Alistair Miles <alimanfoo>`
:user:`Altay Sansal <tasansal>`
:user:`Anderson Banihirwe <andersy005>`
:user:`Andrew Fulton <andrewfulton9>`
:user:`Andrew Thomas <amcnicho>`
:user:`Anthony Scopatz <scopatz>`
:user:`Attila Bergou <abergou>`
:user:`BGCMHou <BGCMHou>`
:user:`Ben Jeffery <benjeffery>`
:user:`Ben Williams <benjaminhwilliams>`
:user:`Boaz Mohar <boazmohar>`
:user:`Charles Noyes <CSNoyes>`
:user:`Chris Barnes <clbarnes>`
:user:`David Baddeley <David-Baddeley>`
:user:`Davis Bennett <d-v-b>`
:user:`Dimitri Papadopoulos Orfanos <DimitriPapadopoulos>`
:user:`Eduardo Gonzalez <eddienko>`
:user:`Elliott Sales de Andrade <QuLogic>`
:user:`Eric Prestat <ericpre>`
:user:`Eric Younkin <ericgyounkin>`
:user:`Francesc Alted <FrancescAlted>`
:user:`Greggory Lee <grlee77>`
:user:`Gregory R. Lee <grlee77>`
:user:`Ian Hunt-Isaak <ianhi>`
:user:`James Bourbeau <jrbourbeau>`
:user:`Jan Funke <funkey>`
:user:`Jerome Kelleher <jeromekelleher>`
:user:`Joe Hamman <jhamman>`
:user:`Joe Jevnik <llllllllll>`
:user:`John Kirkham <jakirkham>`
:user:`Josh Moore <joshmoore>`
:user:`Juan Nunez-Iglesias <jni>`
:user:`Justin Swaney <jmswaney>`
:user:`Mads R. B. Kristensen <madsbk>`
:user:`Mamy Ratsimbazafy <mratsim>`
:user:`Martin Durant <martindurant>`
:user:`Matthew Rocklin <mrocklin>`
:user:`Matthias Bussonnier <Carreau>`
:user:`Mattia Almansi <malmans2>`
:user:`Noah D Brenowitz <nbren12>`
:user:`Oren Watson <orenwatson>`
:user:`Pavithra Eswaramoorthy <pavithraes>`
:user:`Poruri Sai Rahul <rahulporuri>`
:user:`Prakhar Goel <newt0311>`
:user:`Raphael Dussin <raphaeldussin>`
:user:`Ray Bell <raybellwaves>`
:user:`Richard Scott <RichardScottOZ>`
:user:`Richard Shaw <jrs65>`
:user:`Ryan Abernathey <rabernat>`
:user:`Ryan Williams <ryan-williams>`
:user:`Saransh Chopra <Saransh-cpp>`
:user:`Sebastian Grill <yetyetanotherusername>`
:user:`Shikhar Goenka <shikharsg>`
:user:`Shivank Chaudhary <Alt-Shivam>`
:user:`Stephan Hoyer <shoyer>`
:user:`Stephan Saalfeld <axtimwalde>`
:user:`Tarik Onalan <onalant>`
:user:`Tim Crone <tjcrone>`
:user:`Tobias Kölling <d70-t>`
:user:`Tom Augspurger <TomAugspurger>`
:user:`Tom White <tomwhite>`
:user:`Tommy Tran <potter420>`
:user:`Trevor Manz <manzt>`
:user:`Vincent Schut <vincentschut>`
:user:`Vyas Ramasubramani <vyasr>`
:user:`Zain Patel <mzjp2>`
:user:`gsakkis`
:user:`hailiangzhang <hailiangzhang>`
:user:`pmav99 <pmav99>`
:user:`sbalmer <sbalmer>`

Contents
--------

.. toctree::
    :maxdepth: 2

    installation
    tutorial
    api
    spec
    contributing
    release
    license
    View homepage <https://zarr.dev/>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _NumCodecs: https://numcodecs.readthedocs.io/
