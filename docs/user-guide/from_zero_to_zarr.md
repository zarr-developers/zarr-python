# From Zero to Zarr

*From a NumPy array to stored bytes, chunk by chunk.*

This page is for people who are new to Zarr. You don't need to know NumPy, HDF5,
or anything about file formats. We begin with *why* Zarr exists, then build up
the *how* one idea at a time, until you understand **how Zarr stores an array**,
**why** that layout is defined by a written specification, and **how a library
turns those stored bytes back into an array you can use**.

This guide comes in three parts:

- **[Part I: The core idea](from_zero_to_zarr_core_idea.md).** The happy path, with
  pictures and no code.
- **[Part II: Under the hood](from_zero_to_zarr_under_the_hood.md).** A few deeper
  sections that go *off* the happy path. Each one is signposted, so you can read
  on or skip ahead.
- **[Part III: Seeing it for real](from_zero_to_zarr_in_action.md).** A short hands-on
  section with runnable code that ties everything together.

But before the *how*, a word on the *why*.

---

## Why we need Zarr

!!! note "In short"
    Modern instruments and simulations produce arrays of numbers far too big to
    fit in memory, and that data needs to be stored durably and shared widely.
    Zarr stores these giant arrays so a reader can cheaply fetch just the piece
    they want, and so the work of reading them can run in parallel across many
    CPU cores.

Across science and industry, our instruments and simulations have become
extraordinary firehoses of numbers. A satellite streams images of the Earth. A
microscope captures gigapixel scans. A gene sequencer reads thousands of genomes.
A climate model writes out temperature and wind for every point on the globe, hour
after hour. In each case the result has the same shape: a vast grid of numbers,
far more than fits in any single computer's memory, and often arriving as a
continuous stream.

That data is worth little sitting on one machine. It has to be stored somewhere
**durable** and **shareable**, so that many people (often scattered across the
world) can read and analyze it. Increasingly that somewhere is **cloud object
storage** (such as Amazon S3, Google Cloud Storage, or Azure Blob Storage): cheap,
effectively unlimited, and reachable from anywhere. But sheer size makes this
hard. Nobody wants to download terabytes just to inspect one corner. What's needed
is a way to store these giant grids so a reader can efficiently and cheaply fetch
**just the piece they want**.

Zarr was built to solve exactly this, though it didn't begin with the cloud. It
grew out of **genomics**. Around 2015, Alistair Miles needed to analyze arrays of
genetic variation across thousands of malaria-carrying mosquitoes (the
[*Anopheles gambiae* 1000 Genomes Project](https://www.malariagen.net/)), arrays
far too big to fit in memory. His real frustration was *speed*, and to see why, it
helps to understand two things the array formats of the day were already doing:
chunking and compression.

First, **chunking**. To store an array bigger than memory, formats like HDF5 and
netCDF already split it into blocks (called *chunks*) and compress each one. That's
what lets you read part of an array without loading all of it: you only fetch and
decompress the chunks that cover the part you want. None of this is Zarr's
invention. Chunking and compression were well-established ideas, and Zarr
deliberately reuses them. The catch with the existing tools was *speed*:
**decompression takes CPU work**, and for a big analysis that scans millions of
values, that work adds up fast.

Here's where speed came in. Reading a chunk means decompressing it, so reading
*many* chunks is a pile of independent decompression jobs, exactly the kind of work
you'd want to spread across all your CPU cores at once. But the tools of the day
wouldn't let him. In Python, the **global interpreter lock** (GIL) limits how much
work threads can do at the same time, so reading through HDF5 couldn't keep all the
cores busy. And the other chunked format he tried could split an array along only
its **first dimension**, while scientific arrays usually have *several* dimensions.
His analyses kept needing pieces that cut across those dimensions, and chunking
along just one of them made that painfully slow. One core did all the work while
the rest sat idle.

So he built Zarr. It didn't introduce new storage concepts so much as **recombine
familiar ones** (chunks, compression, metadata) in a way that frees the CPU cores to
work in parallel: cut an array into chunks across **all its dimensions at once**,
not just one, and decompress them concurrently. Now a read becomes many
chunk-decompressions running **at the same time**, across every core on the machine
(and, with tools like [Dask](https://www.dask.org/), across many machines), so an
analysis that crunches the whole array finishes in a fraction of the time. (He
tells the story in his early
[Zarr blog posts](http://alimanfoo.github.io/2016/05/16/cpu-blues.html).)

Storing data in the cloud came **later**, and turned out to be a superpower.
Because each chunk is simply one key/value entry (as we'll see), Zarr maps
naturally onto object storage like S3, which made it a backbone of cloud-native
science. Today Zarr is used far beyond genomics: in **Earth and climate science**
(satellite imagery and weather and climate model output, via the
[Pangeo](https://www.pangeo.io/) community), **bio-imaging** (huge microscopy
volumes, via [OME-Zarr](https://ngff.openmicroscopy.org/)), **astronomy**, and
**machine learning**, anywhere people wrestle with large, multi-dimensional grids
of numbers.

Strip away the domain (mosquitoes, galaxies, hurricanes) and the object at the
center is always the same: an **array**, a big grid of numbers. So that's where
we'll begin. In [Part I](from_zero_to_zarr_core_idea.md) we'll look at what an array is,
then at what happens when one grows too big to fit in memory, and build up from
there to how Zarr stores it.

---

Continue to **[Part I: The core idea](from_zero_to_zarr_core_idea.md)**.
