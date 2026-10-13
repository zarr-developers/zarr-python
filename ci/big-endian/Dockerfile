# Test image for a big-endian host (s390x, run under QEMU by .github/workflows/big-endian.yml).
# Debian packages supply numpy and numcodecs, so nothing heavy compiles under emulation;
# this is also the environment Debian tests zarr in. Debian does not package Google's
# crc32c library, so it is built here for the google-crc32c C extension.
FROM debian:sid

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        build-essential cmake git python3 python3-dev python3-pip python3-venv \
        python3-crc32c python3-donfig python3-fsspec python3-hatch-vcs python3-hatchling \
        python3-hypothesis python3-numcodecs python3-numpy python3-numpydoc python3-packaging \
        python3-pytest python3-pytest-asyncio python3-pytest-benchmark python3-pytest-xdist \
        python3-tomlkit python3-typer python3-typing-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -q --depth 1 --branch 1.1.2 https://github.com/google/crc32c /tmp/crc32c \
    && cmake -S /tmp/crc32c -B /tmp/crc32c/build -DBUILD_SHARED_LIBS=1 \
        -DCRC32C_BUILD_TESTS=0 -DCRC32C_BUILD_BENCHMARKS=0 -DCRC32C_USE_GLOG=0 \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && cmake --build /tmp/crc32c/build -j"$(nproc)" \
    && cmake --install /tmp/crc32c/build \
    && ldconfig \
    && rm -rf /tmp/crc32c

RUN python3 -m venv --system-site-packages /venv
ENV PATH=/venv/bin:$PATH
RUN pip install -q --no-binary google-crc32c google-crc32c msgspec pytest-accept
