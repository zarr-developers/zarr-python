#!/usr/bin/env bash

set -e
set -u
set -x

#
# see: https://github.community/t/testing-against-multiple-architectures/17111/6
#

echo =================================
lscpu | grep Endian
echo =================================

export DEBIAN_FRONTEND=${DEBIAN_FRONTEND:-noninteractive}

apt-get update -y
apt-get install -y \
    git \
    python3 \
    python3-setuptools \
    python3-pip \
    python3-pytest \
    python3-msgpack \
    python3-fasteners \
    python3-fsspec \
    python3-numcodecs \
    python3-asciitree

python3 -m pip install .
python3 -m pytest -sv
