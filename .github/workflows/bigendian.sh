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

apt-get update -y
apt-get install -y \
    python3 \
    python3-setuptools \
    python3-pip \
    python3-pytest \
    python3-msgpack \
    python3-fasteners

# These are not found in bionic ports
python -m pip install \
    fsspec \
    numcodecs \
    asciitree

python -m pip install .
pytest -sv
