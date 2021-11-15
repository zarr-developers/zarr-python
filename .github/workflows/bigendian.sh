#!/usr/bin/env bash

set -e
set -u
set -x

#
# see: https://github.community/t/testing-against-multiple-architectures/17111/6
#
apt-get update -y
apt-get install -y python3
python3 --version
python3 -c "import platform; print(platform.machine())"
python -c "import sys;sys.exit(0 if sys.byteorder=='big' else 1)"
lscpu | grep Endian
echo -n "Test endianness: 0=Big, 1=Little ? "
echo -n I | od -to2 | head -n1 | cut -f2 -d" " | cut -c6
