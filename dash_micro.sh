#! /usr/bin/bash

set -x

echo "insert negative positive remove"

for i in 1 2 4 8 16
do
  rm -rf /mnt/pmem1/*
  LD_PRELOAD="/home/aconway/iceberghashtable/dash/build/pmdk/src/PMDK/src/nondebug/libpmemobj.so" numactl -N 1 -m 1 ./dash_micro $1 $i -b
done
