#! /usr/bin/bash

set -x

echo "threads load run"

for i in 1 2 4 8 16
do
  rm -rf /mnt/pmem1/*
  LD_PRELOAD="/home/aconway/iceberghashtable/dash/build/pmdk/src/PMDK/src/nondebug/libpmemobj.so" numactl -N 1 -m 1 ./ycsb dash $1 randint uniform $i
done
