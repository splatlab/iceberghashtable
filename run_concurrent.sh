#! /usr/bin/bash

size=$1

echo "insert negative positive remove"

set -x

for i in 1 2 4 8 16
do
  rm -rf /mnt/pmem1/*
  numactl -N 1 -m 1 ./main $size $i -b
done
