#! /usr/bin/bash

size=$1

echo "threads load run"

set -x

for i in 16
do
  rm -rf /mnt/pmem1/*
  numactl -N 1 -m 1 ./ycsb iceberg $1 randint uniform $i
done
