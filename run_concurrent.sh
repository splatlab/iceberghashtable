#! /usr/bin/bash

size=$1

echo "insert negative positive remove"

set -x

for i in $(seq 1 16)
do
  rm -rf /mnt/pmem0/*
  taskset -c 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 ./main $size $i -b
done

for i in $(seq 17 32)
do
  rm -rf /mnt/pmem0/*
  ./main $size $i -b
done
