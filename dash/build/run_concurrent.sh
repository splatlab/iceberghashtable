#! /usr/bin/bash

ops=63753420

set -x

echo "insert positive negative remove" > tmp

for i in 1 2 4 8 16
do
  rm -rf /mnt/pmem1/*
  numactl -N 1 -m 1 ./test_pmem -p $ops -t $i -i 131072 >> tmp
done

grep throughput tmp | awk '{ print $9 }' | xargs -n4 > dash.csv
