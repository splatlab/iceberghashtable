#! /usr/bin/bash

ops=67108864

set -x

echo "insert positive negative remove" > tmp

for i in $(seq 1 16)
do
  rm -rf /mnt/pmem0/*
  taskset -c 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 ./test_pmem -e 0 -p $ops -t $i >> tmp
done

#for i in $(seq 17 32)
#do
#  rm -rf /mnt/pmem0/*
#  ./test_pmem -p $ops -t $i >> tmp
#done

grep throughput tmp | awk '{ print $9 }' | xargs -n4 > dash.csv
