#1 /usr/bin/bash

maxthreads=32
size=26

echo "insert negative positive remove"

set -x

for i in $(seq 1 16)
do
  taskset -c 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 ./benchmark 26 $i
done

for i in $(seq 17 32)
do
  ./benchmark 26 $i
done
