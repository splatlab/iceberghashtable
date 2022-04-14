#!/usr/bin/env python3

import sys
import os
import re
import math

load_ser = "Throughput: load"
run_ser = "Throughput: run"
worka = "workloada,"
workb = "workloadb,"
workc = "workloadc,"

load = []
arun = []
brun = []
crun = []

# Parse driver output
fname = sys.argv[1]
def ser_app(line, ser, num, idx):
    x = re.search(ser, line)
    if x:
        l = line.split(' ')
        n = l[idx].strip()
        n = float(n) * 1000000
        num.append(n)

with open(fname) as f:
    line = f.readline()
    while line != "":
        if (line.split(' ')[1] == worka):
            line = f.readline()
            # ser_app(line, load_ser, load, 2) 
            line = f.readline()
            ser_app(line, run_ser, arun, 2) 
        elif (line.split(' ')[1] == workb):
            line = f.readline()
            # ser_app(line, load_ser, load, 2) 
            line = f.readline()
            ser_app(line, run_ser, brun, 2) 
        elif (line.split(' ')[1] == workc):
            line = f.readline()
            ser_app(line, load_ser, load, 2) 
            line = f.readline()
            ser_app(line, run_ser, crun, 2) 
        line = f.readline()

# print(load)
# print(arun)
# print(brun)
# print(crun)

# Print output file
opfname = sys.argv[2]
opf = open(opfname, "w")
opf.write("threads\t\tload\t\ta\t\tb\t\tc\n")
idx = 0
for l, a, b, c in zip(load, arun, brun, crun):
    thr = int(math.pow(2, idx))
    opf.write(str(thr) + "\t\t" + str(l) + "\t\t" + str(a) + "\t\t" + str(b) + "\t\t" + str(c) + "\n")
    idx = idx+1

opf.close()

