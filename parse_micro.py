#!/usr/bin/env python3

import sys
import os
import re
import math

ins_ser = "Insertions:"
neg_ser = "Negative queries:"
pos_ser = "Positive queries:"
rem_ser = "Removals:"

ins = []
neg = []
pos = []
rem = []

# Parse driver output
fname = sys.argv[1]
def ser_app(line, ser, num, idx):
    x = re.search(ser, line)
    if x:
        l = line.split(' ')
        num.append(l[idx].strip())

with open(fname) as f:
    line = f.readline()
    while line != "":
        ser_app(line, ins_ser, ins, 1) 
        ser_app(line, neg_ser, neg, 2) 
        ser_app(line, pos_ser, pos, 2) 
        ser_app(line, rem_ser, rem, 1) 
        line = f.readline()

# Print output file
opfname = sys.argv[2]
opf = open(opfname, "w")
opf.write("threads\t\tinsert\t\tnegative\t\tpositive\t\tremove\n")
idx = 0
for i, n, p, r in zip(ins, neg, pos, rem):
    thr = int(math.pow(2, idx))
    opf.write(str(thr) + "\t\t" + i + "\t\t" + n + "\t\t" + p + "\t\t" + r + "\n")
    idx = idx+1

opf.close()

