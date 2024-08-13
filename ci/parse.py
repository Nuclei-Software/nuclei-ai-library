#!/bin/env python3
import os
import sys

PROGRAM_UNKNOWN="unknown"
def find_index(key, arr):
    try:
        index = arr.index(key)
    except:
        index = -1
    return index

def parse_benchmark(lines, lgf=""):
    if isinstance(lines, list) == False:
        return PROGRAM_UNKNOWN, PROGRAM_UNKNOWN, None
    subtype = "ai_ops"
    program_type = "nuclei_ai"
    result = dict()
    for line in lines:
        stripline = line.strip()
        if "csv," in stripline.lower():
            csv_values = stripline.split(',')
            if len(csv_values) == 3:
                key = csv_values[1].strip()
                value = csv_values[2].strip()
                result[key] = value
    if len(result) > 0:
        return program_type, subtype, result
    else:
        return PROGRAM_UNKNOWN, PROGRAM_UNKNOWN, None

# simple entry to test on log
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s <run log>" % (sys.argv[0]))
        sys.exit(1)
    runlog = sys.argv[1]
    if os.path.isfile(runlog) == False:
        print("Run log file %s not exit!" % (runlog))
        sys.exit(1)

    rfh = open(runlog)
    lines = rfh.readlines()
    rfh.close()
    print("Parsing benchmark from %s" %(runlog))
    print(parse_benchmark(lines, runlog))
