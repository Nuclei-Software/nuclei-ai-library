#!/bin/env python
import argparse
from collections import OrderedDict

def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as inputf:
        lines = inputf.readlines()

    output_data = OrderedDict()
    header = ["case"]
    record_data = False
    meta_info = ""

    for line in lines:
        line = line.strip()
        if "PERFCSV" in line:
            parts = line.split('PERFCSV')[1].split(',')
            if parts[-1] == "PASS":
                header.append("-".join(parts[1:-1]))
        elif line.startswith("CSV"):
            parts = line.split(',')
            if len(parts) > 2:
                test_key = parts[1].strip()
                parts[2] = parts[2].strip()
                if test_key not in output_data:
                    output_data[test_key] = []
                wslist = [ ' ' for _ in range(len(header) - 2 - len(output_data[test_key])) ]
                if len(wslist) > 0:
                    output_data[test_key].extend(wslist)
                output_data[test_key].append(parts[2])

    with open(output_file_path, 'w') as tempf:
        tempf.write(",".join(header) + "\n")
        for key in output_data:
            tempf.write(key+","+",".join(output_data[key]) + '\n')
    pass

def main():
    parser = argparse.ArgumentParser(description='Process a text file and output the results.')
    parser.add_argument('-i', '--input', type=str, default='perf.log', help='The path to the input text file.')
    parser.add_argument('-o', '--output', type=str, default='perf.csv', help='The path to the output text file.')

    args = parser.parse_args()

    print("Washing data from %s to csv like file %s" % (args.input, args.output))
    process_file(args.input, args.output)


if __name__ == '__main__':
    main()
