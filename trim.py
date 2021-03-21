import sys

seq_count = 0
source = sys.argv[1]
destination = sys.argv[2]
max_count = int(sys.argv[3])

"""
Trim source CONLL file up to max_count and save to destination
Example to trim source.txt to 500 sequences and save to destination.txt:
python3 trim.py /path/to/source.txt /path/to/destination.txt 500
"""

with open(source, 'r') as f:
    with open(destination, 'w+') as w:
        for line in f:
            if line == '\n':
                seq_count += 1
            w.write(line)
            if seq_count == max_count:
                break
