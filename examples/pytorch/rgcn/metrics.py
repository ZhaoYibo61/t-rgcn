# calculate the metrics
import argparse
import numpy as np
import glob

def parse_line(line):
    fields = line.split('|')
    run = int(fields[1].split(':')[-1].lstrip().rstrip())
    acc = float(fields[2].split(':')[-1].lstrip().rstrip())
    return run,acc

def get_mean_std(filename):
    accs = []
    with open(filename) as fp:
        for line in fp:
            if 'togrep' in line:
                run, acc = parse_line(line)
                accs.append(acc)
    return np.mean(accs), np.std(accs)

def print_mean_std(filenames):
    files = glob.glob(filenames)
    for filename in files:
        mean, std = get_mean_std(filename)
        print('File : {}, Mean: {}, Std : {}'.format(filename, mean, std))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metrics')
    parser.add_argument("-f","--files", type=str, default='',
                        help="files glob")
    args = parser.parse_args()
    print_mean_std(args.files)

