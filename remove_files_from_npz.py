
import os
import sys
import numpy as np

progname = os.path.basename(sys.argv[0])


def usage():
    print(f'usage: {progname} [-F file1,file2,...] [-o outfile] [-f|--force] infile')


if __name__ == '__main__':

    if len(sys.argv) < 2 or sys.argv[1] in ('-h','--help'):
        usage()
        sys.exit(0)

    files_to_remove = ['time', 'Vsoma', 'OU_t', 'OU_x']
    force = False
    outfile = None
    i = 1
    nargs = len(sys.argv)
    
    while i < nargs:
        if sys.argv[i] == '-F':
            files_to_remove = sys.argv[i+1].split(',')
            i += 1
        elif sys.argv[i] == '-o':
            outfile = sys.argv[i+1]
            i += 1
        elif sys.argv[i] in ('-f', '--force'):
            force = True
        else:
            break
        i += 1

    if i == nargs:
        usage()
        sys.exit(1)
    
    infile = sys.argv[i]

    if not os.path.isfile(infile):
        print(f'{progname}: {infile}: no such file')
        sys.exit(2)

    if outfile is None:
        outfile = infile

    if os.path.isfile(outfile) and not force:
        print(f'{progname}: {outfile} exists: use -f to force overwrite')
        sys.exit(3)

    data = {k: v for k,v in np.load(infile, allow_pickle=True).items() \
            if k not in files_to_remove}

    np.savez_compressed(outfile, **data)
    
