
import os
import sys
import pickle
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import ipdb

def compute_mask(rate, I, check_increase, verbose=False):
    n = len(I)
    mask = np.array([False for _ in range(n)])
    prev_good_one = 0
    mask[prev_good_one] = True
    for i in range(1,n):
        good = False
        if rate[prev_good_one] == 0:
            good = True
        else:
            dr = rate[i] - rate[prev_good_one]
            rel_dr = dr / rate[prev_good_one]
            if dr > 0:
                if not check_increase:
                    good = True
                else:
                    # the rate increased
                    if rate[i] < 10 or rel_dr <= 0.75:
                        good = True
            else:
                # the rate decreased
                if rel_dr >= -0.3:
                    good = True
        if good:
            mask[i] = True
            prev_good_one = i
    if verbose:
        if np.all(mask == True):
            print('not masking any rates.')
        else:
            sys.stdout.write('masking rates with I = [')
            for i in range(n):
                if mask[i] == False:
                    sys.stdout.write(' %g' % I[i])
            print(' ] pA.')
    return mask

def compute_mask_old(rate, I, coeff, verbose=False):
    n = len(I)
    mask = np.array([True for _ in range(n)])
    rate_change = np.diff(rate)

    idx, = np.where(rate_change < 0)
    if len(idx) > 0:
        if np.abs(rate_change[idx[0]]) > rate[idx[0]]*coeff:
            if verbose:
                print('masking rates with I >= %g pA.' % I[idx[0]+1])
            mask[idx[0]+1:] = False
        elif len(idx) > 1 and np.all(np.diff(idx) == 1) and idx[-1]+1 == n-1:
            if verbose:
                sys.stdout.write('masking rates with I = (')
                for j in range(len(idx)-1):
                    sys.stdout.write('%g, ' % I[idx[j]+1])
                print('%g) pA.' % I[idx[-1]+1])
            mask[idx+1] = False
        elif len(idx) == 1:
            if idx[-1]+1 == n-1 or np.abs(rate_change[idx[0]]) > rate[idx[0]] * coeff:
                if verbose:
                    print('masking rates with I = %g pA.' % I[idx[0]+1])
                mask[idx[0]+1] = False

    if verbose and np.all(mask == True):
        print('not masking any rates.')

    return mask

def main():
    progname = os.path.basename(sys.argv[0])

    parser = arg.ArgumentParser(description='Mask f-I curves.', prog=progname)
    parser.add_argument('folder', type=str, default='.', nargs='?', action='store', help='folder where data is located (default: .)')
    parser.add_argument('-n', '--num-stds', type=int, default=5, help='number of STDs (default: 5)')
    parser.add_argument('-o', '--output', type=str, default='', help='output file name (default: good_population_[n]_STDs.pkl)')
    parser.add_argument('--no-backup', action='store_true', help='do not make a backup of existing files')
    parser.add_argument('-q', '--quiet', action='store_true', help='be quiet')
    parser.add_argument('-i', '--check-increase', type=str, default='', help='check increase in firing rate')
    args = parser.parse_args(args=sys.argv[1:])

    n_STDs = args.num_stds

    if args.check_increase.lower() in ('y','yes'):
        check_increase = True
    elif args.check_increase.lower() in ('n','no'):
        check_increase = False
    else:
        print('--check-increase must either be "yes" or "no".')
        sys.exit(1)

    if n_STDs <= 0:
        print('The number of STDs must be > 0.')
        sys.exit(2)

    if not os.path.isdir(args.folder):
        print('%s: %s: no such directory.' % args.folder)
        sys.exit(3)

    good_pop_file = os.path.join(args.folder, 'good_population_%d_STDs.pkl' % n_STDs)
    fI_curve_file = os.path.join(args.folder, 'fI_curve_good_population_%d_STDs.pkl' % n_STDs)

    if not os.path.isfile(good_pop_file):
        print('%s: %s: no such file.' % (progname, good_pop_file))
        sys.exit(4)

    if not os.path.isfile(fI_curve_file):
        print('%s: %s: no such file.' % (progname, fI_curve_file))
        sys.exit(5)

    verbose = not args.quiet
    good_pop = pickle.load(open(good_pop_file,'rb'))
    fI_curve = pickle.load(open(fI_curve_file,'rb'))
    I = fI_curve['I']

    n_cells,n_I = fI_curve['no_spikes'].shape
    mask = np.ones((n_cells,n_I))

    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,figsize=(7,3))

    for i in range(n_cells):
        num_spikes = fI_curve['no_spikes'][i]
        sys.stdout.write('[%02d/%02d] number of spikes: ' % (i+1,n_cells))
        mask_a = compute_mask(num_spikes, I, check_increase, verbose)
        inverse_first_isi = fI_curve['inverse_first_isi'][i]
        sys.stdout.write('[%02d/%02d]  inverse 1st ISI: ' % (i+1,n_cells))
        mask_b = compute_mask(inverse_first_isi, I, check_increase, verbose)
        idx, = np.where((mask_a == False) | (mask_b == False))
        mask[i][idx] = np.nan
        ax1.plot(I, num_spikes*mask[i], 'o-', color=cm.jet(i/n_cells), \
                 lw=0.5, markerfacecolor='w', markersize=3)
        ax2.plot(I, inverse_first_isi*mask[i], 'o-', \
                 color=cm.jet(i/n_cells), lw=0.5, markerfacecolor='w', markersize=3)

    fI_curve['mask'] = mask
    if args.output != '':
        output_pkl = os.path.splitext(args.output)[0] + '.pkl'
        output_pdf = os.path.splitext(args.output)[0] + '.pdf'
    else:
        output_pdf = os.path.splitext(fI_curve_file)[0] + '_masked.pdf'
        output_pkl = fI_curve_file

    if not args.no_backup:
        os.rename(fI_curve_file, os.path.splitext(fI_curve_file)[0] + '_unmasked.pkl')
    pickle.dump(fI_curve, open(output_pkl, 'wb'))


    ax1.plot(I, np.nanmean(fI_curve['no_spikes']*mask,axis=0), 'ko-', lw=1, \
             markerfacecolor='w', markersize=5)
    ax1.set_xlabel('Injected current (pA)')
    ax1.set_ylabel('Total number of spikes (1/s)')

    ax2.plot(I, np.nanmean(fI_curve['inverse_first_isi']*mask,axis=0), 'ko-', lw=1, \
             markerfacecolor='w', markersize=5)
    ax2.set_xlabel('Injected current (pA)')
    ax2.set_ylabel('Inverse first ISI (1/s)')

    pos = ax1.get_position()
    pts = pos.get_points()
    pts[0][0] -= 0.025
    pts[1][0] -= 0.025
    pts[0][1] += 0.05
    pos.set_points(pts)
    ax1.set_position(pos)

    pos = ax2.get_position()
    pts = pos.get_points()
    pts[0][0] += 0.025
    pts[0][0] -= 0.025
    pts[0][1] += 0.05
    pos.set_points(pts)
    ax2.set_position(pos)

    plt.savefig(output_pdf)
    plt.show()


if __name__ == '__main__':
    main()
