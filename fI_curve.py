#!/usr/bin/env python

import os
import sys
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
import cell_utils as cu

def main():
    parser = arg.ArgumentParser(description='Compute the f-I curve of a neuron model.')
    parser.add_argument('I', type=str, action='store', help='current values in pA, either comma separated or interval and steps, as in 100:300:50')
    parser.add_argument('-p','--params-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-m','--mech-file', type=str, help='JSON file containing the mechanisms to be inserted into the cell', required=True)
    parser.add_argument('--cell-name', default='MyCell', type=str, help="cell name (default: 'MyCell')")
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=2000., type=float, help='stimulation duration (default: 2000 ms)')
    parser.add_argument('--tran', default=200., type=float, help='transient to be discard after stimulation onset (default: 200 ms)')
    args = parser.parse_args(args=sys.argv[1:])

    try:
        I = np.array([float(args.I)])
    except:
        if ',' in args.I:
            I = np.sort(np.array(map(lambda x: float(x), args.I.split(','))))
        elif ':' in args.I:
            tmp = np.array(map(lambda x: float(x), args.I.split(':')))
            I = np.arange(tmp[0],tmp[1]+tmp[2]/2,tmp[2])
        else:
            print('Unknown current definition: %s.' % args.I)
            sys.exit(1)

    # convert to nA
    I = I*1e-3

    cell = cu.Cell(args.cell_name,{'morphology': args.swc_file,
                                   'mechanisms': args.mech_file,
                                   'parameters': args.params_file})
    cell.instantiate()

    stim = h.IClamp(cell.morpho.soma[0](0.5))
    stim.delay = args.delay
    stim.dur = args.dur

    recorders = {'spike_times': h.Vector()}
    recorders['t'] = h.Vector()
    recorders['t'].record(h._ref_t)
    recorders['v'] = h.Vector()
    recorders['v'].record(cell.morpho.soma[0](0.5)._ref_v)
    apc = h.APCount(cell.morpho.soma[0](0.5))
    apc.record(recorders['spike_times'])

    spike_times = []

    h.cvode_active(1)
    h.tstop = stim.dur + stim.delay + 100

    plt.figure()
    for amp in I:
        stim.amp = amp
        apc.n = 0
        h.t = 0
        h.run()
        spike_times.append(np.array(recorders['spike_times']))
        t = np.array(recorders['t'])
        V = np.array(recorders['v'])
        idx, = np.where((t > args.delay-50) & (t < args.delay+550))
        if amp in [0.15,0.2,0.3]:
            plt.plot(t[idx],V[idx])

    no_spikes = map(lambda x: x.shape[0]/args.dur*1e3, spike_times)
    f = map(lambda x: len(np.where(x > args.delay+args.tran)[0])/(args.dur-args.tran)*1e3, spike_times)
    inverse_first_isi = [1e3/np.diff(t[:2]) if len(t) > 1 else 0 for t in spike_times]
    inverse_last_isi = [1e3/np.diff(t[-2:]) if len(t) > 1 else 0 for t in spike_times]

    plt.figure()
    plt.plot(I,no_spikes,'ro-',label='All spikes')
    plt.plot(I,f,'ko-',label='With transient removed')
    plt.plot(I,inverse_first_isi,'bo-',label='Inverse first ISI')
    plt.plot(I,inverse_last_isi,'mo-',label='Inverse last ISI')
    plt.xlabel('Current (nA)')
    plt.ylabel(r'$f$ (spikes/s)')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
