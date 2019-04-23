#!/usr/bin/env python

import os
import sys
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
import cell_utils as cu
import pickle
import json


def inject_current_step(I, swc_file, mech_file, params_file, delay, dur, cell_name='MyCell', do_plot=False, verbose=False, sim=None):
    cell = cu.Cell(cell_name,{'morphology': swc_file,
                              'mechanisms': mech_file,
                              'parameters': params_file})
    cell.instantiate()

    stim = h.IClamp(cell.morpho.soma[0](0.5))
    stim.delay = delay
    stim.dur = dur
    stim.amp = I*1e-3
    
    recorders = {'spike_times': h.Vector()}
    apc = h.APCount(cell.morpho.soma[0](0.5))
    apc.thresh = -20.
    apc.record(recorders['spike_times'])
    spike_times = []

    for lbl in 't','Vsoma','Vaxon','Vapic','Vbasal':
        recorders[lbl] = h.Vector()
    recorders['t'].record(h._ref_t)
    recorders['Vsoma'].record(cell.morpho.soma[0](0.5)._ref_v)
    try:
        recorders['Vaxon'].record(cell.morpho.axon[4](0.5)._ref_v)
    except:
        print('No axon?')
    for sec,dst in zip(cell.morpho.apic,cell.apical_path_lengths):
        if dst[0] >= 100:
            recorders['Vapic'].record(sec(0.5)._ref_v)
            break
    for sec,dst in zip(cell.morpho.dend,cell.basal_path_lengths):
        if dst[0] >= 100:
            recorders['Vbasal'].record(sec(0.5)._ref_v)
            break

    if sim is None:
        h.cvode_active(1)
        h.tstop = stim.dur + stim.delay + 100
        if verbose:
            print('Simulating I = %g pA.' % (stim.amp*1e3))
        h.run()
    else:
        sim.run(stim.dur + stim.delay + 100)

    if do_plot:
        plt.figure()
        t = np.array(recorders['t'])
        #try:
        #    plt.plot(t,recorders['Vaxon'],'r',label='Axon')
        #except:
        #    pass
        #plt.plot(t,recorders['Vbasal'],'g',label='Basal')
        #plt.plot(t,recorders['Vapic'],'b',label='Apical')
        plt.plot(t,recorders['Vsoma'],'k',label='Soma')
        plt.legend(loc='best')
        plt.ylabel(r'$V_m$ (mV)')
        plt.xlabel('Time (ms)')
        plt.show()
        
    return recorders


def main():
    parser = arg.ArgumentParser(description='Compute the f-I curve of a neuron model.')
    parser.add_argument('I', type=float, action='store', help='current value in pA')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-m','--mech-file', type=str, default='mechanisms.json', help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-p','--params-file', type=str, help='JSON file containing the parameters of the model', required=True)
    parser.add_argument('-o','--output', type=str, default='step.pkl', help='Output file name (default: step.pkl)')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=2000., type=float, help='stimulation duration (default: 2000 ms)')
    parser.add_argument('--plot', action='store_true', help='show a plot (default: no)')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose (default: no)')
    args = parser.parse_args(args=sys.argv[1:])

    rec = inject_current_step(args.I, args.swc_file, args.mech_file, args.params_file, args.delay, args.dur, 'MyCell', args.plot, args.verbose)
    
    step = {'I': args.I, 'time': np.array(rec['t']), 'voltage': np.array(rec['Vsoma']), 'spike_times': np.array(rec['spike_times'])}

    pickle.dump(step, open(args.output,'w'))


if __name__ == '__main__':
    main()
