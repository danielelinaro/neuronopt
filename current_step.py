import os
import sys
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
import cell_utils as cu
import pickle
import json
import time


def inject_current_step(I, delay, dur, swc_file, parameters, mechanisms, cell_name=None, neuron=None, do_plot=False, verbose=False):

    if cell_name is None:
        import random
        cell_name = 'cell_%06d' % random.randint(0,999999)

    cell = cu.Cell(cell_name, swc_file, parameters, mechanisms)
    cell.instantiate()

    if neuron is None:
        h = cu.h
    else:
        h = neuron.h

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
    if cell.n_axonal_sections > 0:
        recorders['Vaxon'].record(cell.morpho.axon[0](0.5)._ref_v)
    else:
        print('The cell has no axon.')
    for sec,dst in zip(cell.morpho.apic,cell.apical_path_lengths):
        if dst[0] >= 100:
            recorders['Vapic'].record(sec(0.5)._ref_v)
            break
    for sec,dst in zip(cell.morpho.dend,cell.basal_path_lengths):
        if dst[0] >= 100:
            recorders['Vbasal'].record(sec(0.5)._ref_v)
            break

    h.cvode_active(1)
    h.tstop = stim.dur + stim.delay + 100

    fmt = lambda now: '%02d:%02d:%02d' % (now.tm_hour,now.tm_min,now.tm_sec)

    start = time.time()
    if verbose:
        print('{}>> I = {} pA started @ {}.'.format(cell_name,stim.amp*1e3,fmt(time.localtime(start))))

    h.run()
    stop = time.time()

    if verbose:
        print('{}>> I = {} pA finished @ {}, f = {} spikes/s elapsed time = {} seconds.'.format(
            cell_name,stim.amp*1e3,fmt(time.localtime(stop)),
            len(recorders['spike_times'])/dur*1e3,stop-start))

    if do_plot:
        plt.figure()
        t = np.array(recorders['t'])
        plt.plot(t,recorders['Vsoma'],'k',label='Soma')
        plt.legend(loc='best')
        plt.ylabel(r'$V_m$ (mV)')
        plt.xlabel('Time (ms)')
        plt.show()
        
    h('forall delete_section()')
    return recorders


def main():
    parser = arg.ArgumentParser(description='Compute the f-I curve of a neuron model.')
    parser.add_argument('I', type=float, action='store', help='current value in pA')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=2000., type=float, help='stimulation duration (default: 2000 ms)')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--params-file', type=str, default=None,
                        help='JSON file containing the parameters of the model', required=True)
    parser.add_argument('-m','--mech-file', type=str, default=None,
                        help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-c','--config-file', type=str, default=None,
                        help='JSON file containing the configuration of the model')
    parser.add_argument('-n','--cell-name', type=str, default=None,
                        help='name of the cell as it appears in the configuration file')
    parser.add_argument('-o','--output', type=str, default='step.pkl', help='output file name (default: step.pkl)')
    parser.add_argument('--plot', action='store_true', help='show a plot (default: no)')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose (default: no)')
    args = parser.parse_args(args=sys.argv[1:])

    if args.mech_file is not None and args.config_file is not None:
        print('--mech-file and --config-file cannot both be present.')
        sys.exit(1)

    if args.config_file is not None and args.cell_name is None:
        print('You must specify --cell-name with --config-file.')
        sys.exit(1)

    parameters = json.load(open(args.params_file,'r'))

    if args.config_file is not None or args.cell_name is not None:
        import utils
        mechanisms = utils.extract_mechanisms(args.config_file, args.cell_name)
    else:
        mechanisms = json.load(open(args.mech_file,'r'))

    rec = inject_current_step(args.I, args.delay, args.dur, args.swc_file,
                              parameters, mechanisms, do_plot=args.plot, verbose=args.verbose)
        
    step = {'I': args.I, 'time': np.array(rec['t']), 'voltage': np.array(rec['Vsoma']), 'spike_times': np.array(rec['spike_times'])}

    pickle.dump(step, open(args.output,'wb'))


if __name__ == '__main__':
    main()
