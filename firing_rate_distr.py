
import os
import sys
import json
import pickle
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import dlutils as dl

import neuron
from current_step import inject_current_step

use_scoop = True
if use_scoop:
    try:
        from scoop import futures
        map_fun = futures.map
    except:
        map_fun = map
else:
    map_fun = map


if __name__ == '__main__':

    parser = arg.ArgumentParser(description='Compute the f-I curve of a neuron model.')
    parser.add_argument('I', type=str, action='store', help='current values in pA, either comma separated or interval and steps, as in 100:300:50')
    parser.add_argument('-f','--swc-file', type=str, required=True,
                        help='SWC file defining the cell morphology')
    parser.add_argument('-c','--config-file', type=str, default='', required=True,
                        help='JSON file(s) containing the configuration')
    parser.add_argument('-C', '--condition', type=str, default='', required=True,
                        help='Experimental condition')
    parser.add_argument('-n','--cell-name', type=str, default='', required=True,
                        help='cell name, if the mechanisms are stored in new style format')
    parser.add_argument('-P','--pickle-file', type=str, default='', required=True,
                        help='Pickle file containing the parameters of a population of individuals')
    parser.add_argument('-o','--output', type=str, default='firing_rates.pkl', help='Output file name')
    parser.add_argument('-s','--swap', type=str, default='', help='parameters to swap')
    parser.add_argument('-A','--save-all', action='store_true', help='save also voltage traces')
    parser.add_argument('-R','--replace-axon', type=str, default='no',
                        help='whether to replace the axon (accepted values: "yes" or "no", default "no")')
    parser.add_argument('-A', '--add-axon-if-missing', type=str, default='no',
                        help='whether add an axon if the cell does not have one (accepted values: "yes" or "no", default "no")')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=1000., type=float, help='stimulation duration (default: 1000 ms)')
    parser.add_argument('--tran', default=0., type=float, help='transient to be discard after stimulation onset (default: 0 ms)')
    args = parser.parse_args(args=sys.argv[1:])

    try:
        I = np.array([float(args.I)])
    except:
        if ',' in args.I:
            I = np.sort([float(x) for x in args.I.split(',')])
        elif ':' in args.I:
            tmp = [float(x) for x in args.I.split(':')]
            I = np.arange(tmp[0],tmp[1]+tmp[2]/2,tmp[2])
        else:
            print('Unknown current definition: %s.' % args.I)
            sys.exit(1)

    if args.replace_axon.lower() in ('y','yes'):
        replace_axon = True
    elif args.replace_axon.lower() in ('n','no'):
        replace_axon = False
    else:
        print('Unknown value for --replace-axon: "{}".'.format(args.replace_axon))
        sys.exit(2)

    if args.add_axon_if_missing.lower() in ('y','yes'):
        add_axon_if_missing = True
    elif args.add_axon_if_missing.lower() in ('n','no'):
        add_axon_if_missing = False
    else:
        print('Unknown value for --add-axon-if-missing: "{}".'.format(args.add_axon_if_missing))
        sys.exit(3)

    cell_name = args.cell_name
    condition = args.condition
    mechanisms = dl.extract_mechanisms(args.config_file, cell_name)

    config = json.load(open(args.config_file,'r'))[cell_name]
    data = pickle.load(open(args.pickle_file,'rb'))
    params = data['populations'][condition][cell_name]
    evaluator = data['evaluators'][condition]
    param_names = evaluator.param_names
    default_parameters = None
    if len(args.swap) > 0:
        s = args.swap.split('/')
        other_condition = s[0]
        other_cell_name = s[1]
        mech_names = s[2].split(',')
        other_params = data['populations'][other_condition][other_cell_name]
        n_individuals = min(params.shape[0], other_params.shape[0])
        params = params[:n_individuals,:]
        other_params = other_params[:n_individuals,:]
        for mech_name in mech_names:
            try:
                idx = param_names.index(mech_name)
            except:
                idx, = np.where(list(map(lambda name: mech_name in name, param_names)))
                if len(idx) == 1:
                    idx = idx[0]
                else:
                    import ipdb
                    ipdb.set_trace()
            avg = np.mean(params[:,idx])
            other_avg = np.mean(other_params[:,idx])
            params[:,idx] *= (other_avg / avg)
            print('Using values from cell {} for parameter {} (index {}).'.format(other_cell_name, mech_name, idx))
    population = dl.build_parameters_dict(params, evaluator, config, None)

    dur = args.dur
    delay = args.delay
    tran = args.tran

    worker = lambda individual: inject_current_step(I, delay, dur, args.swc_file, individual, \
                                                    mechanisms, replace_axon, add_axon_if_missing, \
                                                    cell_name=None, neuron=neuron, do_plot=False, verbose=False)
    runs = list(map_fun(worker, population))
    neuron.h('forall delete_section()')

    spike_times = [np.array(run['spike_times']) for run in runs]
    no_spikes = [len(x)/dur*1e3 for x in spike_times]
    f = [len(x[(x>delay+tran) & (x<delay+dur)])/(dur-tran)*1e3 for x in spike_times]
    inverse_first_isi = [1e3/np.diff(t[:2]) if len(t) > 1 else 0 for t in spike_times]
    inverse_last_isi = [1e3/np.diff(t[-2:]) if len(t) > 1 else 0 for t in spike_times]

    data = {'delay': delay, 'dur': dur, 'tran': tran,
            'I': I, 'spike_times': spike_times,
            'f': f, 'no_spikes': no_spikes,
            'inverse_first_isi': inverse_first_isi,
            'inverse_last_isi': inverse_last_isi}

    if args.save_all:
        data['time'] = [np.array(run['t']) for run in runs]
        data['voltage'] = {'soma': [np.array(run['soma.v']) for run in runs]}

    pickle.dump(data, open(args.output,'wb'))
