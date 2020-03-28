
import os
#import re
import sys
import time
#import glob
import pickle
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import find_peaks
from scipy.optimize import curve_fit, least_squares, minimize
import json
from itertools import repeat

import neuron
neuron.h.load_file('stdrun.hoc')
neuron.h.cvode_active(1)

from dlutils import cell as cu
from dlutils import synapse as su

#from collections import OrderedDict

EVENT_TIME = 500
AFTER_EVENT = 500

use_scoop = True
if use_scoop:
    try:
        from scoop import futures
        map_fun = futures.map
    except:
        map_fun = map
else:
    map_fun = map


###
#
# Edit the file nrn/src/nrnoc/cabcode.c and increase 
#  #define NSECSTACK 20
# by increasing the nu
# the name of this script
progname = os.path.basename(sys.argv[0])

# definitions of normal and log-normal distribution, for the fit and plot
normal = lambda x,m,s: 1. / (np.sqrt(2 * np.pi * s**2)) * np.exp(-(x - m)**2 / (2 * s**2))
lognormal = lambda x,m,s: 1./ (s * x * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - m)**2 / (2 * s**2))


def cost(x, cell, synapse, ampa_nmda_ratio, amplitude, return_EPSP=False):

    weight = x[0]
    synapse.nc[0].weight[0] = weight
    synapse.nc[1].weight[0] = weight * ampa_nmda_ratio

    recorders = {'t': neuron.h.Vector(), 'V': neuron.h.Vector()}
    recorders['t'].record(neuron.h._ref_t)
    recorders['V'].record(cell.morpho.soma[0](0.5)._ref_v)
    
    neuron.h.tstop = EVENT_TIME + AFTER_EVENT
    neuron.h.run()

    t = np.array(recorders['t'])
    V = np.array(recorders['V'])
    idx, = np.where(t > EVENT_TIME)
    V0 = V[idx[0] - 10]

    #if do_plot:
    #    plt.plot(t, V, 'k')
    #    plt.show()

    if return_EPSP:
        return np.max(V[idx]) - V0

    return (np.max(V[idx]) - V0) - amplitude


def worker(segment_index, target, dend_type, ampa_nmda_ratio, swc_file, cell_parameters,
           synapse_parameters, mechanisms, replace_axon, add_axon_if_missing, passive_cell):

    if not dend_type in ('apical', 'basal'):
        raise Exception('Unknown dendrite type: "{}"'.format(dend_type))

    cell = cu.Cell('CA3_cell_{}_{}'.format(dend_type, segment_index),
                   swc_file, cell_parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive_cell)

    try:
        if dend_type == 'basal':
            segment = cell.basal_segments[segment_index]
        else:
            segment = cell.apical_segments[segment_index]
    except:
        import ipdb
        ipdb.set_trace()

    synapse = su.AMPANMDASynapse(segment['sec'], segment['seg'].x, 0, [0, 0], **synapse_parameters)
    synapse.set_presynaptic_spike_times([EVENT_TIME])

    weight_0 = 0.5
    
    res = least_squares(cost, [weight_0], bounds = (0.01, 10), 
                        args = (cell, synapse, ampa_nmda_ratio, target),
                        verbose = 2)

    EPSP_amplitude = cost(res['x'], cell, synapse, ampa_nmda_ratio, target, return_EPSP=True)
    
    neuron.h('forall delete_section()')

    return res['x'][0], EPSP_amplitude


if __name__ == '__main__':
    parser = arg.ArgumentParser(description='Tune synaptic weights')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--cell-params', type=str, default=None, required=True,
                        help='JSON file containing the parameters of the cell')
    parser.add_argument('-s','--synapse-params', type=str, default=None, required=True,
                        help='JSON file containing the parameters of the synapses')
    parser.add_argument('-m','--mech-file', type=str, default=None,
                        help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-c','--config-file', type=str, default=None,
                        help='JSON file containing the configuration of the model')
    parser.add_argument('-n','--cell-name', type=str, default=None,
                        help='name of the cell as it appears in the configuration file')
    parser.add_argument('-R','--replace-axon', type=str, default=None,
                        help='whether to replace the axon (accepted values: "yes" or "no")')
    parser.add_argument('-A', '--add-axon-if-missing', type=str, default=None,
                        help='whether to add an axon if the cell does not have one (accepted values: "yes" or "no")')
    parser.add_argument('--model-type', type=str, default='active',
                        help='whether to use a passive or active model (accepted values: "active" (default) or "passive")')
    parser.add_argument('--mean', default=None, type=float, help='mean of the distribution of EPSPs')
    parser.add_argument('--std', default=None, type=float, help='standard deviation of the distribution of EPSPs')
    parser.add_argument('--distr', default=None, type=str, help='type of distribution of the synaptic weights (accepted values are normal or lognormal)')
    parser.add_argument('--ampa-nmda-ratio', default=1., type=float, help='AMPA/NMDA ratio')
    parser.add_argument('--output-dir', default='.', type=str, help='output folder')
    
    args = parser.parse_args(args=sys.argv[1:])

    from dlutils import utils

    if args.mean is None:
        raise ValueError('You must specify the mean of the distribution of synaptic weights')

    if args.std is None:
        raise ValueError('You must specify the standard deviation of the distribution of synaptic weights')

    if args.std < 0:
        raise ValueError('The standard deviation of the distribution of synaptic weights must be non-negative')

    if not args.distr in ('normal','lognormal'):
        raise ValueError('The distribution of synaptic weights must either be "normal" or "lognormal"')

    if args.ampa_nmda_ratio < 0:
        raise ValueError('The AMPA/NMDA ratio must be non-negative')

    if not os.path.isfile(args.swc_file):
        print('{}: {}: no such file.'.format(progname,args.swc_file))
        sys.exit(1)

    if not os.path.isfile(args.cell_params):
        print('{}: {}: no such file.'.format(progname,args.cell_params))
        sys.exit(1)
    cell_parameters = json.load(open(args.cell_params, 'r'))

    if not os.path.isfile(args.synapse_params):
        print('{}: {}: no such file.'.format(progname,args.synapse_params))
        sys.exit(1)
    synapse_parameters = json.load(open(args.synapse_params, 'r'))

    if args.mech_file is not None:
        if not os.path.isfile(args.mech_file):
            print('{}: {}: no such file.'.format(progname,args.mech_file))
            sys.exit(1)
        mechanisms = json.load(open(args.mech_file,'r'))
    elif args.config_file is not None:
        if not os.path.isfile(args.config_file):
            print('{}: {}: no such file.'.format(progname,args.config_file))
            sys.exit(1)
        if args.cell_name is None:
            print('--cell-name must be present with --config-file option.')
            sys.exit(1)
        mechanisms = utils.extract_mechanisms(args.config_file, args.cell_name)
        
    try:
        sim_pars = pickle.load(open('simulation_parameters.pkl','rb'))
    except:
        sim_pars = None

    if args.replace_axon == None:
        if sim_pars is None:
            replace_axon = False
        else:
            replace_axon = sim_pars['replace_axon']
            print('Setting replace_axon = {} as per original optimization.'.format(replace_axon))
    else:
        if args.replace_axon.lower() in ('y','yes'):
            replace_axon = True
        elif args.replace_axon.lower() in ('n','no'):
            replace_axon = False
        else:
            print('Unknown value for --replace-axon: "{}".'.format(args.replace_axon))
            sys.exit(7)

    if args.add_axon_if_missing == None:
        if sim_pars is None:
            add_axon_if_missing = True
        else:
            add_axon_if_missing = not sim_pars['no_add_axon']
            print('Setting add_axon_if_missing = {} as per original optimization.'.format(add_axon_if_missing))
    else:
        if args.add_axon_if_missing.lower() in ('y','yes'):
            add_axon_if_missing = True
        elif args.add_axon_if_missing.lower() in ('n','no'):
            add_axon_if_missing = False
        else:
            print('Unknown value for --add-axon-if-missing: "{}".'.format(args.add_axon_if_missing))
            sys.exit(8)

    if args.model_type == 'passive':
        passive_cell = True
    elif args.model_type == 'active':
        passive_cell = False
    else:
        print('Unknown value for --model-type: "{}". Accepted values are `active` and `passive`.'.format(args.model_type))
        sys.exit(9)


    swc_file = args.swc_file

    # instantiate a cell to see how many basal and apical compartments it contains
    cell = cu.Cell('CA3_cell' , swc_file, cell_parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive_cell)

    # the basal and apical compartments
    segments = {'basal': cell.basal_segments, 'apical': cell.apical_segments}

    # pick the compartments where we will place a synapse:
    good_segments = {}
    # all basal compartments
    good_segments['basal'] = np.arange(len(segments['basal']))
    # do not insert synapses into the apical dendrites that are in SLM: these are the segments that lie within slm_border
    # microns from the distal tip of the dendrites. also, we will not consider those apical branches that have a diameter
    # smaller than 0.5 microns
    slm_border = 100.
    y_coord = np.array([seg['center'][1] for seg in segments['apical']])
    y_limit = np.max(y_coord) - slm_border
    good_segments['apical'] = np.array([i for i,seg in enumerate(segments['apical'])
                                        if (seg['seg'].diam > 0.5 and seg['center'][1] <= y_limit)])

    trial_run = False
    if trial_run:
        N = 10
        good_segments = {dend_type: np.random.choice(good_segments[dend_type], size=N, replace=False) for dend_type in segments}

    # the number of segments for each dendrite
    N_segments = {k: len(v) for k,v in good_segments.items()}

    # the target EPSP amplitudes
    if args.distr == 'normal':
        targets = {dend_type: np.random.normal(loc=args.mean, scale=args.std, size=N_segments[dend_type]) for dend_type in segments}
    else:
        targets = {dend_type: np.random.lognormal(mean=np.log(args.mean), sigma=args.std, size=N_segments[dend_type]) for dend_type in segments}

    # the distances of each compartment from the soma
    distances = {dend_type: np.array([segments[dend_type][i]['dst'] for i in good_segments[dend_type]]) for dend_type in segments}

    # worker functions
    fun_basal = lambda index, target: worker(index, target, 'basal', args.ampa_nmda_ratio, swc_file, cell_parameters, \
                                             synapse_parameters, mechanisms, replace_axon, add_axon_if_missing, passive_cell)
    fun_apical = lambda index, target: worker(index, target, 'apical', args.ampa_nmda_ratio, swc_file, cell_parameters, \
                                             synapse_parameters, mechanisms, replace_axon, add_axon_if_missing, passive_cell)

    neuron.h('forall delete_section()')

    # run the optimizations
    basal = list( map_fun(fun_basal, good_segments['basal'], targets['basal']) )
    apical = list( map_fun(fun_apical, good_segments['apical'], targets['apical']) )

    weights = {'basal': np.array([res[0] for res in basal]), 'apical': np.array([res[0] for res in apical])}
    EPSP_amplitudes = np.concatenate((np.array([res[1] for res in basal]), np.array([res[1] for res in apical])))
    
    nbins = 30
    hist,edges = np.histogram(EPSP_amplitudes, nbins, density=True)
    binwidth = np.diff(edges[:2])[0]
    x = edges[:-1] + binwidth/2

    # fit the distribution of EPSPs amplitudes
    if args.distr == 'normal':
        p0 = [np.mean(EPSP_amplitudes), np.std(EPSP_amplitudes)]
        popt,pcov = curve_fit(normal, x, hist, p0)
    else:
        p0 = [np.mean(np.log(EPSP_amplitudes)), np.std(np.log(EPSP_amplitudes))]
        popt,pcov = curve_fit(lognormal, x, hist, p0)

    # save everything
    now = time.localtime(time.time())
    filename = args.output_dir + '/EPSP_amplitudes_mu={:.3f}_sigma={:.3f}_{}{:02d}{:02d}_{:02d}{:02d}{:02d}.pkl' \
                   .format(args.mean, args.std, now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    data =  {
        'segments_index': good_segments,
        'weights': weights,
        'target_EPSP': targets,
        'mu': args.mean,
        'sigma': args.std,
        'EPSP_amplitudes': EPSP_amplitudes,
        'swc_file': swc_file,
        'mechanisms': mechanisms,
        'cell_parameters': cell_parameters,
        'synapse_parameters': synapse_parameters,
        'distr_name': args.distr,
        'scaling': args.ampa_nmda_ratio,
        'ampa_nmda_ratio': args.ampa_nmda_ratio,
        'slm_border': slm_border,
        'hist': hist,
        'binwidth': binwidth,
        'edges': edges,
        'popt': popt
    }
    pickle.dump(data, open(filename, 'wb'))
    
    fig,(ax1,ax2) = plt.subplots(1, 2)
    col = {'basal': [.2,.2,.2], 'apical': [.8,.2,.2]}
    for dend_type in weights:
        ax1.plot(distances[dend_type], weights[dend_type], 'o', color=col[dend_type], markersize=4, markerfacecolor='w')
        ax2.plot(targets[dend_type], weights[dend_type], 'o', color=col[dend_type], markersize=4, markerfacecolor='w')
    plt.show()
