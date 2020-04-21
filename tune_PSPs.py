
########################################################################################
#                                                                                      #
# If this script crashes because of a section stack problem when running tune-weights, #
# edit the file nrn/src/nrnoc/cabcode.c and change the line                            #
#      #define NSECSTACK 20                                                            #
# to increase the maximum number of sections that can be pushed on the stack.          #
# Then, just run `make install` in nrn/src/nrnoc                                       #
#                                                                                      #
########################################################################################

import os
import re
import sys
import time
import json
import pickle
import argparse as arg
from collections import OrderedDict

import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=8)

# the name of this script
progname = os.path.basename(sys.argv[0])


############################################################
###        FIT-EPSP and FIT-IPSP common functions        ###
############################################################


synapse_parameter_names = ['kon', 'koff', 'CC', 'CO', 'Beta', 'Alpha']

default_synapse_parameters = {
    'AMPA':  {'kon':  12.880, 'koff': 6.470, 'CC': 69.970, 'CO':  6.160, 'Beta': 100.63, 'Alpha': 173.040, 'weight':  1.0}, \
    'NMDA':  {'kon':  86.890, 'koff': 0.690, 'CC':  9.640, 'CO':  2.600, 'Beta':   0.68, 'Alpha':   0.079}, \
    'GABAA': {'kon':   5.397, 'koff': 4.433, 'CC': 20.945, 'CO':  1.233, 'Beta': 283.09, 'Alpha': 254.520, 'weight': 10.0}, \
    'AMPA_NMDA_ratio': 2
}


def cost_sim(tau_rise, tau_decay, amplitude, delay, t_event, neuron, rec, window, synapse_type='excitatory', ax=None):
    from scipy.interpolate import interp1d
    from dlutils.numerics import double_exp

    neuron.h.t = 0
    neuron.h.run()
    t = np.array(rec['t'])
    idx, = np.where((t >= t_event) & (t <= t_event + window))
    t = t[idx] - t_event

    v = np.array(rec['Vsoma'])
    v = v[idx] - v[idx[0] - 1]
    # unitary amplitude
    v_psp = double_exp(tau_rise, tau_decay, delay, t)
    if amplitude is not None:
        v_psp *= amplitude

    f = interp1d(t, v)
    f_psp = interp1d(t, v_psp)
    t = np.arange(t[0], t[-1], 0.2)
    v = f(t)
    if amplitude is None:
        if synapse_type == 'inhibitory':
            v /= np.min(v)
        else:
            v /= np.max(v)
    v_psp = f_psp(t)

    if amplitude is None and synapse_type == 'inhibitory':
        v = -v
        v_psp = -v_psp

    if ax is not None:
        T = np.array(rec['t'])
        V = np.array(rec['Vsoma'])
        idx, = np.where(T > t_event - 50)
        ax[0].plot(T[idx] - t_event, V[idx], color=[.4,.4,.4], label='Soma', lw=1)
        if 'Vsyn' in rec:
            V = np.array(rec['Vsyn'])
            ax[0].plot(T[idx] - t_event, V[idx], color=[1,.5,0], label='Dendrite', lw=1)
        ax[0].legend(loc='best')
        ax[0].set_xlabel('Time from stim (ms)')
        ax[0].set_ylabel(r'$V_m$ (mV)')
        ax[1].plot(t, v_psp, 'r', lw=2, label='Experiment')
        ax[1].plot(t, v, 'k', lw=1, label='Model')
        ax[1].legend(loc='best')
        ax[1].set_xlabel('Time from stim (ms)')
        if amplitude is None:
            ax[1].set_ylabel(r'Normalized $V_m$')
        else:
            ax[1].set_ylabel(r'$V_m$ (mV)')

    if amplitude is None:
        return v - v_psp

    if synapse_type == 'inhibitory':
        return np.min(v) - np.min(v_psp)

    return np.max(v) - np.max(v_psp)


def fit_PSP_preamble(mode):
    from dlutils import utils
    from dlutils import cell as cu

    parser = arg.ArgumentParser(description='Optimize EPSP amplitude in a neuron model')
    parser.add_argument('-F','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--params-file', type=str, default=None, required=True,
                        help='JSON file containing the parameters of the model')
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
    parser.add_argument('--model-type', type=str, default='passive',
                        help='whether to use a passive or active model (accepted values: "passive" (default) or "active")')
    parser.add_argument('-i','--input-file', type=str, default=None,
                        help='JSON file to use as initial guess for the optimization')
    parser.add_argument('-o','--output-file', type=str, default=None,
                        help='JSON file where the results of the optimization will be saved')
    parser.add_argument('-f','--force', action='store_true', help='force overwrite of existing output file')
    parser.add_argument('-q','--quiet', action='store_true', help='do not show plots')
    parser.add_argument('-w','--weight-only', action='store_true', help='optimize only weight, not synapse parameters')

    parser.add_argument('--ctrl-file', type=str, required=True, help='CTRL data file')
    if mode == 'excitatory':
        parser.add_argument('--ttx-file', type=str, required=True, help='TTX data file')

    parser.add_argument('segment', type=str, default='basal[0]', nargs='?', action='store',
                        help='Segment where the synapse will be placed (default: basal(0))')
    
    args = parser.parse_args(args=sys.argv[2:])

    if not os.path.isfile(args.swc_file):
        print('{}: {}: no such file.'.format(progname,args.swc_file))
        sys.exit(1)

    if not os.path.isfile(args.params_file):
        print('{}: {}: no such file.'.format(progname,args.params_file))
        sys.exit(1)

    data_files = {'CTRL': args.ctrl_file}
    if mode == 'excitatory':
        data_files['TTX'] = args.ttx_file
    for v in data_files.values():
        if not os.path.isfile(v):
            print('{}: {}: no such file.'.format(v))

    if args.model_type == 'passive':
        passive = True
    elif args.model_type == 'active':
        passive = False
    else:
        print('Unknown value for --model option: "{}".'.format(args.model_type))
        sys.exit(1)

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

    parameters = json.load(open(args.params_file,'r'))

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

    if args.output_file is not None:
        output_file = args.output_file
    else:
        if mode == 'excitatory':
            output_file = 'AMPA_NMDA_synapse_parameters'
        else:
            output_file = 'GABAA_synapse_parameters'
        output_file += '_' + os.path.splitext(args.params_file)[0]
        output_file += '_' + args.segment.replace('[','_').replace(']','')
        output_file += '_' + args.model_type
        if args.weight_only:
            output_file += '_weight_only'
        output_file += '.json'

    if os.path.exists(output_file) and not args.force:
        print('{} exists: use -f to force overwrite.'.format(output_file))
        sys.exit(1)

    cell = cu.Cell('CA3_cell_%d' % int(np.random.uniform()*1e5), args.swc_file, parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive)

    if 'basal' in args.segment:
        segments = cell.basal_segments
    elif 'apical' in args.segment:
        segments = cell.apical_segments
    elif 'axon' in args.segment:
        segments = cell.axonal_segments
    elif 'soma' in args.segment:
        segments = cell.somatic_segments
    else:
        print('Unknown segment type: "{}". Segment type must be one of "somatic", "basal", "apical" or "axonal".'.format(args.segment))
        sys.exit(1)

    idx = int(re.findall(r'\d+', args.segment)[0])
    try:
        segment = segments[idx]
    except:
        print('{} segments do not contain index {} (max value is {}).'.format(args.segment.split('[')[0], idx, len(segments)-1))
        sys.exit(1)

    data = {k: pickle.load(open(v, 'rb')) for k,v in data_files.items()}
    tau_rise = {k: data[k]['tau_rise'] * 1e3 for k in data}
    tau_decay = {k: data[k]['tau_decay'] * 1e3 for k in data}
    amplitude = {k: data[k]['amplitude'] for k in data}

    try:
        parameters = json.load(open(args.input_file, 'r'))
        print('Using parameters stored in {} to initialize the optimization.'.format(args.input_file))
    except:
        if args.input_file is not None:
            print('{} does not exist: using default parameter guesses to initialize the optimization.'.format(args.input_file))
        else:
            print('Using default parameter guesses to initialize the optimization.')
        parameters = default_synapse_parameters

    return cell, segment, output_file, tau_rise, tau_decay, amplitude, parameters, args.weight_only, args.quiet


def make_axes(n_rows, n_cols=2):
    x_offset = 0.15
    x_spacing = 0.125
    y_spacing = 0.125
    x_bound = 0.03
    if n_rows == 1:
        y_offset = 0.25
        y_bound = 0.05
    else:
        y_offset = 0.125
        y_bound = 0.03
    ax_width = (1 - x_offset - (n_cols-1) * x_spacing - x_bound) / n_cols
    ax_height = (1 - y_offset - (n_rows-1) * y_spacing - y_bound) / n_rows
    if n_rows == 1:
        return [
            [plt.axes([x_offset, y_offset, ax_width, ax_height]),
             plt.axes([x_offset+x_spacing+ax_width, y_offset, ax_width, ax_height])]
        ]
    return [
        [plt.axes([x_offset, y_offset+y_spacing+ax_height, ax_width, ax_height]),
         plt.axes([x_offset+x_spacing+ax_width, y_offset+y_spacing+ax_height, ax_width, ax_height])],
        [plt.axes([x_offset, y_offset, ax_width, ax_height]),
         plt.axes([x_offset+x_spacing+ax_width, y_offset, ax_width, ax_height])]
    ]


############################################################
###                       FIT-EPSP                       ###
############################################################


def cost_fit_EPSP(x, synapse, synapse_parameters, EPSP_parameters, delay, t_event, neuron, rec, window, ax=None):
    tau_rise = EPSP_parameters['tau_rise']
    tau_decay = EPSP_parameters['tau_decay']
    amplitude = EPSP_parameters['amplitude'] if 'amplitude' in EPSP_parameters else None

    if len(x) == 1:
        weight = x[0]
        AMPA_pars = synapse_parameters['AMPA']
        NMDA_pars = synapse_parameters['NMDA']
    elif len(x) == 6:
        weight = synapse_parameters['weight']
        if synapse_parameters['to_optimize'] == 'AMPA':
            AMPA_pars = x
            NMDA_pars = synapse_parameters['NMDA']
        elif synapse_parameters['to_optimize'] == 'NMDA':
            NMDA_pars = x
            AMPA_pars = synapse_parameters['AMPA']
        else:
            raise Exception("Unknown synapse type: '{}'".format(synapse_parameters['to_optimize']))
    else:
        raise Exception('Wrong number of parameters')

    ampa_nmda_ratio = synapse_parameters['ampa_nmda_ratio'] if 'ampa_nmda_ratio' in synapse_parameters else 0.0

    # AMPA synapse
    synapse.syn[0].kon      = AMPA_pars[0]
    synapse.syn[0].koff     = AMPA_pars[1]
    synapse.syn[0].CC       = AMPA_pars[2]
    synapse.syn[0].CO       = AMPA_pars[3]
    synapse.syn[0].Beta     = AMPA_pars[4]
    synapse.syn[0].Alpha    = AMPA_pars[5]
    # NMDA synapse
    synapse.syn[1].kon      = NMDA_pars[0]
    synapse.syn[1].koff     = NMDA_pars[1]
    synapse.syn[1].CC       = NMDA_pars[2]
    synapse.syn[1].CO       = NMDA_pars[3]
    synapse.syn[1].Beta     = NMDA_pars[4]
    synapse.syn[1].Alpha    = NMDA_pars[5]
    # synaptic weights:
    # AMPA
    synapse.nc[0].weight[0] = weight
    # NMDA
    synapse.nc[1].weight[0] = weight * ampa_nmda_ratio

    return cost_sim(tau_rise, tau_decay, amplitude, delay, t_event, neuron, rec, window, 'excitatory', ax)


def fit_EPSP():
    import neuron
    from dlutils import synapse as su

    height = 2
    width = 2.5 * height

    cell, seg, output_file, tau_rise, tau_decay, amplitude, \
        parameters, optimize_only_weight, quiet = fit_PSP_preamble('excitatory')

    delay = 1
    t_event = 700
    window = 200

    synapse = su.AMPANMDASynapse(seg['sec'], seg['seg'].x, 0, [0,0], delay)
    synapse.set_presynaptic_spike_times([t_event])

    rec = {'t': neuron.h.Vector(), 'Vsoma': neuron.h.Vector(), 'Vsyn': neuron.h.Vector()}
    rec['t'].record(neuron.h._ref_t)
    rec['Vsoma'].record(cell.morpho.soma[0](0.5)._ref_v)
    rec['Vsyn'].record(seg['seg']._ref_v)

    neuron.h.load_file('stdrun.hoc')
    neuron.h.v_init = -60
    neuron.h.cvode_active(1)

    optim = {}

    AMPA_parameters_0 = [parameters['AMPA'][name] for name in synapse_parameter_names]
    weight_0 = 3.0
    synapse_parameters = {
        'to_optimize': 'AMPA',
        'weight': weight_0,
        'ampa_nmda_ratio': 0,
        'NMDA': [parameters['NMDA'][name] for name in synapse_parameter_names]
    }
    EPSP_parameters = {'tau_rise': tau_rise['TTX'], 'tau_decay': tau_decay['TTX']}
    neuron.h.tstop = t_event + window

    bounds_dict = {'AMPA': OrderedDict(), 'NMDA': OrderedDict()}
    bounds_dict['AMPA']['kon']   = ( 5.0, 100.0)
    bounds_dict['AMPA']['koff']  = ( 0.1,  10.0)
    bounds_dict['AMPA']['CC']    = (10.0, 100.0)
    bounds_dict['AMPA']['CO']    = ( 1.0,  30.0)
    bounds_dict['AMPA']['Beta']  = (50.0, 200.0)
    bounds_dict['AMPA']['Alpha'] = (50.0, 200.0)

    for k,v in bounds_dict['AMPA'].items():
        bounds_dict['NMDA'][k] = v
    bounds_dict['NMDA']['CC']    = (2.00, 100.0)
    bounds_dict['NMDA']['Beta']  = (0.10, 200.0)
    bounds_dict['NMDA']['Alpha'] = (0.01, 200.0)

    optim['TTX'] = {}

    if not optimize_only_weight:
        func = lambda x: np.sqrt(np.sum(cost_fit_EPSP(x, synapse, synapse_parameters,
                                                      EPSP_parameters, delay, t_event, neuron,
                                                      rec, window, None) ** 2))
        optim['TTX']['MIN'] = minimize(func, AMPA_parameters_0, bounds = [v for v in bounds_dict['AMPA'].values()],
                                       options = {'maxiter': 100, 'disp': True})

        AMPA_parameters = optim['TTX']['MIN']['x']

        fig = plt.figure(figsize=(width, height*2))
        ax = make_axes(n_rows=2)
        cost_fit_EPSP(optim['TTX']['MIN']['x'], synapse, synapse_parameters, EPSP_parameters,
                      delay, t_event, neuron, rec, window, ax[0])
    else:
        AMPA_parameters = AMPA_parameters_0
        fig = plt.figure(figsize=(width, height))
        ax = make_axes(n_rows=1)

    synapse_parameters['AMPA'] = AMPA_parameters
    synapse_parameters.pop('to_optimize')
    synapse_parameters.pop('weight')
    bounds_dict['AMPA']['weight'] = (0.1, 20.0)
    EPSP_parameters['amplitude'] = amplitude['TTX']
    optim['TTX']['LS'] = least_squares(cost_fit_EPSP, [weight_0], bounds = (0.1, 20.0), \
                                       args = (synapse, synapse_parameters, EPSP_parameters, delay, t_event, \
                                               neuron, rec, window, None), verbose=2)

    cost_fit_EPSP(optim['TTX']['LS']['x'], synapse, synapse_parameters, EPSP_parameters, \
                  delay, t_event, neuron, rec, window, ax[-1])

    pdf_filename = os.path.splitext(output_file.replace('NMDA_',''))[0] + '.pdf'
    plt.savefig(pdf_filename)

    NMDA_parameters_0 = [parameters['NMDA'][name] for name in synapse_parameter_names]
    weight_0 = 1
    synapse_parameters = {
        'to_optimize': 'NMDA',
        'weight': weight_0,
        'ampa_nmda_ratio': 2.0,
        'AMPA': AMPA_parameters
    }
    EPSP_parameters = {'tau_rise': tau_rise['CTRL'], 'tau_decay': tau_decay['CTRL']}
    neuron.h.tstop = t_event + window

    optim['CTRL'] = {}

    if not optimize_only_weight:
        func = lambda x: np.sqrt(np.sum(cost_fit_EPSP(x, synapse, synapse_parameters, EPSP_parameters, delay,
                                                      t_event, neuron, rec, window, None) ** 2))
        optim['CTRL']['MIN'] = minimize(func, NMDA_parameters_0, bounds = [v for v in bounds_dict['NMDA'].values()],
                                        options = {'maxiter': 100, 'disp': True})

        NMDA_parameters = optim['CTRL']['MIN']['x']

        fig = plt.figure(figsize=(width, height*2))
        ax = make_axes(n_rows=2)
        cost_fit_EPSP(optim['CTRL']['MIN']['x'], synapse, synapse_parameters, EPSP_parameters, \
                      delay, t_event, neuron, rec, window, ax[0])
    else:
        NMDA_parameters = NMDA_parameters_0
        fig = plt.figure(figsize=(width, height))
        ax = make_axes(n_rows=1)

    synapse_parameters['NMDA'] = NMDA_parameters
    synapse_parameters.pop('to_optimize')
    synapse_parameters.pop('weight')
    bounds_dict['NMDA']['weight'] = (0.1, 20.0)
    EPSP_parameters['amplitude'] = amplitude['CTRL']

    optim['CTRL']['LS'] = least_squares(cost_fit_EPSP, [weight_0], bounds = (0.1, 20.0), \
                                        args = (synapse, synapse_parameters, EPSP_parameters, delay, t_event, \
                                                neuron, rec, window, None), verbose=2)

    cost_fit_EPSP(optim['CTRL']['LS']['x'], synapse, synapse_parameters, EPSP_parameters, \
                  delay, t_event, neuron, rec, window, ax[-1])

    pdf_filename = os.path.splitext(output_file.replace('AMPA_',''))[0] + '.pdf'
    plt.savefig(pdf_filename)

    parameters = {'AMPA_NMDA_ratio': synapse_parameters['ampa_nmda_ratio']}
    for syn_type in ('AMPA','NMDA'):
        parameters[syn_type] = {}
        for j,param_name in enumerate(synapse_parameter_names):
            parameters[syn_type] = {name: synapse_parameters[syn_type][i] for i,name in enumerate(synapse_parameter_names)}
    parameters['AMPA']['weight'] = optim['CTRL']['LS']['x'][0]
    json.dump(parameters, open(output_file, 'w'), indent=4)

    if not quiet:
        plt.show()


############################################################
###                       FIT-IPSP                       ###
############################################################


def cost_fit_IPSP(x, synapse, synapse_parameters, IPSP_parameters, delay, t_event, neuron, rec, window, ax=None):
    tau_rise = IPSP_parameters['tau_rise']
    tau_decay = IPSP_parameters['tau_decay']
    amplitude = IPSP_parameters['amplitude'] if 'amplitude' in IPSP_parameters else None

    if len(x) == 1:
        weight = x[0]
        GABAA_pars = synapse_parameters['GABAA']
    elif len(x) == 6:
        weight = synapse_parameters['weight']
        GABAA_pars = x
    else:
        raise Exception('Wrong number of parameters')

    # GABAA synapse
    synapse.syn.kon      = GABAA_pars[0]
    synapse.syn.koff     = GABAA_pars[1]
    synapse.syn.CC       = GABAA_pars[2]
    synapse.syn.CO       = GABAA_pars[3]
    synapse.syn.Beta     = GABAA_pars[4]
    synapse.syn.Alpha    = GABAA_pars[5]
    # synaptic weight:
    synapse.nc.weight[0] = weight

    return cost_sim(tau_rise, tau_decay, amplitude, delay, t_event, neuron, rec, window, 'inhibitory', ax)


def fit_IPSP():
    import neuron
    from dlutils import synapse as su

    height = 2
    width = 2.5 * height

    cell, seg, output_file, tau_rise, tau_decay, amplitude, \
        parameters, optimize_only_weight, quiet = fit_PSP_preamble('inhibitory')

    delay = 1
    t_event = 700
    window = 200

    synapse = su.GABAASynapse(seg['sec'], seg['seg'].x, -73, 0, delay)
    synapse.set_presynaptic_spike_times([t_event])

    rec = {'t': neuron.h.Vector(), 'Vsoma': neuron.h.Vector(), 'Vsyn': neuron.h.Vector()}
    rec['t'].record(neuron.h._ref_t)
    rec['Vsoma'].record(cell.morpho.soma[0](0.5)._ref_v)
    rec['Vsyn'].record(seg['seg']._ref_v)

    neuron.h.load_file('stdrun.hoc')
    neuron.h.v_init = -60
    neuron.h.cvode_active(1)

    optim = {}

    GABAA_parameters_0 = [parameters['GABAA'][name] for name in synapse_parameter_names]
    weight_0 = 3.0
    synapse_parameters = {
        'weight': weight_0,
    }
    IPSP_parameters = {'tau_rise': tau_rise['CTRL'], 'tau_decay': tau_decay['CTRL']}
    neuron.h.tstop = t_event + window

    bounds_dict = {'GABAA': OrderedDict()}
    bounds_dict['GABAA']['kon']   = ( 1.0, 100.0)
    bounds_dict['GABAA']['koff']  = ( 0.1,  10.0)
    bounds_dict['GABAA']['CC']    = (10.0, 100.0)
    bounds_dict['GABAA']['CO']    = ( 1.0,  30.0)
    bounds_dict['GABAA']['Beta']  = (50.0, 400.0)
    bounds_dict['GABAA']['Alpha'] = (50.0, 400.0)

    optim['CTRL'] = {}

    if not optimize_only_weight:
        func = lambda x: np.sqrt(np.sum(cost_fit_IPSP(x, synapse, synapse_parameters, IPSP_parameters,
                                                      delay, t_event, neuron, rec, window, None) ** 2))
        optim['CTRL']['MIN'] = minimize(func, GABAA_parameters_0, bounds = [v for v in bounds_dict['GABAA'].values()],
                                        options = {'maxiter': 100, 'disp': True})

        GABAA_parameters = optim['CTRL']['MIN']['x']

        fig = plt.figure(figsize=(width, height*2))
        ax = make_axes(n_rows=2)
        cost_fit_IPSP(optim['CTRL']['MIN']['x'], synapse, synapse_parameters, IPSP_parameters, delay,
                      t_event, neuron, rec, window, ax[0])
    else:
        GABAA_parameters = GABAA_parameters_0

        fig = plt.figure(figsize=(width, height))
        ax = make_axes(n_rows=1)

    synapse_parameters['GABAA'] = GABAA_parameters
    synapse_parameters.pop('weight')
    bounds_dict['GABAA']['weight'] = (0.1, 100.0)
    IPSP_parameters['amplitude'] = amplitude['CTRL']
    optim['CTRL']['LS'] = least_squares(cost_fit_IPSP, [weight_0], bounds = (0.1, 100.0), \
                                        args = (synapse, synapse_parameters, IPSP_parameters, delay, t_event, \
                                                neuron, rec, window, None), verbose=2)

    cost_fit_IPSP(optim['CTRL']['LS']['x'], synapse, synapse_parameters, IPSP_parameters, \
                  delay, t_event, neuron, rec, window, ax[-1])

    pdf_filename = os.path.splitext(output_file)[0] + '.pdf'
    plt.savefig(pdf_filename)

    parameters = {'GABAA': {name: synapse_parameters['GABAA'][i] for i,name in enumerate(synapse_parameter_names)}}
    parameters['GABAA']['weight'] = optim['CTRL']['LS']['x'][0]
    json.dump(parameters, open(output_file, 'w'), indent=4)


############################################################
###                         PLOT                         ###
############################################################


# definitions of normal and log-normal distribution, for the fit and plot
normal = lambda x,m,s: 1. / (np.sqrt(2 * np.pi * s**2)) * np.exp(-(x - m)**2 / (2 * s**2))
lognormal = lambda x,m,s: 1./ (s * x * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - m)**2 / (2 * s**2))


def plot():
    parser = arg.ArgumentParser(description='Plot the results of the simulation',
                                prog=progname+' plot')
    parser.add_argument('pkl_file', type=str, action='store', help='data file')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')
    parser.add_argument('-s', '--show', action='store_true', help='show plot')

    args = parser.parse_args(args=sys.argv[2:])

    f_in = args.pkl_file
    if not os.path.isfile(f_in):
        print('{}: {}: no such file.'.format(progname,f_in))
        sys.exit(0)

    f_out = args.output

    data = pickle.load(open(f_in,'rb'))

    from dlutils import graphics
    graphics.set_rc_defaults()

    ms = 3
    col = {'apical': [.8,0,0], 'basal': [.3,.3,.3]}
    fig_width = 8
    fig_height = 6

    n_bins = 20

    # a bunch of parameters for the location of the axes
    x_offset = 0.075
    y_offset = 0.1
    hist_height = 0.12
    hist_width = hist_height * fig_height / fig_width
    width_left = 0.4
    height_left = 0.7
    top = y_offset + hist_height * 1.2 + height_left
    spacing = 0.1
    height_right = (top - y_offset - 2*spacing) / 3
    width_right = 1 - (x_offset + hist_width * 1.2 + width_left + spacing + 0.025)

    for key in data['weights']:

        fig = plt.figure(figsize=(fig_width, fig_height))
        if args.output is None:
            fname = os.path.splitext(f_in)[0]
            timestamp = '_'.join([s for s in fname.split('_') if s.isdigit()])
            prefix = fname.split(timestamp)[0][:-1]
            timestamp = ''.join(timestamp.split('_'))
            f_out = '{}_{}_mean={:.1f}_std={:.1f}_{}.pdf'.format(
                prefix, key, data['weights_mean'][key], data['weights_std'][key], timestamp)

        ax = [
            plt.axes([x_offset + hist_width * 1.2, y_offset + hist_height * 1.2, width_left, height_left]),
            plt.axes([x_offset + hist_width * 1.2, y_offset, width_left, hist_height]),
            plt.axes([x_offset, y_offset + hist_height * 1.2, hist_width, height_left]),
            plt.axes([x_offset + hist_width * 1.2 + width_left + spacing,
                      y_offset + 2 * (height_right + spacing),
                      width_right,
                      height_right]),
            plt.axes([x_offset + hist_width * 1.2 + width_left + spacing,
                      y_offset + height_right + spacing,
                      width_right,
                      height_right]),
            plt.axes([x_offset + hist_width * 1.2 + width_left + spacing,
                      y_offset,
                      width_right,
                      height_right])
        ]

        dend_types = data['weights'][key].keys()
        for dend_type in dend_types:
            X = data['somatic_PSP_amplitudes'][key][dend_type]
            Y = data['dendritic_PSP_amplitudes'][key][dend_type]

            scale = 1.025
            for x,y,w in zip(X, Y, data['weights'][key][dend_type]):
                if y > 20:
                    ax[0].plot([x, x*scale], [y, y*scale], linewidth=0.5, color=col[dend_type])
                    ax[0].text(x*scale, y*scale, '{:.1f}'.format(w), fontsize=6, color=col[dend_type],
                               horizontalalignment='left', verticalalignment='bottom')

            ax[0].plot(X, Y, 'o', color=col[dend_type], linewidth=1,
                       markerfacecolor='w', markersize=ms, label=dend_type)
            ax[0].set_xticklabels([])
            ax[0].set_yticklabels([])
            ax[0].set_title('mu = {} mV - sigma = {} mV'.format(data['weights_mean'][key], data['weights_std'][key]))

            hist,edges = np.histogram(X, bins=n_bins)
            dx = np.diff(edges[:2])
            ax[1].bar(edges[:-1], hist, width=0.7*dx, align='edge', color=col[dend_type])
            ax[1].set_xlim(ax[0].get_xlim())
            ax[1].set_xlabel('Somatic {} amplitudes (mV)'.format(key))

            hist,edges = np.histogram(Y, bins=n_bins)
            dy = np.diff(edges[:2])
            ax[2].barh(edges[:-1], hist, height=0.7*dy, align='edge', color=col[dend_type])
            ax[2].set_ylim(ax[0].get_ylim())
            ax[2].set_ylabel('Dendritic {} amplitudes (mV)'.format(key))

            max_weight = data['max_weights'][key]
            delta = max_weight * 0.05
            lim = [-delta, max_weight+delta]
            X = data['distances'][dend_type][data['segments_index'][key][dend_type]]
            Y = data['weights'][key][dend_type]
            ax[3].plot(X, Y, 'o', color=col[dend_type], linewidth=1, markerfacecolor='w', markersize=ms, label=dend_type)
            ax[3].set_xlabel(r'Distance from soma ($\mu$m)')
            ax[3].set_ylabel('Synaptic weight')
            ax[3].set_ylim(lim)
            ax[3].legend(loc='best')

            X = Y
            Y = data['somatic_PSP_amplitudes'][key][dend_type]
            ax[4].plot(X, Y, 'o', color=col[dend_type], linewidth=1, markerfacecolor='w', markersize=ms)
            ax[4].set_xlabel('Synaptic weight')
            ax[4].set_ylabel('s{} ampl (mV)'.format(key))
            ax[4].set_xlim(ax[3].get_ylim())

            Y = data['dendritic_PSP_amplitudes'][key][dend_type]
            ax[5].plot(X, Y, 'o', color=col[dend_type], linewidth=1, markerfacecolor='w', markersize=ms)
            ax[5].set_xlabel('Synaptic weight')
            ax[5].set_ylabel('d{} ampl (mV)'.format(key))
            ax[5].set_xlim(ax[3].get_ylim())

        plt.savefig(f_out)

    if args.show:
        plt.show()



############################################################
###                         HELP                         ###
############################################################


def help():
    if len(sys.argv) > 2 and sys.argv[2] in commands:
        cmd = sys.argv[2]
        sys.argv = [sys.argv[0], cmd, '-h']
        commands[cmd]()
    else:
        print('Usage: {} <command> [<args>]'.format(progname))
        print('')
        print('Available commands are:')
        print('   tune-weights     Tune the weights of synapses on the whole dendritic tree')
        print('   fit-epsp         Fit EPSPs dynamics')
        print('   fit-ipsp         Fit IPSPs dynamics')
        print('   plot             Plot the results')
        print('')
        print('Type \'{} help <command>\' for help about a specific command.'.format(progname))


############################################################
###                     TUNE-WEIGHTS                     ###
############################################################

DEBUG = False

# time of an incoming spike used to tune synaptic weights
EVENT_TIME = 500
# time window after the arrival of a presynaptic spike (we are interested only in the
# peak, so this does not have to be long)
AFTER_EVENT = 200


def cost(x, cell, segment, synapse, ampa_nmda_ratio, amplitude, neuron, return_PSP_amplitudes=False):

    weight = x[0]
    if ampa_nmda_ratio >= 0:
        synapse.nc[0].weight[0] = weight
        synapse.nc[1].weight[0] = weight * ampa_nmda_ratio
    else:
        synapse.nc.weight[0] = weight

    recorders = {'t': neuron.h.Vector(), 'Vsoma': neuron.h.Vector()}
    recorders['t'].record(neuron.h._ref_t)
    recorders['Vsoma'].record(cell.morpho.soma[0](0.5)._ref_v)
    if return_PSP_amplitudes or DEBUG:
        recorders['Vdend'] = neuron.h.Vector()
        recorders['Vdend'].record(segment._ref_v)
    
    neuron.h.v_init = -60
    neuron.h.tstop = EVENT_TIME + AFTER_EVENT
    neuron.h.run()

    t = np.array(recorders['t'])
    Vsoma = np.array(recorders['Vsoma'])
    idx, = np.where(t > EVENT_TIME)
    Vsoma0 = Vsoma[idx[0] - 10]

    peak = np.max if amplitude > 0 else np.min

    if peak(Vsoma[idx]) > 0:
        print('The cell spiked.')
        if DEBUG:
            Vdend = np.array(recorders['Vdend'])
            plt.plot(t[idx], Vsoma[idx], 'k', lw=1)
            plt.plot(t[idx], Vdend[idx], 'r', lw=1)

            weight = x[0] / 2
            synapse.nc[0].weight[0] = weight
            synapse.nc[1].weight[0] = weight * ampa_nmda_ratio

            neuron.h.v_init = Vsoma[idx[0] - 10]
            neuron.h.t = 0
            neuron.h.run()

            t = np.array(recorders['t'])
            Vsoma = np.array(recorders['Vsoma'])
            idx, = np.where(t > EVENT_TIME)
            Vsoma0 = Vsoma[idx[0] - 10]
            Vdend = np.array(recorders['Vdend'])
            plt.plot(t[idx], Vsoma[idx], 'g', lw=1)
            plt.plot(t[idx], Vdend[idx], 'b', lw=1)
            plt.show()

    if not return_PSP_amplitudes:
        # when used as a cost function
        return (peak(Vsoma[idx]) - Vsoma0) - amplitude

    # when used to measure the EPSP amplitude at the soma and at the synapse where the event was localized
    Vdend = np.array(recorders['Vdend'])
    Vdend0 = Vdend[idx[0] - 10]

    if DEBUG:
        plt.plot(t[idx], Vsoma[idx], 'k')
        plt.plot(t[idx], Vdend[idx], 'r')
        plt.show()

    return (peak(Vsoma[idx]) - Vsoma0, peak(Vdend[idx]) - Vdend0)


def worker(segment_index, target, dend_type, max_weight, ampa_nmda_ratio, swc_file, cell_parameters,
           synapse_parameters, mechanisms, replace_axon, add_axon_if_missing, passive_cell):

    import neuron
    from dlutils import cell as cu
    from dlutils import synapse as su

    if not dend_type in ('apical', 'basal'):
        raise Exception('Unknown dendrite type: "{}"'.format(dend_type))

    neuron.h.load_file('stdrun.hoc')
    neuron.h.cvode_active(1)

    cell = cu.Cell('CA3_cell_{}_{}'.format(dend_type, segment_index),
                   swc_file, cell_parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive_cell)

    if dend_type == 'basal':
        segment = cell.basal_segments[segment_index]
    else:
        segment = cell.apical_segments[segment_index]

    if ampa_nmda_ratio >= 0:
        synapse = su.AMPANMDASynapse(segment['sec'], segment['seg'].x, 0, [0, 0], **synapse_parameters)
        weight_0 = 0.1
    else:
        synapse = su.GABAASynapse(segment['sec'], segment['seg'].x, -73, 0, **synapse_parameters)
        weight_0 = 5

    synapse.set_presynaptic_spike_times([EVENT_TIME])

    
    res = least_squares(cost, [weight_0], bounds = (0.01, max_weight),
                        args = (cell, segment['seg'], synapse, ampa_nmda_ratio, target, neuron),
                        verbose = 2)

    PSP_amplitudes = cost(res['x'], cell, segment['seg'], synapse, ampa_nmda_ratio, target,
                          neuron, return_PSP_amplitudes=True)
    
    neuron.h('forall delete_section()')

    return res['x'][0], PSP_amplitudes


############################################################
###                         MAIN                         ###
############################################################


# all the commands currently implemented: tune-weights is implemented by the main function
# because that's required by SCOOP
commands = {'help': help, 'plot': plot, 'fit-epsp': fit_EPSP, 'fit-ipsp': fit_IPSP}


if __name__ == '__main__':

    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        commands['help']()
        sys.exit(0)

    if sys.argv[1] != 'tune-weights':
        if not sys.argv[1] in commands:
            print('{}: {} is not a recognized command. See \'{} --help\'.'.format(progname, sys.argv[1], progname))
            sys.exit(1)
        commands[sys.argv[1]]()
        sys.exit(0)
        
    parser = arg.ArgumentParser(description='Tune synaptic weights')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--cell-params', type=str, default=None, required=True,
                        help='JSON file containing the parameters of the cell')
    parser.add_argument('--exc-syn-pars', type=str, default=None,
                        help='JSON file containing the parameters of the excitatory synapses')
    parser.add_argument('--inh-syn-pars', type=str, default=None,
                        help='JSON file containing the parameters of the inhibitory synapses')
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
    parser.add_argument('--min-basal-diam', default=0.5, type=float, help='minimum basal segments diameter (default 0.5 um)')
    parser.add_argument('--min-apical-diam', default=0.5, type=float, help='minimum apical segments diameter (default 0.5 um)')
    parser.add_argument('--model-type', type=str, default='active',
                        help='whether to use a passive or active model (accepted values: "active" (default) or "passive")')
    parser.add_argument('--exc-mean', default=None, type=float, help='mean of the distribution of EPSPs')
    parser.add_argument('--exc-std', default=None, type=float, help='standard deviation of the distribution of EPSPs')
    parser.add_argument('--exc-distr', default='normal', type=str,
                        help='type of EPSP distribution (accepted values are normal or lognormal)')
    parser.add_argument('--ampa-nmda-ratio', default=1., type=float, help='AMPA/NMDA ratio')
    parser.add_argument('--max-exc-weight', default=10., type=float, help='maximum value of excitatory weight (default: 10)')
    parser.add_argument('--inh-mean', default=None, type=float, help='mean of the distribution of IPSPs')
    parser.add_argument('--inh-std', default=None, type=float, help='standard deviation of the distribution of IPSPs')
    parser.add_argument('--max-inh-weight', default=100., type=float, help='maximum value of inhibitory weight (default: 100)')
    parser.add_argument('--output-dir', default='.', type=str, help='output folder')
    parser.add_argument('--serial', action='store_true', help='do not use SCOOP')
    parser.add_argument('--trial-run', action='store_true', help='only optimize 10 basal and 10 apical synapses')

    args = parser.parse_args(args=sys.argv[2:])

    if args.serial:
        map_fun = map
    else:
        try:
            from scoop import futures
            map_fun = futures.map
        except:
            map_fun = map
            print('SCOOP not found: will run sequentially')

    from dlutils import utils

    ########## DEFINITION OF MODEL CELL
    
    if not os.path.isfile(args.swc_file):
        print('{}: {}: no such file.'.format(progname,args.swc_file))
        sys.exit(1)

    if not os.path.isfile(args.cell_params):
        print('{}: {}: no such file.'.format(progname,args.cell_params))
        sys.exit(1)
    cell_parameters = json.load(open(args.cell_params, 'r'))

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
        print('Loaded optimization parameters.')
    except:
        sim_pars = None
        print('Could not find a file containing optimization parameters.')

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


    ########## WHAT TO OPTIMIZE
    
    optimize_excitatory_weights = False
    optimize_inhibitory_weights = False
    if args.exc_syn_pars is not None:
        optimize_excitatory_weights = True
    if args.inh_syn_pars is not None:
        optimize_inhibitory_weights = True

    if not optimize_excitatory_weights and not optimize_inhibitory_weights:
        print('At least one of --exc-syn-pars and --inh-syn-pars must be specified.')
        sys.exit(1)

    if optimize_excitatory_weights:

        if not os.path.isfile(args.exc_syn_pars):
            print('{}: {}: no such file.'.format(progname,args.exc_syn_pars))
            sys.exit(1)
        excitatory_synapse_parameters = json.load(open(args.exc_syn_pars, 'r'))
        excitatory_synapse_parameters['AMPA'].pop('weight')

        if not args.exc_distr in ('normal','lognormal'):
            raise ValueError('The EPSP distribution must be either "normal" or "lognormal"')

        if args.exc_mean is None or args.exc_mean <= 0:
            raise ValueError('The mean of the EPSP distribution must be > 0')

        if args.exc_std is None or args.exc_std < 0:
            raise ValueError('The standard deviation of the EPSP distribution must be non-negative')

        if args.ampa_nmda_ratio < 0:
            raise ValueError('The AMPA/NMDA ratio must be non-negative')

        if args.max_exc_weight <= 0:
            raise ValueError('The maximum excitatory synaptic weight must be > 0')

    if optimize_inhibitory_weights:

        if not os.path.isfile(args.inh_syn_pars):
            print('{}: {}: no such file.'.format(progname,args.inh_syn_pars))
            sys.exit(1)
        tmp = json.load(open(args.inh_syn_pars, 'r'))
        tmp['GABAA'].pop('weight')
        inhibitory_synapse_parameters = tmp['GABAA']

        if args.inh_mean is None or args.inh_mean >= 0:
            raise ValueError('The mean of the IPSP distribution must be < 0')

        if args.inh_std is None or args.inh_std < 0:
            raise ValueError('The standard deviation of the IPSP distribution must be non-negative')

        if args.max_inh_weight <= 0:
            raise ValueError('The maximum inhibitory synaptic weight must be > 0')


    import neuron
    from dlutils import cell as cu

    # the file containing the morphology of the cell
    swc_file = args.swc_file

    # instantiate a cell to see how many basal and apical compartments it contains
    cell = cu.Cell('CA3_cell' , swc_file, cell_parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive_cell)

    # the basal and apical compartments
    segments = {'basal': cell.basal_segments, 'apical': cell.apical_segments}

    # the distance of each compartment from the soma
    distances = {key: np.array([seg['dst'] for seg in segments[key]]) for key in segments}

    # the compartments where we will place a synapse
    good_segments = {}
    # target EPSPs and IPSPs
    targets = {}
    # the number of segments for each dendrite
    N_segments = {}

    slm_border = 100.

    ########## EXCITATORY EPSPs
    if optimize_excitatory_weights:
        good_segments['EPSP'] = {}
        # only take those basal compartments that have a diameter greater than args.min_basal_diam
        good_segments['EPSP']['basal'] = np.array([i for i,seg in enumerate(segments['basal']) if seg['seg'].diam >= args.min_basal_diam])
        print('{} out of {} basal segments have a diameter greater than {} um.'.format(
            len(good_segments['EPSP']['basal']), len(segments['basal']), args.min_basal_diam))
        # do not insert synapses into the apical dendrites that are in SLM: these are the segments that lie within slm_border
        # microns from the distal tip of the dendrites.
        # additionally, only take those apical branches that have a diameter greater than args.min_apical_diam
        y_coord = np.array([seg['center'][1] for seg in segments['apical']])
        y_limit = np.max(y_coord) - slm_border
        good_segments['EPSP']['apical'] = np.array([i for i,seg in enumerate(segments['apical'])
                                                    if (seg['seg'].diam >= args.min_apical_diam and seg['center'][1] <= y_limit)])
        print('{} out of {} apical segments are within the SLM boundary and have a diameter greater than {} um.'.format(
            len(good_segments['EPSP']['apical']), len(segments['apical']), args.min_apical_diam))
        if args.trial_run:
            N = {dend_type: np.min([10, len(good_segments['EPSP'][dend_type])]) for dend_type in segments}
            good_segments['EPSP'] = {dend_type: np.random.choice(good_segments['EPSP'][dend_type], size=N[dend_type], replace=False)
                                     for dend_type in segments}

        N_segments['EPSP'] = {k: len(v) for k,v in good_segments['EPSP'].items()}

        if args.exc_distr == 'normal':
            targets['EPSP'] = {dend_type: np.random.normal(loc=args.exc_mean, scale=args.exc_std, size=N_segments['EPSP'][dend_type])
                               for dend_type in segments}
        else:
            targets['EPSP'] = {dend_type: np.random.lognormal(mean=np.log(args.exc_mean), sigma=args.exc_std, size=N_segments['EPSP'][dend_type])
                               for dend_type in segments}

        # worker functions
        fun_basal_EPSP = lambda index, target: worker(index, target, 'basal', args.max_exc_weight, args.ampa_nmda_ratio,
                                                      swc_file, cell_parameters, excitatory_synapse_parameters, mechanisms,
                                                      replace_axon, add_axon_if_missing, passive_cell)
        fun_apical_EPSP = lambda index, target: worker(index, target, 'apical', args.max_exc_weight, args.ampa_nmda_ratio,
                                                       swc_file, cell_parameters, excitatory_synapse_parameters, mechanisms,
                                                       replace_axon, add_axon_if_missing, passive_cell)

    ########## INHIBITORY EPSPs
    if optimize_inhibitory_weights:
        good_segments['IPSP'] = {}
        # inhibitory synapses on basal segments that are within 25 um from the soma
        good_segments['IPSP']['basal'] = np.array([i for i,seg in enumerate(segments['basal']) if seg['dst'] <= 25])
        # and on the first two apical sections
        good_segments['IPSP']['apical'] = np.array([i for i,seg in enumerate(segments['apical']) if seg['sec']
                                                    in (cell.morpho.apic[0], cell.morpho.apic[1])])
        if args.trial_run:
            N = {dend_type: np.min([10, len(good_segments['IPSP'][dend_type])]) for dend_type in segments}
            good_segments['IPSP'] = {dend_type: np.random.choice(good_segments['IPSP'][dend_type], size=N[dend_type], replace=False)
                                     for dend_type in segments}

        N_segments['IPSP'] = {k: len(v) for k,v in good_segments['IPSP'].items()}

        targets['IPSP'] = {dend_type: np.random.normal(loc=args.inh_mean, scale=args.inh_std, size=N_segments['IPSP'][dend_type])
                           for dend_type in segments}

        fun_basal_IPSP = lambda index, target: worker(index, target, 'basal', args.max_inh_weight, -1, swc_file, cell_parameters,
                                                      inhibitory_synapse_parameters, mechanisms, replace_axon,
                                                      add_axon_if_missing, passive_cell)
        fun_apical_IPSP = lambda index, target: worker(index, target, 'apical', args.max_inh_weight, -1, swc_file, cell_parameters,
                                                       inhibitory_synapse_parameters, mechanisms, replace_axon,
                                                       add_axon_if_missing, passive_cell)

    neuron.h('forall delete_section()')

    basal = {}
    apical = {}

    # run the optimizations
    if optimize_excitatory_weights:
        basal['EPSP'] = list( map_fun(fun_basal_EPSP, good_segments['EPSP']['basal'], targets['EPSP']['basal']) )
        apical['EPSP'] = list( map_fun(fun_apical_EPSP, good_segments['EPSP']['apical'], targets['EPSP']['apical']) )

    if optimize_inhibitory_weights:
        basal['IPSP'] = list( map_fun(fun_basal_IPSP, good_segments['IPSP']['basal'], targets['IPSP']['basal']) )
        apical['IPSP'] = list( map_fun(fun_apical_IPSP, good_segments['IPSP']['apical'], targets['IPSP']['apical']) )

    weights = {psp: {'basal': np.array([res[0] for res in basal[psp]]), 'apical': np.array([res[0] for res in apical[psp]])}
               for psp in basal}
    
    somatic_PSP_amplitudes = {psp: {'basal':  np.array([res[1][0] for res in basal[psp]]),
                                    'apical': np.array([res[1][0] for res in apical[psp]])}
                              for psp in basal}
    dendritic_PSP_amplitudes = {psp: {'basal':  np.array([res[1][1] for res in basal[psp]]),
                                      'apical': np.array([res[1][1] for res in apical[psp]])}
                                for psp in basal}
    PSP_amplitudes = {psp: np.concatenate(list(somatic_PSP_amplitudes[psp].values())) for psp in basal}
    
    # save everything
    now = time.localtime(time.time())
    filename = args.output_dir + '/synaptic_weights_{}{:02d}{:02d}_{:02d}{:02d}{:02d}.pkl' \
                   .format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    max_weights = {}
    weights_mean = {}
    weights_std = {}
    weights_distr = {}
    synapse_parameters = {}

    if optimize_excitatory_weights:
        max_weights['EPSP'] = args.max_exc_weight
        weights_mean['EPSP'] = args.exc_mean
        weights_std['EPSP'] = args.exc_std
        weights_distr['EPSP'] = args.exc_distr
        synapse_parameters['EPSP'] = excitatory_synapse_parameters
    if optimize_inhibitory_weights:
        max_weights['IPSP'] = args.max_inh_weight
        weights_mean['IPSP'] = args.inh_mean
        weights_std['IPSP'] = args.inh_std
        weights_distr['IPSP'] = 'normal'
        synapse_parameters['IPSP'] = inhibitory_synapse_parameters

    data =  {
        'segments_index': good_segments,
        'weights': weights,
        'max_weights': max_weights,
        'distances': distances,
        'PSP_amplitudes': PSP_amplitudes,
        'somatic_PSP_amplitudes': somatic_PSP_amplitudes,
        'dendritic_PSP_amplitudes': dendritic_PSP_amplitudes,
        'target_PSPs': targets,
        'weights_mean': weights_mean,
        'weights_std': weights_std,
        'weights_distr': weights_distr,
        'swc_file': swc_file,
        'mechanisms': mechanisms,
        'cell_parameters': cell_parameters,
        'synapse_parameters': synapse_parameters,
        'ampa_nmda_ratio': args.ampa_nmda_ratio,
        'slm_border': slm_border,
    }
    pickle.dump(data, open(filename, 'wb'))

