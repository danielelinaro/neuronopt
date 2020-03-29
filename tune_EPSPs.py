
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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares, minimize


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


def cost_sim(tau_rise, tau_decay, amplitude, delay, t_event, neuron, rec, window, synapse_type='excitatory', do_plot=False):
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

    if do_plot:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
        ax1.plot(rec['t'], rec['Vsoma'], 'k', lw=1)
        ax2.plot(t + t_event, v, 'k', lw=1)
        ax2.plot(t + t_event, v_psp, 'r', lw=1)
        plt.show()

    if amplitude is None:
        return v - v_psp

    if synapse_type == 'inhibitory':
        return np.min(v) - np.min(v_psp)

    return np.max(v) - np.max(v_psp)


def fit_PSP_preamble(mode, data_files):
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
    parser.add_argument('segment', type=str, default='basal[0]', nargs='?', action='store',
                        help='Segment where the synapse will be placed (default: basal(0))')
    
    args = parser.parse_args(args=sys.argv[2:])

    if not os.path.isfile(args.swc_file):
        print('{}: {}: no such file.'.format(progname,args.swc_file))
        sys.exit(1)

    if not os.path.isfile(args.params_file):
        print('{}: {}: no such file.'.format(progname,args.params_file))
        sys.exit(1)

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


############################################################
###                       FIT-EPSP                       ###
############################################################


def cost_fit_EPSP(x, synapse, synapse_parameters, EPSP_parameters, delay, t_event, neuron, rec, window, do_plot=False):
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

    return cost_sim(tau_rise, tau_decay, amplitude, delay, t_event, neuron, rec, window, 'excitatory', do_plot)


def fit_EPSP():
    import neuron
    from dlutils import synapse as su

    data_files = {'CTRL': '/Users/daniele/Postdoc/Research/Janelia/in_vitro_data/EPSPs/EPSP_RS_CTRL.pkl', \
                  'TTX': '/Users/daniele/Postdoc/Research/Janelia/in_vitro_data/EPSPs/EPSP_RS_TTX.pkl'}

    cell, seg, output_file, tau_rise, tau_decay, amplitude, \
        parameters, optimize_only_weight, quiet = fit_PSP_preamble('excitatory', data_files)

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
    neuron.h.v_init = -66
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
                                                      rec, window, False) ** 2))
        optim['TTX']['MIN'] = minimize(func, AMPA_parameters_0, bounds = [v for v in bounds_dict['AMPA'].values()],
                                       options = {'maxiter': 100, 'disp': True})
        if not quiet:
            cost_fit_EPSP(optim['TTX']['MIN']['x'], synapse, synapse_parameters, EPSP_parameters,
                          delay, t_event, neuron, rec, window, do_plot=True)
        AMPA_parameters = optim['TTX']['MIN']['x']
    else:
        AMPA_parameters = AMPA_parameters_0

    synapse_parameters['AMPA'] = AMPA_parameters
    synapse_parameters.pop('to_optimize')
    synapse_parameters.pop('weight')
    bounds_dict['AMPA']['weight'] = (0.1, 20.0)
    EPSP_parameters['amplitude'] = amplitude['TTX']
    optim['TTX']['LS'] = least_squares(cost_fit_EPSP, [weight_0], bounds = (0.1, 20.0), \
                                       args = (synapse, synapse_parameters, EPSP_parameters, delay, t_event, \
                                               neuron, rec, window, False), verbose=2)

    if not quiet:
        cost_fit_EPSP(optim['TTX']['LS']['x'], synapse, synapse_parameters, EPSP_parameters, \
                      delay, t_event, neuron, rec, window, do_plot=True)

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
                                                      t_event, neuron, rec, window, False) ** 2))
        optim['CTRL']['MIN'] = minimize(func, NMDA_parameters_0, bounds = [v for v in bounds_dict['NMDA'].values()],
                                        options = {'maxiter': 100, 'disp': True})
        if not quiet:
            cost_fit_EPSP(optim['CTRL']['MIN']['x'], synapse, synapse_parameters, EPSP_parameters, \
                          delay, t_event, neuron, rec, window, do_plot=True)
        NMDA_parameters = optim['CTRL']['MIN']['x']
    else:
        NMDA_parameters = NMDA_parameters_0

    synapse_parameters['NMDA'] = NMDA_parameters
    synapse_parameters.pop('to_optimize')
    synapse_parameters.pop('weight')
    bounds_dict['NMDA']['weight'] = (0.1, 20.0)
    EPSP_parameters['amplitude'] = amplitude['CTRL']

    optim['CTRL']['LS'] = least_squares(cost_fit_EPSP, [weight_0], bounds = (0.1, 20.0), \
                                        args = (synapse, synapse_parameters, EPSP_parameters, delay, t_event, \
                                                neuron, rec, window, False), verbose=2)

    if not quiet:
        cost_fit_EPSP(optim['CTRL']['LS']['x'], synapse, synapse_parameters, EPSP_parameters, \
                      delay, t_event, neuron, rec, window, do_plot=True)

    parameters = {'AMPA_NMDA_ratio': synapse_parameters['ampa_nmda_ratio']}
    for syn_type in ('AMPA','NMDA'):
        parameters[syn_type] = {}
        for j,param_name in enumerate(synapse_parameter_names):
            parameters[syn_type] = {name: synapse_parameters[syn_type][i] for i,name in enumerate(synapse_parameter_names)}
    parameters['AMPA']['weight'] = optim['CTRL']['LS']['x'][0]
    json.dump(parameters, open(output_file, 'w'), indent=4)


############################################################
###                       FIT-IPSP                       ###
############################################################


def cost_fit_IPSP(x, synapse, synapse_parameters, IPSP_parameters, delay, t_event, neuron, rec, window, do_plot=False):
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

    return cost_sim(tau_rise, tau_decay, amplitude, delay, t_event, neuron, rec, window, 'inhibitory', do_plot)


def fit_IPSP():
    import neuron
    from dlutils import synapse as su

    data_files = {'CTRL': '/Users/daniele/Postdoc/Research/Janelia/in_vitro_data/IPSPs/IPSP_RS_CTRL.pkl'}

    cell, seg, output_file, tau_rise, tau_decay, amplitude, \
        parameters, optimize_only_weight, quiet = fit_PSP_preamble('inhibitory', data_files)

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
    neuron.h.v_init = -66
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
                                                      delay, t_event, neuron, rec, window, False) ** 2))
        optim['CTRL']['MIN'] = minimize(func, GABAA_parameters_0, bounds = [v for v in bounds_dict['GABAA'].values()],
                                        options = {'maxiter': 100, 'disp': True})
        if not quiet:
            cost_fit_IPSP(optim['CTRL']['MIN']['x'], synapse, synapse_parameters, IPSP_parameters, delay,
                          t_event, neuron, rec, window, do_plot=True)
        GABAA_parameters = optim['CTRL']['MIN']['x']
    else:
        GABAA_parameters = GABAA_parameters_0

    synapse_parameters['GABAA'] = GABAA_parameters
    synapse_parameters.pop('weight')
    bounds_dict['GABAA']['weight'] = (0.1, 100.0)
    IPSP_parameters['amplitude'] = amplitude['CTRL']
    optim['CTRL']['LS'] = least_squares(cost_fit_IPSP, [weight_0], bounds = (0.1, 100.0), \
                                        args = (synapse, synapse_parameters, IPSP_parameters, delay, t_event, \
                                                neuron, rec, window, False), verbose=2)

    if not quiet:
        cost_fit_IPSP(optim['CTRL']['LS']['x'], synapse, synapse_parameters, IPSP_parameters, \
                      delay, t_event, neuron, rec, window, do_plot=True)

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
    parser.add_argument('pkl_file', type=str, action='store', help='Data file')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')

    args = parser.parse_args(args=sys.argv[2:])

    f_in = args.pkl_file
    if not os.path.isfile(f_in):
        print('{}: {}: no such file.'.format(progname,f_in))
        sys.exit(0)

    data = pickle.load(open(f_in,'rb'))

    from dlutils import graphics
    graphics.set_rc_defaults()

    fix = False
    if fix:
        nbins = 100
        data['EPSP_amplitudes'] = data['EPSP_amplitudes'][data['EPSP_amplitudes'] < 10]
        data['hist'],data['edges'] = np.histogram(data['EPSP_amplitudes'],nbins,density=True)
        data['binwidth'] = np.diff(data['edges'][:2])[0]
        x = data['edges'][:-1] + data['binwidth']/2
        if data['distr_name'] == 'normal':
            p0 = [np.mean(data['EPSP_amplitudes']),np.std(data['EPSP_amplitudes'])]
            data['popt'],pcov = curve_fit(normal,x,data['hist'],p0)
        else:
            p0 = [np.mean(np.log(data['EPSP_amplitudes'])),np.std(np.log(data['EPSP_amplitudes']))]
            data['popt'],pcov = curve_fit(lognormal,x,data['hist'],p0)
        pickle.dump(data,open(f_in,'wb'))

    if args.output is None:
        f_out = f_in.split('.pkl')[0] + '.pdf'
    else:
        f_out = args.output

    x = data['edges'][:-1] + data['binwidth']/2
    fig = plt.figure(figsize=(4,3))
    ax = plt.axes([0.15,0.175,0.8,0.725])
    ax.bar(x, data['hist'], width=data['binwidth'], facecolor='k', edgecolor='w')
    if data['distr_name'] == 'normal':
        ax.plot(data['edges'],normal(data['edges'],data['popt'][0],data['popt'][1]),'r',lw=2)
    else:
        ax.plot(data['edges'],lognormal(data['edges'],data['popt'][0],data['popt'][1]),'r',lw=2)
    ax.set_xlabel('EPSP amplitude (mV)')
    ax.set_ylabel('PDF')
    if 'mu' in data:
        m = data['mu']
        s = data['sigma']
    else:
        m = data['mean']
        s = data['std']
    ax.set_title('mu,sigma = {},{}'.format(m,s))
    plt.savefig(f_out)


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


# time of an incoming spike used to tune synaptic weights
EVENT_TIME = 500
# time window after the arrival of a presynaptic spike (we are interested only in the
# peak, so this does not have to be long)
AFTER_EVENT = 200


def cost(x, cell, synapse, ampa_nmda_ratio, amplitude, neuron, return_EPSP=False):

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

    if return_EPSP:
        return np.max(V[idx]) - V0

    return (np.max(V[idx]) - V0) - amplitude


def worker(segment_index, target, dend_type, ampa_nmda_ratio, swc_file, cell_parameters,
           synapse_parameters, mechanisms, replace_axon, add_axon_if_missing, passive_cell):

    import neuron
    from dlutils import cell as cu
    from dlutils import synapse as su

    if not dend_type in ('apical', 'basal'):
        raise Exception('Unknown dendrite type: "{}"'.format(dend_type))

    neuron.h.load_file('stdrun.hoc')
    neuron.h.v_init = -66
    neuron.h.cvode_active(1)

    cell = cu.Cell('CA3_cell_{}_{}'.format(dend_type, segment_index),
                   swc_file, cell_parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive_cell)

    if dend_type == 'basal':
        segment = cell.basal_segments[segment_index]
    else:
        segment = cell.apical_segments[segment_index]

    synapse = su.AMPANMDASynapse(segment['sec'], segment['seg'].x, 0, [0, 0], **synapse_parameters)
    synapse.set_presynaptic_spike_times([EVENT_TIME])

    weight_0 = 0.5
    
    res = least_squares(cost, [weight_0], bounds = (0.01, 10), 
                        args = (cell, synapse, ampa_nmda_ratio, target, neuron),
                        verbose = 2)

    EPSP_amplitude = cost(res['x'], cell, synapse, ampa_nmda_ratio, target, neuron, return_EPSP=True)
    
    neuron.h('forall delete_section()')

    return res['x'][0], EPSP_amplitude


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

    import neuron
    from dlutils import cell as cu

    # the file containing the morphology of the cell
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

    if args.trial_run:
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

