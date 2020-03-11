import os
import sys
import time
import glob
import pickle
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, least_squares, minimize
import json
from collections import OrderedDict

# the name of this script
progname = os.path.basename(sys.argv[0])

# definitions of normal and log-normal distribution, for the fit and plot
normal = lambda x,m,s: 1./(np.sqrt(2*np.pi*s**2)) * np.exp(-(x-m)**2/(2*s**2))
lognormal = lambda x,m,s: 1./(s*x*np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-m)**2/(2*s**2))


def simulate_synaptic_activation(swc_file, parameters, mechanisms, replace_axon, add_axon_if_missing, distr_name, mu, sigma, scaling, reps, output_dir='.'):
    """
    Instantiates reps cells and simulates them while recording the membrane potential. EPSPs are then extracted and their
    distribution is computed and fit with an appropriate distribution. Also saves the results to disk.
    """

    from neuron import h
    from dlutils import synapse as su

    # do not insert synapses into the apical dendrites that are in SLM: these are the segments that lie within slm_border
    # microns from the distal tip of the dendrites
    slm_border = 100.
    # beginning of the synaptic activation
    delay = 1000.
    # interval between consecutive synapses being activated
    dt = 250.

    print('(mu,sigma) = ({},{}): '.format(mu,sigma))

    # instantiate cells, synapses and recorders
    cells = []
    synapses = []
    recorders = [h.Vector() for rep in range(reps)]
    for rep in range(reps):
        cell,syn = su.build_cell_with_synapses(swc_file, parameters, mechanisms, replace_axon, \
                                               add_axon_if_missing, distr_name, mu, sigma, \
                                               scaling, slm_border)
        # activate each synapse in a sequential fashion
        t_event = delay
        for syn_group in syn.values():
            for s in syn_group:
                s.set_presynaptic_spike_times([t_event])
                t_event += dt
        cells.append(cell)
        synapses.append(syn)
        recorders[rep].record(cell.morpho.soma[0](0.5)._ref_v)

    # total duration of the simulation
    tend = delay + sum(map(len,syn.values()))*dt
    print('Total duration of the simulation: {:.0f} ms.'.format(tend))

    # run the simulation
    start = time.time()
    now = time.localtime(start)
    sys.stdout.write('{:02d}/{:02d} {:02d}:{:02d}:{:02d} >> '.format(now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec))
    sys.stdout.flush()
    h.cvode_active(1)
    h.tstop = tend
    h.run()
    sys.stdout.write('elapsed time: {:.0f} seconds.\n'.format(time.time()-start))

    # extract the EPSPs from all the recordings
    EPSP_amplitudes = np.array([])
    for rec in recorders:
        V = np.array(rec)
        Vrest = V[-1]
        pks = find_peaks(V,height=Vrest+0.1)[0]
        EPSP_amplitudes = np.concatenate((EPSP_amplitudes, V[pks]-Vrest))
        
    print('Mean EPSP amplitude: {} mV.'.format(np.mean(EPSP_amplitudes)))

    # compute the EPSPs amplitudes histogram
    EPSP_amplitudes = EPSP_amplitudes[EPSP_amplitudes < 20]
    nbins = 100
    hist,edges = np.histogram(EPSP_amplitudes,nbins,density=True)
    binwidth = np.diff(edges[:2])[0]
    x = edges[:-1] + binwidth/2

    # fit the distribution of EPSPs amplitudes
    if distr_name == 'normal':
        p0 = [np.mean(EPSP_amplitudes),np.std(EPSP_amplitudes)]
        popt,pcov = curve_fit(normal,x,hist,p0)
    else:
        p0 = [np.mean(np.log(EPSP_amplitudes)),np.std(np.log(EPSP_amplitudes))]
        popt,pcov = curve_fit(lognormal,x,hist,p0)

    # save everything
    now = time.localtime(time.time())
    filename = 'EPSP_amplitudes_mu=%.3f_sigma=%.3f_%d%02d%02d_%02d%02d%02d' % \
               (mu,sigma,now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)

    data = {'EPSP_amplitudes': EPSP_amplitudes, 'swc_file': swc_file, 'mechanisms': mechanisms, \
            'parameters': parameters, 'distr_name': distr_name, 'mu': mu, 'sigma': sigma, \
            'scaling': scaling, 'slm_border': slm_border, 'hist': hist, 'binwidth': binwidth, \
            'edges': edges, 'popt': popt}
    pickle.dump(data,open(output_dir + '/' + filename + '.pkl','wb'))


############################################################
###                       SIMULATE                       ###
############################################################


def simulate():
    parser = arg.ArgumentParser(description='Simulate synaptic activation in a neuron model.')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
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
                        help='whether add an axon if the cell does not have one (accepted values: "yes" or "no")')
    parser.add_argument('--mean', default=None, type=float, help='mean of the distribution of the synaptic weights')
    parser.add_argument('--std', default=None, type=float, help='standard deviation of the distribution of the synaptic weights')
    parser.add_argument('--distr', default=None, type=str, help='type of distribution of the synaptic weights (accepted values are normal or lognormal)')
    parser.add_argument('--reps', default=10, type=int, help='number of repetitions')
    parser.add_argument('--scaling', default=1., type=float, help='AMPA/NMDA scaling')
    parser.add_argument('--output-dir', default='.', type=str, help='output folder')
    
    args = parser.parse_args(args=sys.argv[2:])

    from dlutils import utils

    if args.mean is None:
        raise ValueError('You must specify the mean of the distribution of synaptic weights')

    if args.std is None:
        raise ValueError('You must specify the standard deviation of the distribution of synaptic weights')

    if args.std < 0:
        raise ValueError('The standard deviation of the distribution of synaptic weights must be non-negative')

    if not args.distr in ('normal','lognormal'):
        raise ValueError('The distribution of synaptic weights must either be "normal" or "lognormal"')

    if args.scaling < 0:
        raise ValueError('The AMPA/NMDA scaling must be non-negative')

    if args.reps <= 0:
        raise ValueError('The number of repetitions must be positive')
    
    if not os.path.isfile(args.swc_file):
        print('{}: {}: no such file.'.format(progname,args.swc_file))
        sys.exit(1)

    if not os.path.isfile(args.params_file):
        print('{}: {}: no such file.'.format(progname,args.params_file))
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

    simulate_synaptic_activation(args.swc_file, parameters, mechanisms, replace_axon, \
                                 add_axon_if_missing, args.distr, args.mean, args.std, \
                                 args.scaling, args.reps, args.output_dir)


############################################################
###                         PLOT                         ###
############################################################


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
    ax.set_xlim([0,6])
    plt.savefig(f_out)


############################################################
###                       FIT-IPSP                       ###
############################################################


def fit_IPSP():
    print('fit_IPSP not implemented yet')


############################################################
###                       FIT-EPSP                       ###
############################################################


default_synapse_parameters = {
    'AMPA':  {'kon':  12.880, 'koff': 6.470, 'CC': 69.970, 'CO':  6.160, 'Beta': 100.63, 'Alpha': 173.040, 'weight':  1.0}, \
    'NMDA':  {'kon':  86.890, 'koff': 0.690, 'CC':  9.640, 'CO':  2.600, 'Beta':   0.68, 'Alpha':   0.079}, \
    'GABAA': {'kon':   5.397, 'koff': 4.433, 'CC': 20.945, 'CO':  1.233, 'Beta': 283.09, 'Alpha': 254.520, 'weight': 10.0}, \
    'AMPA_NMDA_ratio': 2
}


def cost_sim(tau_rise, tau_decay, amplitude, delay, t_event, h, rec, window, do_plot=False):
    from scipy.interpolate import interp1d
    from dlutils.numerics import double_exp

    h.t = 0
    h.run()
    t = np.array(rec['t'])
    idx, = np.where((t >= t_event) & (t <= t_event + window))
    t = t[idx] - t_event

    v = np.array(rec['Vsoma'])
    v = v[idx] - v[idx[0] - 1]
    # unitary amplitude
    v_epsp = double_exp(tau_rise, tau_decay, delay, t)
    if amplitude is not None:
        v_epsp *= amplitude

    f = interp1d(t, v)
    f_epsp = interp1d(t, v_epsp)
    t = np.arange(t[0], t[-1], 0.2)
    v = f(t)
    if amplitude is None:
        v /= np.max(v)
    v_epsp = f_epsp(t)

    if do_plot:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
        ax1.plot(rec['t'], rec['Vsoma'], 'k', lw=1)
        ax2.plot(t + t_event, v, 'k', lw=1)
        ax2.plot(t + t_event, v_epsp, 'r', lw=1)
        plt.show()

    if amplitude is None:
        return v - v_epsp
    return np.max(v) - np.max(v_epsp)


def cost_fit_EPSP(x, synapse, synapse_parameters, EPSP_parameters, delay, t_event, h, rec, window, do_plot=False):
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

    return cost_sim(tau_rise, tau_decay, amplitude, delay, t_event, h, rec, window, do_plot)


def fit_EPSP():
    parser = arg.ArgumentParser(description='Simulate synaptic activation in a neuron model.')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
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
                        help='whether add an axon if the cell does not have one (accepted values: "yes" or "no")')
    
    args = parser.parse_args(args=sys.argv[2:])

    from neuron import h
    from dlutils import utils
    from dlutils import cell as cu
    from dlutils import synapse as su

    if not os.path.isfile(args.swc_file):
        print('{}: {}: no such file.'.format(progname,args.swc_file))
        sys.exit(1)

    if not os.path.isfile(args.params_file):
        print('{}: {}: no such file.'.format(progname,args.params_file))
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

    params_file = 'AMPA_NMDA_synapse_params.json'

    data_files = {'CTRL': '/Users/daniele/Postdoc/Research/Janelia/in_vitro_data/EPSPs/EPSP_RS_CTRL.pkl', \
                  'TTX': '/Users/daniele/Postdoc/Research/Janelia/in_vitro_data/EPSPs/EPSP_RS_TTX.pkl'}
    data = {k: pickle.load(open(v, 'rb')) for k,v in data_files.items()}
    tau_rise = {k: data[k]['tau_rise'] * 1e3 for k in data}
    tau_decay = {k: data[k]['tau_decay'] * 1e3 for k in data}
    amplitude = {k: data[k]['amplitude'] for k in data}

    cell = cu.Cell('CA3_cell_%d' % int(np.random.uniform()*1e5), args.swc_file, parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=True)

    delay = 1
    t_event = 700
    window = {'CTRL': 200, 'TTX': 200}
    
    seg = cell.apical_segments[0]

    synapse = su.AMPANMDASynapse(seg['sec'], seg['seg'].x, 0, [0,0], delay)
       
    synapse.set_presynaptic_spike_times([t_event])

    rec = {'t': h.Vector(), 'Vsoma': h.Vector(), 'Vsyn': h.Vector()}
    rec['t'].record(h._ref_t)
    rec['Vsoma'].record(cell.morpho.soma[0](0.5)._ref_v)
    rec['Vsyn'].record(seg['seg']._ref_v)

    h.v_init = -66
    h.cvode_active(1)

    param_names = ['kon', 'koff', 'CC', 'CO', 'Beta', 'Alpha']
    try:
        parameters = json.load(open(params_file, 'r'))
    except:
        parameters = default_synapse_parameters

    optim = {}

    AMPA_parameters_0 = [parameters['AMPA'][name] for name in param_names]
    weight_0 = 3.0
    synapse_parameters = {
        'to_optimize': 'AMPA',
        'weight': weight_0,
        'ampa_nmda_ratio': 0,
        'NMDA': [parameters['NMDA'][name] for name in param_names]
    }
    EPSP_parameters = {'tau_rise': tau_rise['TTX'], 'tau_decay': tau_decay['TTX']}
    h.tstop = t_event + window['TTX']

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

    ### First step: optimize the parameters of the AMPA synapse using as a target
    ### the normalized EPSP. The weight is set to 3.

    func = lambda x: np.sqrt(np.sum(cost_fit_EPSP(x, synapse, synapse_parameters, EPSP_parameters, delay, t_event, h, rec, window['TTX'], False) ** 2))

    optim['TTX'] = {}
    optim['TTX']['MIN'] = minimize(func, AMPA_parameters_0, bounds = [v for v in bounds_dict['AMPA'].values()], options = {'maxiter': 100, 'disp': True})

    cost_fit_EPSP(optim['TTX']['MIN']['x'], synapse, synapse_parameters, EPSP_parameters, delay, t_event, h, rec, window['TTX'], do_plot=True)

    AMPA_parameters = optim['TTX']['MIN']['x']
    synapse_parameters['AMPA'] = AMPA_parameters
    synapse_parameters.pop('to_optimize')
    synapse_parameters.pop('weight')
    bounds_dict['AMPA']['weight'] = (0.1, 20.0)
    EPSP_parameters['amplitude'] = amplitude['TTX']
    optim['TTX']['LS'] = least_squares(cost_fit_EPSP, [weight_0], bounds = (0.1, 20.0), \
                                       args = (synapse, synapse_parameters, EPSP_parameters, delay, t_event, \
                                               h, rec, window['TTX'], False), verbose=2)

    cost_fit_EPSP(optim['TTX']['LS']['x'], synapse, synapse_parameters, EPSP_parameters, \
                  delay, t_event, h, rec, window['TTX'], do_plot=True)

    NMDA_parameters_0 = [parameters['NMDA'][name] for name in param_names]
    weight_0 = 1
    synapse_parameters = {
        'to_optimize': 'NMDA',
        'weight': weight_0,
        'ampa_nmda_ratio': 2.0,
        'AMPA': AMPA_parameters
    }
    EPSP_parameters = {'tau_rise': tau_rise['CTRL'], 'tau_decay': tau_decay['CTRL']}
    h.tstop = t_event + window['CTRL']

    func = lambda x: np.sqrt(np.sum(cost_fit_EPSP(x, synapse, synapse_parameters, EPSP_parameters, delay, t_event, h, rec, window['TTX'], False) ** 2))

    optim['CTRL'] = {}
    optim['CTRL']['MIN'] = minimize(func, NMDA_parameters_0, bounds = [v for v in bounds_dict['NMDA'].values()], options = {'maxiter': 100, 'disp': True})

    cost_fit_EPSP(optim['CTRL']['MIN']['x'], synapse, synapse_parameters, EPSP_parameters, \
                  delay, t_event, h, rec, window['CTRL'], do_plot=True)


    NMDA_parameters = optim['CTRL']['MIN']['x']
    synapse_parameters['NMDA'] = NMDA_parameters
    synapse_parameters.pop('to_optimize')
    synapse_parameters.pop('weight')
    bounds_dict['NMDA']['weight'] = (0.1, 20.0)
    EPSP_parameters['amplitude'] = amplitude['CTRL']

    optim['CTRL']['LS'] = least_squares(cost_fit_EPSP, [weight_0], bounds = (0.1, 20.0), \
                                        args = (synapse, synapse_parameters, EPSP_parameters, delay, t_event, \
                                                h, rec, window['CTRL'], False), verbose=2)

    cost_fit_EPSP(optim['CTRL']['LS']['x'], synapse, synapse_parameters, EPSP_parameters, \
                  delay, t_event, h, rec, window['CTRL'], do_plot=True)

    parameters = {'AMPA_NMDA_ratio': synapse_parameters['ampa_nmda_ratio']}
    for syn_type in ('AMPA','NMDA'):
        parameters[syn_type] = {}
        for j,param_name in enumerate(param_names):
            parameters[syn_type] = {name: synapse_parameters[syn_type][i] for i,name in enumerate(param_names)}
    parameters['AMPA']['weight'] = optim['CTRL']['LS']['x'][0]
    json.dump(parameters, open(params_file, 'w'), indent=4)



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
        print('   fit-epsp       Fit EPSPs dynamics')
        print('   fit-ipsp       Fit IPSPs dynamics')
        print('   simulate       Simulate synaptic activation')
        print('   plot           Plot the results')
        print('')
        print('Type \'{} help <command>\' for help about a specific command.'.format(progname))


############################################################
###                         MAIN                         ###
############################################################


# all the commands currently implemented
commands = {'help': help, 'simulate': simulate, 'plot': plot, 'fit-epsp': fit_EPSP, 'fit-ipsp': fit_IPSP}

def main():
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        commands['help']()
        sys.exit(0)
    if not sys.argv[1] in commands:
        print('{}: {} is not a recognized command. See \'{} --help\'.'.format(progname,sys.argv[1],progname))
        sys.exit(1)
    commands[sys.argv[1]]()


if __name__ == '__main__':
    main()
    


