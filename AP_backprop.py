
import os
import sys
import json
import pickle
import argparse as arg
import numpy as np
from random import randint
from dlutils import cell as cu

import matplotlib.pyplot as plt
from matplotlib import cm
import btmorph
from scipy.interpolate import NearestNDInterpolator

import neuron

use_scoop = True
if use_scoop:
    try:
        from scoop import futures
        map_fun = futures.map
    except:
        map_fun = map
else:
    map_fun = map


def inject_current_step(I, delay, dur, swc_file, parameters, mechanisms, cell_name=None, neuron=None, do_plot=False):

    if use_scoop:
        do_plot = False

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
    
    n_segments = {'somatic': len(cell.somatic_segments),
                  'axonal': len(cell.axonal_segments),
                  'basal': len(cell.basal_segments),
                  'apical': len(cell.apical_segments)}

    distances = {'somatic': np.array([seg['dst'] for seg in cell.somatic_segments]),
                 'axonal': np.array([seg['dst'] for seg in cell.axonal_segments]),
                 'basal': np.array([seg['dst'] for seg in cell.basal_segments]),
                 'apical': np.array([seg['dst'] for seg in cell.apical_segments])}

    centers = {'somatic': np.array([seg['center'] for seg in cell.somatic_segments]),
               'axonal': np.array([seg['center'] for seg in cell.axonal_segments]),
               'basal': np.array([seg['center'] for seg in cell.basal_segments]),
               'apical': np.array([seg['center'] for seg in cell.apical_segments])}
                 
    recorders = {'spike_times': h.Vector(), 't': h.Vector()}
    apc = h.APCount(cell.morpho.soma[0](0.5))
    apc.thresh = -20.
    apc.record(recorders['spike_times'])
    recorders['t'].record(h._ref_t)

    for area,n in n_segments.items():
        recorders[area] = [h.Vector() for _ in range(n)]
        
    for rec,seg in zip(recorders['somatic'],cell.somatic_segments):
        rec.record(seg['seg']._ref_v)

    for rec,seg in zip(recorders['axonal'],cell.axonal_segments):
        rec.record(seg['seg']._ref_v)

    for rec,seg in zip(recorders['basal'],cell.basal_segments):
        rec.record(seg['seg']._ref_v)

    for rec,seg in zip(recorders['apical'],cell.apical_segments):
        rec.record(seg['seg']._ref_v)

    h.cvode_active(1)
    h.tstop = stim.dur + stim.delay + 100
    h.run()

    spike_times = np.array(recorders['spike_times'])
    if len(spike_times) == 0:
        return None
    
    t = np.array(recorders['t'])
    V_soma = np.array(recorders['somatic'][0])
    AP, = np.where(np.abs(t - spike_times[0]) < 1e-6)
    start = np.where(t < delay)[0][-1]
    stop = np.where(t < delay + dur)[0][-1]

    idx, = np.where((t > spike_times[0]) & (t < spike_times[0] + 2))
    AP_amplitudes = {'somatic': np.max(V_soma[idx]) - V_soma[start]}
    V_rest = {'somatic': V_soma[start]}

    window = 10
    if len(spike_times) > 1:
        t_stop = np.min([spike_times[0] + window, spike_times[1] - 1])
    else:
        t_stop = spike_times[0] + window
    print('Window for peak detection: [%g,%g] ms.' % (spike_times[0],t_stop))
    idx, = np.where((t > spike_times[0]-3) & (t < t_stop))

    for area,n in n_segments.items():
        AP_amplitudes[area] = np.zeros(n)
        V_rest[area] = np.zeros(n)
        for i in range(n):
            V = np.array(recorders[area][i])
            AP_amplitudes[area][i] = np.max(V[idx]) - V[start]
            V_rest[area][i] = V[start] - V_rest['somatic']

    apical_distances_norm = distances['apical'] / np.max(distances['apical'])
    AP_amplitudes_norm = AP_amplitudes['apical'] / AP_amplitudes['somatic'][0]

    if do_plot:
        fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4))
        ax1.plot(t[idx],V_soma[idx],'k')
        for i in range(0,n_segments['apical'],10):
            ax1.plot(t[idx],np.array(recorders['apical'][i])[idx],lw=0.5)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel(r'$V_m$ (mV)')
        ax1.set_xlim([t[idx[0]], t[idx[-1]]])
        ax2.plot(distances['apical'], AP_amplitudes_norm, 'k.')
        ax2.set_xlabel('Distance from soma (um)')
        ax2.set_ylabel('Normalized AP amplitude')
        ax2.set_xlim([0, np.max(distances['apical'])])
        ax2.set_ylim([0, 1.01])
        ax3.plot(distances['apical'], V_rest['apical'], 'k.')
        ax3.set_xlabel('Distance from soma (um)')
        ax3.set_ylabel(r'Resting $\Delta V_m$ (mV)')
        ax3.set_xlim([0, np.max(distances['apical'])])
        ax3.set_ylim([-10, 0])
        plt.show()

    h('forall delete_section()')

    Vm = {area: np.array(list(map(lambda x: np.array(x)[idx], recorders[area]))) \
          for area in ('somatic','axonal','basal','apical')}

    return {'AP_amplitudes': AP_amplitudes, 'V_rest': V_rest, \
            'distances': distances, 'centers': centers, \
            't': t[idx], 'Vm': Vm}


def individuals_from_pickle(pkl_file, config_file, cell_name=None, evaluator_file='evaluator.pkl'):
    try:
        data = pickle.load(open(pkl_file,'rb'))
        population = data['good_population']
    except:
        population = np.array(pickle.load(open(pkl_file,'rb'), encoding='latin1'))

    evaluator = pickle.load(open(evaluator_file,'rb'))

    if cell_name is None:
        default_parameters = json.load(open(parameters_file,'r'))
        config = None
    else:
        default_parameters = None
        config = json.load(open(config_file,'r'))[cell_name]

    import dlutils
    return dlutils.build_parameters_dict(population, evaluator, config, default_parameters)


def plot_means_with_errorbars(x, y, color='k', label='', mode='sem', ax=None):
    if ax is None:
        ax = plt.gca()
    Ym = np.nanmean(y,axis=0)
    if mode == 'sem':
        Ys = np.nanstd(y,axis=0) / np.sqrt(y.shape[0])
    else:
        Ys = np.nanstd(y,axis=0)
    for i,ym,ys in zip(x,Ym,Ys):
        ax.plot([i,i], [ym-ys,ym+ys], color=color, lw=2)
    ax.plot(x, Ym, 'o-', color=color, lw=2, label=label)


def plot_parameters_map(population, evaluator, config, ax, sort_parameters=True, parameter_names_on_ticks=True):
    n_parameters,n_individuals = population.shape

    bounds = {}
    for region,params in config['optimized'].items():
        for param in params:
            param_name = param[0] + '.' + region
            bounds[param_name] = np.array([param[1],param[2]])

    normalized = np.zeros(population.shape)
    pop_maxima = np.max(population,axis=1)
    pop_minima = np.min(population,axis=1)
    for i,name in enumerate(evaluator.param_names):
        normalized[i,:] = (population[i,:] - bounds[name][0]) / (bounds[name][1] - bounds[name][0])

    if sort_parameters:
        m = np.mean(normalized, axis=1)
        idx = np.argsort(m)[::-1]
        normalized_sorted_by_mean = normalized[idx,:].copy()
        s = np.std(normalized_sorted_by_mean, axis=1)
        param_names_sorted_by_mean = [evaluator.param_names[i] for i in idx]
        for i,name in enumerate(param_names_sorted_by_mean):
            if 'bar' in name:
                param_names_sorted_by_mean[i] = name.split('bar_')[1]
        img = ax.imshow(normalized_sorted_by_mean, cmap='jet')
    else:
        s = np.std(normalized, axis=1)
        img = ax.imshow(normalized, cmap='jet')

    ax.set_xlabel('Individual #')

    if n_individuals < 20:
        ax.set_xticks(np.arange(n_individuals))
        ax.set_xticklabels(1+np.arange(n_individuals))

    ax.set_yticks(np.arange(n_parameters))
    if parameter_names_on_ticks:
        if sort_parameters:
            ax.set_yticklabels(param_names)
        else:
            ax.set_yticklabelss(evaluator.param_names)
        for i in range(n_parameters):
            if s[i] < 0.2:
                ax.get_yticklabels()[i].set_color('red')
    else:
        ax.set_yticklabels(1+np.arange(n_parameters))


def set_rc_defaults():
    plt.rc('font', family='Arial', size=10)
    plt.rc('lines', linewidth=1, color='k')
    plt.rc('axes', linewidth=1, titlesize='medium', labelsize='medium')
    plt.rc('xtick', direction='out')
    plt.rc('ytick', direction='out')
    #plt.rc('figure', dpi=300)


if __name__ == '__main__':

    set_rc_defaults()

    parser = arg.ArgumentParser(description='Record back-propagating APs in a cell apical dendrites.')
    parser.add_argument('I', type=float, action='store', help='current value in pA')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--params-files', type=str, default='', help='JSON file(s) containing the parameters of the model (comma separated)')
    parser.add_argument('-m','--mech-file', type=str, default='', help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-c','--config-file', type=str, default='', help='JSON file(s) containing the configuration')
    parser.add_argument('-n','--cell-name', type=str, default='', help='cell name, if the mechanisms are stored in new style format')
    parser.add_argument('-P','--pickle-file', type=str, default='', help='Pickle file containing the parameters of a population of individuals')
    parser.add_argument('-e','--evaluator-file', type=str, default='evaluator.pkl', help='Pickle file containing the evaluator')
    parser.add_argument('-o','--output', type=str, default='', help='Output file name')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=100., type=float, help='stimulation duration (default: 100 ms)')
    parser.add_argument('--plot', action='store_true', help='plot the results (only if SCOOP is disabled)')
    args = parser.parse_args(args=sys.argv[1:])

    new_config_style = False
    if args.config_file != '':
        new_config_style = True

    if new_config_style and args.cell_name == '':
        print('You must provide the --cell-name option along with the --config-file option.')
        sys.exit(1)

    if '*' in args.params_files:
        import glob
        params_files = glob.glob(args.params_files)
    else:
        params_files = args.params_files.split(',')

    if args.mech_file == '':
        if not new_config_style:
            print('You must provide the --mech-file option if no configuration file is specified.')
            sys.exit(1)
        import dlutils
        cell_name = args.cell_name
        mechanisms = dlutils.extract_mechanisms(args.config_file, cell_name)
    else:
        cell_name = None
        mechanisms = json.load(open(args.mech_file,'r'))

    if args.pickle_file == '':
        population = [json.load(open(params_file,'r')) for params_file in params_files]
    else:
        if len(params_files) > 1:
            print('You cannot specify multiple parameter files and one pickle file.')
            sys.exit(1)
        population = individuals_from_pickle(args.pickle_file, args.config_file, cell_name, args.evaluator_file)

    if args.output == '':
        if args.pickle_file != '':
            folder,_ = os.path.split(args.pickle_file)
        else:
            folder,_ = os.path.split(params_files[0])
        if folder == '':
            folder = '.'
        pkl_output_file = folder + '/AP_backpropagation.pkl'
    else:
        pkl_output_file = args.output
    pdf_output_file = os.path.splitext(pkl_output_file)[0] + '.pdf'

    I = args.I
    dur = args.dur
    delay = args.delay
    swc_file = args.swc_file
    
    worker = lambda individual: inject_current_step(I, delay, dur, swc_file, individual, \
                                                    mechanisms, cell_name=None, neuron=None, do_plot=args.plot)
    
    data = list(map_fun(worker, population))
    neuron.h('forall delete_section()')

    for expt in data:
        expt.pop('t')
        expt.pop('Vm')
    pickle.dump(data, open(pkl_output_file,'wb'))

    normalize_distances = False
    
    apical_distances = data[0]['distances']['apical']
    normalized_apical_distances = data[0]['distances']['apical'] / np.max(data[0]['distances']['apical'])
    n_individuals = len(population)
    n_segments = len(data[0]['distances']['apical'])
    normalized_AP_amplitudes = np.nan + np.zeros((n_individuals,n_segments))

    if normalize_distances:
        n_bins = 21
        edges = np.linspace(0, 1, n_bins)
        apical_distances = normalized_apical_distances
        xlabel = 'Normalized distance from soma'
    else:
        bin_size = 40
        edges = np.arange(0, np.max(apical_distances)+bin_size/2, bin_size)
        n_bins = len(edges)
        xlabel = 'Distance from soma (um)'
        
    binned_apical_distances = np.digitize(apical_distances, edges, True)
    binned_AP_amplitudes = np.nan + np.zeros((n_individuals,n_bins))
    for i in range(n_individuals):
        if data[i] is None:
            continue
        normalized_AP_amplitudes[i] = data[i]['AP_amplitudes']['apical'] / data[i]['AP_amplitudes']['somatic']
        for j in range(1,n_bins+1):
            binned_AP_amplitudes[i][j-1] = np.mean(normalized_AP_amplitudes[i, binned_apical_distances == j])


    evaluator = pickle.load(open(args.evaluator_file, 'rb'))
    config = json.load(open(args.config_file, 'r'))

    cols = np.min([n_individuals, 5])      # how many cells per row
    rows = np.ceil(n_individuals / cols)

    morpho = np.loadtxt(swc_file)
    xyz = morpho[:,2:5]
    idx, = np.where(morpho[:,1] != 2)
    x_min,x_max = np.min(xyz[idx,0]),np.max(xyz[idx,0])
    y_min,y_max = np.min(xyz[idx,1]),np.max(xyz[idx,1])
    dx = (x_max - x_min) * 1.1
    dy = (y_max - y_min) * 1.1
    
    x_lim = [x_min,(cols-1)*dx + x_max]
    y_lim = [y_min,(rows-1)*dy + y_max]
    x_lim[0] -= (x_lim[1]-x_lim[0]) * 0.05
    x_lim[1] += (x_lim[1]-x_lim[0]) * 0.05
    y_lim[0] -= (y_lim[1]-y_lim[0]) * 0.05
    y_lim[1] += (y_lim[1]-y_lim[0]) * 0.05

    x_width = 0.6
    y_width = 0.8

    x_size = (x_lim[1] - x_lim[0]) / 300
    y_size = (y_lim[1] - y_lim[0]) / 300
    x_size *= (y_width / x_width)

    offset = np.zeros(3)

    fig = plt.figure(figsize=(x_size,y_size))
    ax1 = plt.axes([0.1,0.1,x_width,y_width])

    max_amp = np.max([np.max([np.max(v) for v in expt['AP_amplitudes'].values()]) for expt in data])
    min_amp = np.min([np.min([np.min(v) for v in expt['AP_amplitudes'].values()]) for expt in data])
    for i,expt in enumerate(data):
        points = np.r_[expt['centers']['somatic'], expt['centers']['axonal'], \
                       expt['centers']['basal'], expt['centers']['apical']] + offset
        amp = np.r_[expt['AP_amplitudes']['somatic'], expt['AP_amplitudes']['axonal'], \
                    expt['AP_amplitudes']['basal'], expt['AP_amplitudes']['apical']]
        amp = (amp - min_amp) / (max_amp - min_amp)
        interp = NearestNDInterpolator(points, amp)
        btmorph.plot_2D_SWC(swc_file, color_fun=lambda pt: cm.jet(interp(pt))[0][:3], offset=offset.tolist(), \
                            new_fig=False, filter=[1,3,4], tight=False)
        offset[0] += dx
        if (i+1) % cols == 0:
            offset[1] += dy
            offset[0] = 0

    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)

    ax2 = plt.axes([0.1+x_width+0.05, 0.1, 0.2, y_width])
    data = pickle.load(open(args.pickle_file,'rb'))
    population = data['good_population'].T
    plot_parameters_map(population, evaluator, config[cell_name], ax2, sort_parameters=False, parameter_names_on_ticks=False)
    
    plt.savefig(pdf_output_file)

