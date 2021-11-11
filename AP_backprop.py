
import os
import sys
import json
import pickle
import argparse as arg
import numpy as np
from random import randint

import matplotlib.pyplot as plt
from matplotlib import cm
import btmorph
from scipy.interpolate import NearestNDInterpolator

from dlutils import cell as cu
from dlutils.utils import *
from dlutils.graphics import *
from dlutils.analysis import plot_parameters_map
import neuron

set_rc_defaults()

use_scoop = False
if use_scoop:
    try:
        from scoop import futures
        map_fun = futures.map
    except:
        map_fun = map
else:
    map_fun = map


def inject_current_step(I, delay, dur, swc_file, parameters, mechanisms, replace_axon=False, \
                        add_axon_if_missing=True, cell_name=None, neuron=None, do_plot=False):

    if use_scoop:
        print('Disabling plot because of SCOOP.')
        do_plot = False

    if cell_name is None:
        import random
        cell_name = 'cell_%06d' % random.randint(0,999999)

    cell = cu.Cell(cell_name, swc_file, parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing)

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
               'basal': np.array([seg['center'] for seg in cell.basal_segments]),
               'apical': np.array([seg['center'] for seg in cell.apical_segments])}
    try:
        centers['axonal'] = np.array([seg['center'] for seg in cell.axonal_segments])
    except:
        centers['axonal'] = np.array([[0,seg['dst'],0] for seg in cell.axonal_segments])
                 
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

    window = 30
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
        ax2.set_xlabel(r'Distance from soma ($\mu$m)')
        ax2.set_ylabel('Normalized AP amplitude')
        ax2.set_xlim([0, np.max(distances['apical'])])
        ax2.set_ylim([0, 1.01])
        ax3.plot(distances['apical'], V_rest['apical'], 'k.')
        ax3.set_xlabel(r'Distance from soma ($\mu$m)')
        ax3.set_ylabel(r'Resting $\Delta V_m$ (mV)')
        ax3.set_xlim([0, np.max(distances['apical'])])
        fig.tight_layout()
        plt.show()

    h('forall delete_section()')

    idx, = np.where((t > spike_times[0]-5) & (t < spike_times[0]+100))
    Vm = {area: np.array(list(map(lambda x: np.array(x)[idx], recorders[area]))) \
          for area in ('somatic','axonal','basal','apical')}

    return {'AP_amplitudes': AP_amplitudes, 'V_rest': V_rest, \
            'distances': distances, 'centers': centers, \
            't': t[idx], 'Vm': Vm}


if __name__ == '__main__':

    parser = arg.ArgumentParser(description='Record back-propagating APs in a cell apical dendrites.')
    parser.add_argument('I', type=float, action='store', help='current value in pA')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--params-files', type=str, default=None, help='JSON file(s) containing the parameters of the model (comma separated)')
    parser.add_argument('-m','--mech-file', type=str, default='', help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-c','--config-file', type=str, default='', help='JSON file(s) containing the configuration')
    parser.add_argument('-n','--cell-name', type=str, default='', help='cell name, if the mechanisms are stored in new style format')
    parser.add_argument('-P','--pickle-file', type=str, default=None, help='Pickle file containing the parameters of a population of individuals')
    parser.add_argument('-e','--evaluator-file', type=str, default='evaluator.pkl', help='Pickle file containing the evaluator')
    parser.add_argument('-o','--output', type=str, default='', help='Output file name')
    parser.add_argument('-R','--replace-axon', type=str, default=None,
                        help='whether to replace the axon (accepted values: "yes" or "no")')
    parser.add_argument('-A', '--add-axon-if-missing', type=str, default=None,
                        help='whether add an axon if the cell does not have one (accepted values: "yes" or "no")')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=100., type=float, help='stimulation duration (default: 100 ms)')
    parser.add_argument('--plot', action='store_true', help='plot the results of each integration (only if SCOOP is disabled)')
    parser.add_argument('--with-traces', action='store_true', help='plot voltage traces on the morphology.')
    args = parser.parse_args(args=sys.argv[1:])

    new_config_style = False
    if args.config_file != '':
        new_config_style = True

    if new_config_style and args.cell_name == '':
        print('You must provide the --cell-name option along with the --config-file option.')
        sys.exit(1)

    if args.params_files is not None:
        if '*' in args.params_files:
            import glob
            params_files = glob.glob(args.params_files)
        else:
            params_files = args.params_files.split(',')

    if args.mech_file == '':
        if not new_config_style:
            print('You must provide the --mech-file option if no configuration file is specified.')
            sys.exit(1)
        cell_name = args.cell_name
        mechanisms = extract_mechanisms(args.config_file, cell_name)
    else:
        cell_name = None
        mechanisms = json.load(open(args.mech_file,'r'))

    if args.pickle_file is None:
        population = [json.load(open(params_file,'r')) for params_file in params_files]
    else:
        if len(params_files) > 1:
            print('You cannot specify multiple parameter files and one pickle file.')
            sys.exit(1)
        population = individuals_from_pickle(args.pickle_file, args.config_file, cell_name, args.evaluator_file)

    if args.output == '':
        if args.pickle_file is not None:
            folder,_ = os.path.split(args.pickle_file)
        else:
            folder,_ = os.path.split(params_files[0])
        if folder == '':
            folder = '.'
        pkl_output_file = folder + '/AP_backpropagation.pkl'
    else:
        pkl_output_file = args.output
    pdf_output_file = os.path.splitext(pkl_output_file)[0] + '.pdf'

    try:
        sim_pars = pickle.load(open('simulation_parameters.pkl','rb'))
    except:
        sim_pars = None

    if args.replace_axon is None:
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
            sys.exit(3)

    if args.add_axon_if_missing is None:
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
            sys.exit(4)

    I = args.I
    dur = args.dur
    delay = args.delay
    swc_file = args.swc_file
    
    worker = lambda individual: inject_current_step(I, delay, dur, swc_file, individual, mechanisms, \
                                                    replace_axon, add_axon_if_missing, cell_name=None, \
                                                    neuron=None, do_plot=args.plot)
    
    data = list(map_fun(worker, population))
    neuron.h('forall delete_section()')

    T = []
    VM = []
    for expt in data:
        T.append(expt.pop('t'))
        VM.append(expt.pop('Vm'))
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

        if args.with_traces:
            scale = {'t': 3, 'v': 2}
            t = offset[0] + (T[i] - T[i][0]) * scale['t']
            v = offset[1] + (VM[i]['somatic'][0,:] - VM[i]['somatic'][0,0]) * scale['v']
            ax1.plot(t, v, color=[.8,0,.8], lw=1)
            for dst in (400,600,800):
                idx = np.where(expt['distances']['apical'] > dst)[0][0]
                xy = expt['centers']['apical'][idx,:2]
                t = offset[0] + xy[0] + (T[i] - T[i][0]) * scale['t']
                v = offset[1] + xy[1] + (VM[i]['apical'][idx,:] - VM[i]['apical'][idx,0]) * scale['v']
                ax1.plot(t, v, color=[.6,.6,.6], lw=1)

        offset[0] += dx
        if (i+1) % cols == 0:
            offset[1] += dy
            offset[0] = 0

    dx = np.diff(x_lim)
    dy = np.diff(y_lim)
    x = x_lim[0] + 0.05*dx
    y = y_lim[0] + 0.05*dx
    length = 200
    ax1.plot(x+np.zeros(2), y+np.array([0,length]), 'k', lw=1)
    ax1.text(x_lim[0]+0.01*dx, y+length/2, '{} um'.format(length), rotation=90, \
             verticalalignment='center', fontsize=8)

    if args.with_traces:
        x = x_lim[0] + 0.05*dx
        y = y_lim[0] + 0.5*dy
        x_length = 10 * scale['t']
        y_length = 100 * scale['v']
        ax1.plot(x+np.zeros(2), y+np.array([0,y_length]), 'k', lw=1)
        ax1.text(x_lim[0]+0.01*dx, y+y_length/2, '{:.0f} mV'.format(y_length/scale['v']), rotation=90, \
                 verticalalignment='center', fontsize=8)
        ax1.plot(x+np.array([0,x_length]), y+np.zeros(2), 'k', lw=1)
        ax1.text(x+x_length/2, y, '{:.0f} ms'.format(x_length/scale['t']), horizontalalignment='center', \
                 verticalalignment='top', fontsize=8)

    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)

    ax2 = plt.axes([0.1+x_width+0.05, 0.1, 0.2, y_width])
    if args.pickle_file is not None:
        data = pickle.load(open(args.pickle_file,'rb'))
        population = data['good_population'].T
    elif len(params_files) > 0:
        import re
        idx = np.array([int(re.findall(r'[0-9]+', f)[0]) for f in params_files])
        population = np.array(pickle.load(open('hall_of_fame.pkl','rb'))).T
        population = population[:,idx]

    plot_parameters_map(population, evaluator, config[cell_name], ax2, \
                        sort_parameters=False, parameter_names_on_ticks=False)
    
    plt.savefig(pdf_output_file)

