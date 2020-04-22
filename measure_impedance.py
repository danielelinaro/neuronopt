
import os
import sys
import json
import pickle
import argparse as arg
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import btmorph

import neuron
from dlutils.cell import Cell
from dlutils.utils import individuals_from_pickle, extract_mechanisms

neuron.h.load_file('stdrun.hoc')
neuron.h.cvode_active(1)

# the name of this script
progname = os.path.basename(sys.argv[0])


def measure_impedance(cell, segment, stim_pars):

    recorders = {}
    for lbl in 't', 'v':
        recorders[lbl] = neuron.h.Vector()
    recorders['t'].record(neuron.h._ref_t)
    recorders['v'].record(segment._ref_v)

    stim = neuron.h.IClamp(segment)

    stim.delay = stim_pars['delay']
    stim.dur = stim_pars['duration']
    stim.amp = stim_pars['amplitude']

    neuron.h.v_init = -60
    neuron.h.tstop = stim.delay + stim.dur
    neuron.h.t = 0
    neuron.h.run()

    t = np.array(recorders['t'])
    v = np.array(recorders['v'])

    idx, = np.where(t > stim.delay)
    v0 = v[idx[0] - 10]
    v1 = v[idx[-1]]
    dv = (v1 - v0) * 1e-3 # [mV]
    di = stim.amp * 1e-9  # [A]
    R = dv / di * 1e-6    # [MOhm]

    return R


def worker(segment_num, segment_group, stim_pars, swc_file, parameters, mechanisms, replace_axon, add_axon_if_missing, passive_cell, cell_id=0):

    cell_name = '{}_{:03d}_{}'.format(segment_group, segment_num, cell_id)

    cell = Cell(cell_name, swc_file, parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive_cell)

    if segment_group == 'soma':
        segments = cell.somatic_segments
    elif segment_group == 'basal':
        segments = cell.basal_segments
    elif segment_group == 'apical':
        segments = cell.apical_segments

    R = measure_impedance(cell, segments[segment_num]['seg'], stim_pars)
    neuron.h('forall delete_section()')
    return R


def plot_morpho(data, n_levels=64):
    morpho = np.loadtxt(data['swc_file'])
    xyz = morpho[:,2:5]
    idx, = np.where(morpho[:,1] != 2)
    x_min,x_max = np.min(xyz[idx,0]),np.max(xyz[idx,0])
    y_min,y_max = np.min(xyz[idx,1]),np.max(xyz[idx,1])
    dx = (x_max - x_min) * 1.1
    dy = (y_max - y_min) * 1.1
    
    x_lim = [x_min, x_max]
    y_lim = [y_min, y_max]
    x_lim[0] -= (x_lim[1] - x_lim[0]) * 0.05
    x_lim[1] += (x_lim[1] - x_lim[0]) * 0.05
    y_lim[0] -= (y_lim[1] - y_lim[0]) * 0.05
    y_lim[1] += (y_lim[1] - y_lim[0]) * 0.05

    height = 0.5
    width = (x_lim[1] - x_lim[0]) / (y_lim[1] - y_lim[0]) * height
    height += 0.3
    x_offset = 0.05
    y_offset = (1 - height) / 2
    x_spacing = 0.05

    X = np.concatenate(list(data['centers'].values()))
    R = np.concatenate(list(data['R'].values()))
    R_min = 10 # R.min()
    R_max = 2000 # R.max()

    Y = (R - R_min) / (R_max - R_min)

    fig = plt.figure(figsize=(8,4))
    ticks = np.concatenate([[R_min], np.arange(500, R_max+1, 500)])
    levels = np.linspace(R_min, R_max, n_levels)

    # linear plot
    norm = colors.Normalize(vmin = R_min, vmax = R_max)
    interp = NearestNDInterpolator(X, Y)

    ax1 = plt.axes([x_offset, y_offset, width, height])
    plt.contourf([[0,0], [0,0]], levels, norm=norm, cmap=cm.jet)
    btmorph.plot_2D_SWC(data['swc_file'], color_fun=lambda pt: cm.jet(interp(pt))[0][:3], new_fig=False,
                        filter=[1,3,4], tight=True, align=True)
    cbar = plt.colorbar(fraction=0.1, shrink=1, aspect=20, ticks=ticks, orientation='horizontal')
    cbar.set_label(r'Impedance (M$\Omega$)')
    cbar.ax.set_xticklabels(ticks)

    # log plot
    R_log = np.log10(R)
    Y = (R_log - np.log10(R_min)) / (np.log10(R_max) - np.log10(R_min))
    norm = colors.LogNorm(vmin = R_min, vmax = R_max)
    interp = NearestNDInterpolator(X, Y)
    ax2 = plt.axes([x_offset+width+x_spacing, y_offset, width, height])

    plt.contourf([[0,0], [0,0]], levels, norm=norm, cmap=cm.jet)
    btmorph.plot_2D_SWC(data['swc_file'], color_fun=lambda pt: cm.jet(interp(pt))[0][:3], new_fig=False,
                        filter=[1,3,4], tight=True, align=True)
    cbar = plt.colorbar(fraction=0.1, shrink=1, aspect=20, ticks=ticks, orientation='horizontal')
    cbar.set_label(r'Impedance (M$\Omega$)')
    cbar.ax.set_xticklabels(ticks)

    # other panel
    x0 = width * 2 + x_spacing * 3 + x_offset
    w = 1 - x0 - 0.01
    ax3 = plt.axes([x0, 0.2, w, 0.7])

    X = np.concatenate(list(data['diameters'].values()))
    ax3.plot(X[1:], R[1:], 'ko', markerfacecolor='w', linewidth=1, markersize=4)
    ax3.set_xlabel(r'Diameter ($\mu$m)')
    ax3.set_ylabel(r'Impedance (M$\Omega$)')


def plot(*args, **kwargs):

    if len(args) == 0:
        parser = arg.ArgumentParser(description='Plot results of an impedance measurement experiment')
        parser.add_argument('file', type=str, action='store', help='pickle file containing the results of the experiment')
        parser.add_argument('--levels', type=int, default=64, help='number of colormap levels')
        args = parser.parse_args(args=sys.argv[2:])
        pkl_file = args.file
        n_levels = args.levels
    else:
        pkl_file = args[0]
        try:
            n_levels = kwargs['n_levels']
        except:
            n_levels = 64

    if not os.path.isfile(pkl_file):
        print('{}: {}: no such file.'.format(progname, pkl_file))
        return

    data = pickle.load(open(pkl_file, 'rb'))
    plot_morpho(data, n_levels)
    pdf_file = os.path.splitext(pkl_file)[0] + '.pdf'
    plt.savefig(pdf_file)


if __name__ == '__main__':

    if sys.argv[1] == 'plot':
        plot()
        sys.exit(0)
    
    parser = arg.ArgumentParser(description='Measure the impedance of each compartment in a neuron model.')
    parser.add_argument('I', type=float, action='store', default=-50, nargs='?', help='current value in pA')
    parser.add_argument('-P','--pickle-file', type=str, default='', help='Pickle file containing the parameters of a population of individuals')
    parser.add_argument('-e','--evaluator-file', type=str, default='evaluator.pkl', help='Pickle file containing the evaluator')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--params-files', type=str, default='', help='JSON file(s) containing the parameters of the cell(s)')
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
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=500., type=float, help='stimulation duration (default: 500 ms)')
    parser.add_argument('--serial', action='store_true', help='do not use SCOOP')
    parser.add_argument('--trial-run', action='store_true', help='measure impedance in a random sample of 10 basal and 10 apical synapses')
    parser.add_argument('--model-type', type=str, default='active',
                        help='whether to use a passive or active model (accepted values: "active" (default) or "passive")')

    args = parser.parse_args(args=sys.argv[1:])

    if args.serial:
        map_fun = map
    else:
        try:
            from scoop import futures
            map_fun = futures.map
        except:
            map_fun = map
            print('SCOOP not found: will run sequentially')

    if not os.path.isfile(args.swc_file):
        print('{}: {}: no such file.'.format(progname,args.swc_file))
        sys.exit(1)

    if args.mech_file is not None:
        if not os.path.isfile(args.mech_file):
            print('{}: {}: no such file.'.format(progname,args.mech_file))
            sys.exit(1)
        mechanisms = json.load(open(args.mech_file,'r'))
        cell_name = os.path.splitext(os.path.basename(swc_file))[0]
    elif args.config_file is not None:
        if not os.path.isfile(args.config_file):
            print('{}: {}: no such file.'.format(progname,args.config_file))
            sys.exit(1)
        if args.cell_name is None:
            print('--cell-name must be present with --config-file option.')
            sys.exit(1)
        mechanisms = extract_mechanisms(args.config_file, args.cell_name)
        cell_name = args.cell_name

    if '*' in args.params_files:
        import glob
        params_files = glob.glob(args.params_files)
    else:
        params_files = args.params_files.split(',')
    if params_files[0] == '':
        params_files = []

    if args.pickle_file == '':
        population = [json.load(open(params_file,'r')) for params_file in params_files]
        working_dir = os.path.split(params_files[0])[0]
    else:
        if len(params_files) > 0:
            print('You cannot simultaneously specify parameter and pickle files.')
            sys.exit(1)
        population = individuals_from_pickle(args.pickle_file, args.config_file, cell_name, args.evaluator_file)
        working_dir = os.path.split(args.pickle_file)[0]

    if working_dir == '':
        working_dir = '.'

    if cell_name[-1] == '_':
        cell_name = cell_name[:-1]

    try:
        sim_pars = pickle.load(open(working_dir + '/simulation_parameters.pkl','rb'))
        if working_dir == '.':
            print('Found pickle file with simulation parameters in current directory.')
        else:
            print('Found pickle file with simulation parameters in {}.'.format(working_dir))
    except:
        sim_pars = None
        print('No pickle file with simulation parameters in {}.'.format(working_dir))

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
    stim_pars = {'delay': args.delay, 'duration': args.dur, 'amplitude': args.I * 1e-3}


    for i,parameters in enumerate(population):
        cell = Cell('CA3_cell_{}'.format(i), swc_file, parameters, mechanisms)
        cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive_cell)

        R = {'soma': np.array([measure_impedance(cell, cell.somatic_segments[0]['seg'], stim_pars)])}

        centers = {'soma': np.array([cell.somatic_segments[0]['center']])}
        centers['basal'] = np.array([seg['center'] for seg in cell.basal_segments])
        centers['apical'] = np.array([seg['center'] for seg in cell.apical_segments])

        diameters = {'soma': np.array([cell.somatic_segments[0]['sec'].diam])}
        diameters['basal'] = np.array([seg['sec'].diam for seg in cell.basal_segments])
        diameters['apical'] = np.array([seg['sec'].diam for seg in cell.apical_segments])

        areas = {'soma': np.array([cell.somatic_segments[0]['area']])}
        areas['basal'] = np.array([seg['area'] for seg in cell.basal_segments])
        areas['apical'] = np.array([seg['area'] for seg in cell.apical_segments])

        print('Somatic impedance: {:.2f} MOhm.'.format(R['soma'][0]))

        N = {'basal': len(cell.basal_segments), 'apical': len(cell.apical_segments)}
        print('The cell has {} basal and {} apical segments.'.format(N['basal'], N['apical']))
    
        idx = {k: np.arange(v) for k,v in N.items()}
        if args.trial_run:
            idx = {k: np.random.choice(v, size=10, replace=False) for k,v in idx.items()}

        neuron.h('forall delete_section()')

        for dend_type in idx:
            centers[dend_type] = centers[dend_type][idx[dend_type],:]
            diameters[dend_type] = diameters[dend_type][idx[dend_type]]
            areas[dend_type] = areas[dend_type][idx[dend_type]]
            fun = lambda num: worker(num, dend_type, stim_pars, swc_file, parameters,
                                     mechanisms, replace_axon, add_axon_if_missing,
                                     passive_cell, i)
            R[dend_type] = np.array(list(map_fun(fun, idx[dend_type])))

        data = {
            'N': N,
            'centers': centers,
            'diameters': diameters,
            'areas': areas,
            'morphology': np.loadtxt(swc_file),
            'R': R,
            'swc_file': swc_file,
            'stim_delay': args.delay,
            'stim_dur': args.dur,
            'stim_amp': args.I * 1e-3,
            'segment_indexes': idx,
            'parameters': parameters,
            'mechanisms': mechanisms,
            'replace_axon': replace_axon,
            'add_axon_if_missing': add_axon_if_missing,
            'cell_name': cell_name,
            'passive_cell': passive_cell
        }

        if args.config_file is not None:
            data['config_file'] = args.config_file
        else:
            data['mech_file'] = args.mech_file

        if len(params_files) > 0:
            data['params_file'] = args.params_files[i]
            suffix = os.path.splitext(params_files[i])[0]
        else:
            data['pickle_file'] = args.pickle_file
            data['individual'] = i
            suffix = 'pkl_individual_{}'.format(i)

        outfile = cell_name + '_impedance_' + suffix + '_' + args.model_type + '.pkl'
        pickle.dump(data, open(outfile, 'wb'))
        plot(outfile)

