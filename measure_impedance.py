
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

import neuron
import dlutils.cell as cu

import btmorph

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


def worker(segment_num, segment_group, stim_pars, swc_file, parameters, mechanisms, replace_axon, add_axon_if_missing, passive_cell):

    cell_name = '{}_{:03d}'.format(segment_group, segment_num)

    cell = cu.Cell(cell_name, swc_file, parameters, mechanisms)
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


def plot_morpho(data, use_log=False, n_levels=64):
    morpho = np.loadtxt(data['swc_file'])
    xyz = morpho[:,2:5]
    idx, = np.where(morpho[:,1] != 2)
    x_min,x_max = np.min(xyz[idx,0]),np.max(xyz[idx,0])
    y_min,y_max = np.min(xyz[idx,1]),np.max(xyz[idx,1])
    dx = (x_max - x_min) * 1.1
    dy = (y_max - y_min) * 1.1
    
    x_lim = [x_min, x_max]
    y_lim = [y_min, y_max]
    x_lim[0] -= (x_lim[1]-x_lim[0]) * 0.05
    x_lim[1] += (x_lim[1]-x_lim[0]) * 0.05
    y_lim[0] -= (y_lim[1]-y_lim[0]) * 0.05
    y_lim[1] += (y_lim[1]-y_lim[0]) * 0.05

    x_width = 0.4
    y_width = 0.9

    x_size = (x_lim[1] - x_lim[0]) / 100
    y_size = (y_lim[1] - y_lim[0]) / 100
    x_size *= (y_width / x_width)

    fig = plt.figure(figsize=(x_size, y_size))
    ax1 = plt.axes([0.05, 0.05, x_width, y_width])

    X = np.concatenate(list(data['centers'].values()))
    R = np.concatenate(list(data['R'].values()))
    R_min = 10 # R.min()
    R_max = 2000 # R.max()

    if use_log:
        R_log = np.log10(R)
        Y = (R_log - np.log10(R_min)) / (np.log10(R_max) - np.log10(R_min))
    else:
        Y = (R - R_min) / (R_max - R_min)

    if use_log:
        norm = colors.LogNorm(vmin = R_min, vmax = R_max)
    else:
        norm = colors.Normalize(vmin = R_min, vmax = R_max)

    ticks = np.concatenate([[R_min], np.arange(500, R_max+1, 500)])

    levels = np.linspace(R_min, R_max, n_levels)

    interp = NearestNDInterpolator(X, Y)

    plt.contourf([[0,0], [0,0]], levels, norm=norm, cmap=cm.jet)
    btmorph.plot_2D_SWC(data['swc_file'], color_fun=lambda pt: cm.jet(interp(pt))[0][:3], new_fig=False,
                        filter=[1,3,4], tight=True, align=True)
    cbar = plt.colorbar(fraction=0.1, shrink=0.7, aspect=30, ticks=ticks)
    cbar.set_label(r'Impedance (M$\Omega$)')
    cbar.ax.set_yticklabels(ticks)

    ax2 = plt.axes([0.6, 0.1, 0.35, 0.35])
    ax3 = plt.axes([0.6, 0.6, 0.35, 0.35])

    X = np.concatenate(list(data['areas'].values()))
    ax2.plot(X[1:], R[1:], 'ko', markerfacecolor='w', linewidth=1, markersize=4)
    ax2.set_xlabel(r'Area ($\mu$m$^2$)')
    ax2.set_ylabel(r'Impedance (M$\Omega$)')
    X = np.concatenate(list(data['diameters'].values()))
    ax3.plot(X[1:], R[1:], 'ko', markerfacecolor='w', linewidth=1, markersize=4)
    ax3.set_xlabel(r'Diameter ($\mu$m)')
    ax3.set_ylabel(r'Impedance (M$\Omega$)')


def plot(*args, **kwargs):

    if len(args) == 0:
        parser = arg.ArgumentParser(description='Plot results of an impedance measurement experiment')
        parser.add_argument('file', type=str, action='store', help='pickle file containing the results of the experiment')
        parser.add_argument('--log', action='store_true', help='print log of data')

        args = parser.parse_args(args=sys.argv[2:])
        pkl_file = args.file
        use_log = args.log
    else:
        pkl_file = args[0]
        try:
            use_log = kwargs['use_log']
        except:
            use_log = False

    if not os.path.isfile(pkl_file):
        print('{}: {}: no such file.'.format(progname, pkl_file))
        return

    data = pickle.load(open(pkl_file, 'rb'))
    plot_morpho(data, use_log)
    pdf_file = os.path.splitext(pkl_file)[0]
    if use_log:
        pdf_file += '_log.pdf'
    else:
        pdf_file += '_linear.pdf'
    plt.savefig(pdf_file)


if __name__ == '__main__':

    if sys.argv[1] == 'plot':
        plot()
        sys.exit(0)
    
    parser = arg.ArgumentParser(description='Measure the impedance of each compartment in a neuron model.')
    parser.add_argument('I', type=float, action='store', default=-50, nargs='?', help='current value in pA')

    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--params-file', type=str, default=None, required=True,
                        help='JSON file containing the parameters of the cell')
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

    from dlutils import utils

    if not os.path.isfile(args.swc_file):
        print('{}: {}: no such file.'.format(progname,args.swc_file))
        sys.exit(1)

    if not os.path.isfile(args.params_file):
        print('{}: {}: no such file.'.format(progname,args.params_file))
        sys.exit(1)
    parameters = json.load(open(args.params_file, 'r'))

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
        mechanisms = utils.extract_mechanisms(args.config_file, args.cell_name)
        cell_name = args.cell_name

    if cell_name[-1] == '_':
        cell_name = cell_name[:-1]

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

    swc_file = args.swc_file
    stim_pars = {'delay': args.delay, 'duration': args.dur, 'amplitude': args.I * 1e-3}

    cell = cu.Cell('CA3_cell', swc_file, parameters, mechanisms)
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
                                 passive_cell)
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
        'passive_cell': passive_cell,
        'params_file': args.params_file
    }

    if args.config_file is not None:
        data['config_file'] = args.config_file
    else:
        data['mech_file'] = args.mech_file

    outfile = cell_name + '_impedance_' + os.path.splitext(args.params_file)[0] + '_' + args.model_type + '.pkl'
    pickle.dump(data, open(outfile, 'wb'))

    plot(outfile, use_log=False)
    plot(outfile, use_log=True)


