
import os
import sys
import json
import pickle
from itertools import chain
import argparse as arg

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})

from neuron import h
h.load_file('stdrun.hoc')
h.cvode_active(1)

from dlutils import utils
from dlutils.cell import Cell, branch_order
from dlutils.synapse import AMPAExp2Synapse, NMDAExp2Synapse
from dlutils.spine import Spine
from dlutils.numerics import double_exp


def compute_Rin(seg, I0, do_plot=False):
    stim = h.IClamp(seg)
    stim.delay = 500
    stim.amp = I0
    stim.dur = 200

    rec = {}
    for lbl in 't','V':
        rec[lbl] = h.Vector()
    rec['t'].record(h._ref_t)
    rec['V'].record(seg._ref_v)

    h.tstop = 1000
    h.run()
    
    t = np.array(rec['t'])
    V = np.array(rec['V'])
    idx, = np.where(t < stim.delay)
    V0 = V[idx[-10]]
    idx, = np.where(t < stim.delay + stim.dur)
    V1 = V[idx[-10]]
    dV = V1 - V0
    Rin = dV / stim.amp
    
    if do_plot:
        plt.plot(t, V, 'k')
        plt.xlabel('Time (ms)')
        plt.ylabel('Vm (mV)')
        plt.xlim([480,800]);

    stim.amp = 0
    return Rin


def cost(weights, target_dV, vec, time, EPSP, rec, t_onset):
    locations = list(target_dV.keys())
    
    for i,k in enumerate(locations):
        for j in range(len(time)):
            vec[k][j] = EPSP[k][j] * weights[i]
    h.tstop = time[-1]
    h.run()
    
    t = np.array(rec['t'])
    V = {k: np.array(rec[k]) for k in locations}
    idx, = np.where(t < t_onset['dend'] - 10)
    V0 = {k: V[k][idx[-1]] for k in locations}
    idx, = np.where(t < t_onset['dend'] + 50)
    dV = {}
    dV['dend'] = np.max(V['dend'][idx]) - V0['dend']
    idx, = np.where(t > t_onset['dend'] + 50)
    dV['spine'] = np.max(V['spine'][idx]) - V0['spine']
    return np.sum([np.abs(target_dV[k] - dV[k]) for k in locations])


if __name__ == '__main__':

    parser = arg.ArgumentParser(description='Compute spine EPSP amplitude reduction in a neuron model')
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
    parser.add_argument('--model-type', type=str, default='ttx',
                        help='whether to use a passive or active model (accepted values: "TTX" (default), "passive" or "active")')
    parser.add_argument('-o','--output-file', type=str, default=None,
                        help='Pickle file where the results of the optimization will be saved')
    parser.add_argument('-f','--force', action='store_true', help='force overwrite of existing output file')
    parser.add_argument('-q','--quiet', action='store_true', help='do not show plots')
    parser.add_argument('--Ra', type=str, default='150', help='axial resistivity of the spine (default: 150 Ohm cm)')
    parser.add_argument('--target-dV', type=float, default=10, help='target deflection (default: 10 mV)')
    parser.add_argument('--max-iter', type=int, default=20, help='maximum number of iteration for optimize (default: 20)')
    parser.add_argument('segment', type=str, default='apical[0](0.5)', nargs='?', action='store',
                        help='Segment where the spine will be placed (default: apical[0](0.5))')
    
    args = parser.parse_args(args=sys.argv[1:])

    if not os.path.isfile(args.swc_file):
        print('{}: {}: no such file.'.format(progname,args.swc_file))
        sys.exit(1)

    if not os.path.isfile(args.params_file):
        print('{}: {}: no such file.'.format(progname,args.params_file))
        sys.exit(1)

    if args.model_type.lower() == 'ttx':
        passive = False
        with_TTX = True
    elif args.model_type.lower() == 'passive':
        passive = True
        with_TTX = False
    elif args.model_type.lower() == 'active':
        passive = False
        with_TTX = False
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
        output_folder,filename = os.path.split(output_file)
        suffix,_ = os.path.splitext(filename)
    else:
        output_folder = os.getcwd()
        suffix = args.segment + '_' + args.model_type + '_Ra=' + args.Ra + '_dV=' + str(args.target_dV)
        output_file = output_folder + '/AR_' + suffix + '.pkl'

    if os.path.exists(output_file) and not args.force:
        print('{} exists: use -f to force overwrite.'.format(output_file))
        sys.exit(9)

    if '[' in args.segment:
        ss = args.segment.split('[')
        dendrite = ss[0]
        ss = ss[1].split(']')
        section_num = int(ss[0])
        segment_x = float(ss[1][1:-1])
        # segment selection mode: section x
        seg_sel_mode = 'sec_x'
    elif '{' in args.segment:
        ss = args.segment.split('{')
        dendrite = ss[0]
        segment_num = int(ss[1][:-1])
        # segment selection mode: sequential
        seg_sel_mode = 'seq'
    else:
        print('Unable to interpret `{}`.'.format(args.segment))
        sys.exit(10)

    if dendrite not in ('apical','basal'):
        print('segment must be located either on basal or apical dendrites.')
        sys.exit(11)

    ### Instantiate the cell
    cell = Cell('CA3_cell_%d' % int(np.random.uniform()*1e5), args.swc_file, parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive, TTX=with_TTX)

    if seg_sel_mode == 'sec_x':
        
        if dendrite == 'apical':
            section = cell.morpho.apic[section_num]
            all_segments = cell.apical_segments
        elif dendrite == 'basal':
            section = cell.morpho.dend[section_num]
            all_segments = cell.basal_segments

        segment = section(segment_x)

        for segment_num in range(len(all_segments)):
            if all_segments[segment_num]['seg'] == segment:
                break

    elif seg_sel_mode == 'seq':

        if dendrite == 'apical':
            seg = cell.apical_segments[segment_num]
            all_sections = cell.morpho.apic
        elif dendrite == 'basal':
            seg = cell.basal_segments[segment_num]
            all_sections = cell.morpho.dend

        segment = seg['seg']
        section = seg['sec']
        segment_x = segment.x
        for section_num in range(len(all_sections)):
            if all_sections[section_num] == section:
                break
    
    segment_dst = seg['dst']
    segment_center = seg['center']
    segment_diam = segment.diam
    segment_branch_order = branch_order(section)

    weights_0 = np.array([0.015, 0.05])

    ### Instantiate the spine
    # in the Harnett paper, the head is spherical with a diameter of 0.5 um: a cylinder
    # with diameter and length equal to 0.5 has the same (outer) surface area as the sphere
    if 'x' in args.Ra:
        Ra = float(args.Ra[:-1]) * section.Ra
    else:
        Ra = float(args.Ra)

    head_L = 0.5
    head_diam = 0.5
    neck_L = 1.58
    neck_diam = 0.077
    spine = Spine(section, segment.x, head_L, head_diam, neck_L, neck_diam, Ra)
    spine.instantiate()

    ### Compute the input resistance of the dendrite
    R_dend = compute_Rin(segment, -0.1, False)
    print('R_dend = {:.1f} MOhm.'.format(R_dend))

    ### Show where the spine is located on the dendritic tree
    plt.figure(figsize=(2,2))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    for sec in chain(cell.morpho.apic, cell.morpho.dend):
        if sec in cell.morpho.apic:
            color = 'k'
        else:
            color = 'b'
        lbl = sec.name().split('.')[1].split('[')[1][:-1]
        n = sec.n3d()
        sec_coords = np.zeros((n,2))
        for i in range(n):
            sec_coords[i,:] = np.array([sec.x3d(i), sec.y3d(i)])
        middle = int(n / 2)
        plt.text(sec_coords[middle,0], sec_coords[middle,1], lbl, fontsize=5, color='m')
        plt.plot(sec_coords[:,0], sec_coords[:,1], color, lw=1)
    plt.plot(spine._points[:,0], spine._points[:,1], 'ro', markerfacecolor='r', markersize=2)
    plt.axis('equal')
    plt.savefig(output_folder + '/spine_' + suffix + '.pdf')
    if not args.quiet:
        plt.show()

    ### Optimize the weights to have a comparable deflection in the spine when injecting current in both the spine and the dendrite
    locations = ['spine', 'dend']
    seg = {'spine': spine.head(1), 'dend': spine._sec(0.8)}
    t_onset = {'spine': 350, 'dend': 150}
    weights = [0.1, 0.1]

    tr = 1          # [ms] rise time constant
    td = 10         # [ms] decay time constant
    t_end = 400     # [ms]
    dt = 0.1        # [ms]
    time = np.arange(0, t_end, dt)
    EPSP = {k: double_exp(tr, td, t_onset[k], time) for k in t_onset}

    vec = {}
    vec['time'] = h.Vector(time)
    for i,k in enumerate(locations):
        vec[k] = h.Vector(EPSP[k])

    stimuli = {k: h.IClamp(seg[k]) for k in seg}
    rec = {'t': h.Vector(), 'soma': h.Vector()}
    rec['t'].record(h._ref_t)
    rec['soma'].record(cell.morpho.soma[0](0.5)._ref_v)
    for k in locations:
        stimuli[k].dur = 10 * t_end
        rec[k] = h.Vector()
        rec[k].record(seg[k]._ref_v)
    
    for i,k in enumerate(locations):
        vec[k].play(stimuli[k]._ref_amp, vec['time'], 1)
    
    h.cvode_active(1)

    target_dV = {k: args.target_dV for k in locations}
    opt = minimize(lambda x: cost(x, target_dV, vec, time, EPSP, rec, t_onset), \
                   weights_0, \
                   bounds = [(0,1), (0,1)], \
                   options = {'maxiter': args.max_iter, 'disp': True})

    locations.append('soma')
    
    weights = opt['x']
    c = cost(weights, target_dV, vec, time, EPSP, rec, t_onset)
    print('Cost: {:.3e}.'.format(c))

    col = {'spine': 'g', 'dend': 'm', 'soma': 'k'}
    plt.figure(figsize=(3,2))
    ax = plt.axes([0.175, 0.2, 0.775, 0.775])
    for k in locations:
        ax.plot(rec['t'], rec[k], col[k], label=k, linewidth=1)
    ax.legend(loc='best')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Vm (mV)');
    ax.set_xlim([t_onset['dend'] - 20, t_end])
    v = np.concatenate([np.array(rec[loc]) for loc in ('spine', 'dend', 'soma')])
    ax.set_ylim([v[-1] - 5, v.max() + 5])
    plt.savefig(output_folder + '/EPSPs_' + suffix + '.pdf')
    if not args.quiet:
        plt.show()

    ### Compute the amplitude ratio of the EPSPs in the spine and in the dendrite when injecting current in the spine
    t = np.array(rec['t'])
    V = {k: np.array(rec[k]) for k in locations}
    idx, = np.where(t < t_onset['dend'] - 10)
    V0 = {k: V[k][idx[-1]] for k in locations}
    idx, = np.where(t < t_onset['dend'] + 50)
    dend_to_spine_dV = {}
    dend_to_spine_dV = {k: np.max(V[k][idx]) - V0[k] for k in locations}
    idx, = np.where(t > t_onset['dend'] + 50)
    spine_to_dend_dV = {k: np.max(V[k][idx]) - V0[k] for k in locations}

    AR = spine_to_dend_dV['spine'] / spine_to_dend_dV['dend']

    R_neck = (AR - 1) * R_dend

    print('Current injected in the dendrite:')
    print('   Dendrite deflection: {:.3f} mV.'.format(dend_to_spine_dV['dend']))
    print('      Spine deflection: {:.3f} mV.'.format(dend_to_spine_dV['spine']))
    print('    Somatic deflection: {:.3f} mV.'.format(dend_to_spine_dV['soma']))
    print('')
    print('Current injected in the spine:')
    print('      Spine deflection: {:.3f} mV.'.format(spine_to_dend_dV['spine']))
    print('   Dendrite deflection: {:.3f} mV.'.format(spine_to_dend_dV['dend']))
    print('    Somatic deflection: {:.3f} mV.'.format(spine_to_dend_dV['soma']))
    print('')
    print('Amplitude ratio: {:.2f}.'.format(AR))
    print('')
    print('R_neck: {:.1f} MOhm.'.format(R_neck))

    data = {'dend_to_spine_dV': dend_to_spine_dV, 'spine_to_dend_dV': spine_to_dend_dV,
            'AR': AR, 'R_dend': R_dend, 'R_neck': R_neck, 'target_dV': target_dV, 'model_type': args.model_type,
            'weights': weights, 'cost_fun': c, 'seg_sel_mode': seg_sel_mode, 'dend_center': segment_center,
            'Ra': Ra, 'passive': passive, 'with_TTX': with_TTX, 'target_dV': args.target_dV,
            'head_L': head_L, 'head_diam': head_diam, 'neck_L': neck_L, 'neck_diam': neck_diam,
            'spine_dst': segment_dst, 'dend_diam': segment_diam, 'dend_branch_order': segment_branch_order,
            'swc_file': os.path.abspath(args.swc_file), 'params_file': os.path.abspath(args.params_file),
            'section_num': section_num, 'segment_x': segment_x, 'segment_num': segment_num}

    pickle.dump(data, open(output_file, 'wb'))


