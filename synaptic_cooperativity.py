import os
import sys
import json
from time import strftime, localtime
from time import time as TIME
import pickle
from itertools import chain
import argparse as arg
import logging

import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import matplotlib.pyplot as plt

from neuron import h

from dlutils.cell import Cell, branch_order
from dlutils.synapse import AMPANMDAExp2Synapse
from dlutils.spine import Spine
from dlutils.utils import extract_mechanisms


prog_name = os.path.basename(sys.argv[0])



if __name__ == '__main__':

    parser = arg.ArgumentParser(description='Compute the f-I curve of a neuron model.')
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('--plot-morpho', action='store_true',
                        help='plot the morphology with the spines highlighted (default: no)')
    args = parser.parse_args(args=sys.argv[1:])

    config_file = args.config_file
    if not os.path.isfile(args.config_file):
        print(f'{prog_name}: {config_file}: no such file')
        sys.exit(1)
        
    ts = strftime('%Y%m%d-%H%M%S', localtime())
    config = json.load(open(config_file, 'r'))

    optimization_folder = config['optimization_folder']
    if not os.path.isdir(optimization_folder):
        print(f'{prog_name}: {optimization_folder}: no such directory')
        sys.exit(2)

    log_fmt = logging.Formatter('%(asctime)s  |  %(message)s', '%Y-%m-%d  %H:%M:%S')
    logger = logging.getLogger()
    file_hndl = logging.FileHandler(f'synaptic_activation_{ts}.log')
    file_hndl.setFormatter(log_fmt)
    logger.addHandler(file_hndl)
    console_hndl = logging.StreamHandler(sys.stdout)
    console_hndl.setFormatter(log_fmt)
    logger.addHandler(console_hndl)
    logger.setLevel(logging.INFO)
    
    try:
        seed = config['seed']
    except:
        with open('/dev/random', 'rb') as fid:
            seed = int.from_bytes(fid.read(4), 'little')
            config['seed'] = seed

    logger.info(f'Random number generator seed: {seed}')
    rs = RandomState(MT19937(SeedSequence(seed)))
    
    cell_type = config['cell_type']
    prefix = cell_type[0].upper() + cell_type[1:]
    base_folder = optimization_folder + prefix + '/' + config['cell_name'] + '/' + config['optimization_run'] + '/'
    swc_file = config['swc_file']
    cell_name = config['cell_name'] + '_'
    individual = config['individual']

    swc_file = base_folder + swc_file
    params_file = base_folder + f'individual_{individual}.json'
    config_file = base_folder + 'parameters.json'

    parameters = json.load(open(params_file, 'r'))
    mechanisms = extract_mechanisms(config_file, cell_name)
    sim_pars = pickle.load(open(base_folder + 'simulation_parameters.pkl','rb'))
    replace_axon = sim_pars['replace_axon']
    add_axon_if_missing = not sim_pars['no_add_axon']


    #############################
    ### simulation parameters ###
    #############################

    stim_dur = config['sim']['stim_dur']
    delay    = config['sim']['delay']
    after    = config['sim']['after']
    tstop    = delay + stim_dur + after


    ######################################
    ### build and instantiate the cell ###
    ######################################

    cell = Cell('cell_%d' % int(rs.uniform()*1e5), swc_file, parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=False, TTX=False)
    section_num = config['section_num']
    section = cell.morpho.apic[section_num]
    Ra = section.Ra * config['Ra_neck_coeff']
    logger.info(f'Branch order of section {section.name()}: {branch_order(section)}')


    ##############################
    ### instantiate the spines ###
    ##############################

    # in the Harnett paper, the head is spherical with a diameter of 0.5 um: a cylinder
    # with diameter and length equal to 0.5 has the same (outer) surface area as the sphere
    head_L = config['spine']['head_L']           # [um]
    head_diam = config['spine']['head_diam']     # [um]
    neck_L = config['spine']['neck_L']           # [um]
    neck_diam = config['spine']['neck_diam']     # [um]
    spine_distance = config['spine_distance']    # [um] distance between neighboring spines
    n_spines = config['n_spines']                # number of spines
    L = spine_distance * (n_spines - 1)
    norm_L = L / section.L

    spine_loc = config['spine_loc']
    start, stop = spine_loc + norm_L/2 * np.array([-1,1])
    if start < 0:
        start = 0
        stop = start + norm_L
    if stop > 1:
        stop = 1
        start = stop - norm_L
    spines = [Spine(section, x, head_L, head_diam, neck_L, neck_diam, Ra, i) \
              for i,x in enumerate(np.linspace(start, stop, n_spines))]

    for spine in spines:
        spine.instantiate()
    logger.info(f'Spines axial resistivity: {Ra:.1f} Ohm cm')
    
    if args.plot_morpho:
        # show where the spines are located on the dendritic tree
        plt.figure(figsize=(6,6))
        for sec in chain(cell.morpho.apic, cell.morpho.basal):
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
            plt.text(sec_coords[middle,0], sec_coords[middle,1], lbl, \
                     fontsize=14, color='m')
            plt.plot(sec_coords[:,0], sec_coords[:,1], color, lw=1)
        for spine in spines:
            plt.plot(spine._points[:,0], spine._points[:,1], 'r.')
        plt.axis('equal')
        plt.savefig(f'morpho_with_spines_{ts}.pdf')


    # check the location of the spines in terms of distinct segments
    segments = [section(spines[0]._sec_x)]
    segments_idx = [[0]]
    for i,spine in enumerate(spines[1:]):
        if section(spine._sec_x) == segments[-1]:
            segments_idx[-1].append(i+1)
        else:
            segments.append(section(spine._sec_x))
            segments_idx.append([i+1])
    if len(segments_idx) == 1:
        logger.info('All spines are connected to the same segment')
    elif len(segments_idx) == n_spines:
        logger.info('Each spine is connected to a different segment on the dendritic branch')
    else:
        for group in segments_idx:
            if len(group) > 1:
                logger.info(f'Spines {np.array(group)+1} are connected to the same segment')
            else:
                logger.info(f'Spine {group[0]+1} is connected to a distinct segment')


    ########################################
    ### insert a synapse into each spine ###
    ########################################

    MG_MODELS = {'MDS': 1, 'HRN': 2, 'JS': 3}
    Mg_unblock_model = config['NMDA']['model']

    E = config['E_syn'] # [mV] reversal potential of the synapses

    AMPA_taus = config['AMPA']['time_constants']
    NMDA_taus = config['NMDA']['time_constants']
    weights = np.array([config['AMPA']['weight'], config['NMDA']['weight']])

    
    logger.info('AMPA:')
    logger.info('    tau_rise = {:.3f} ms'.format(AMPA_taus['tau1']))
    logger.info('   tau_decay = {:.3f} ms'.format(AMPA_taus['tau2']))
    logger.info('NMDA:')
    logger.info('    tau_rise = {:.3f} ms'.format(NMDA_taus['tau1']))
    logger.info('   tau_decay = {:.3f} ms'.format(NMDA_taus['tau2']))

    synapses = [AMPANMDAExp2Synapse(spine.head, 1, E, weights, AMPA = AMPA_taus, \
                                    NMDA = NMDA_taus) for spine in spines]

    for syn in synapses:
        syn.nmda_syn.mg_unblock_model = MG_MODELS[Mg_unblock_model]
        if Mg_unblock_model == 'MDS':
            syn.nmda_syn.alpha_vspom = config['NMDA']['alpha_vspom']
            syn.nmda_syn.v0_block = config['NMDA']['v0_block']
            syn.nmda_syn.eta = config['NMDA']['eta']
        elif Mg_unblock_model == 'JS':
            syn.nmda_syn.Kd = config['NMDA']['Kd']
            syn.nmda_syn.gamma = config['NMDA']['gamma']
            syn.nmda_syn.sh = config['NMDA']['sh']

    if Mg_unblock_model == 'MDS':
        logger.info('Using Maex & De Schutter Mg unblock model. Modified parameters:')
        logger.info('       alpha = {:.3f} 1/mV'.format(synapses[0].nmda_syn.alpha_vspom))
        logger.info('    v0_block = {:.3f} mV'.format(synapses[0].nmda_syn.v0_block))
        logger.info('         eta = {:.3f}'.format(synapses[0].nmda_syn.eta))
    elif Mg_unblock_model == 'JS':
        logger.info('Using Jahr & Stevens Mg unblock model. Modified parameters:')
        logger.info('          Kd = {:.3f} 1/mV'.format(synapses[0].nmda_syn.Kd))
        logger.info('       gamma = {:.3f} 1/mV'.format(synapses[0].nmda_syn.gamma))
        logger.info('          sh = {:.3f} mV'.format(synapses[0].nmda_syn.sh))
    elif Mg_unblock_model == 'HRN':
        logger.info('Using Harnett Mg unblock model with default parameters')


    ###########################################
    ### compute the presynaptic spike times ###
    ###########################################

    # spines will be activated in a Poisson fashion with this average interval between activations
    F_burst = config['synaptic_activation_frequency']
    n_bursts = stim_dur * F_burst * 2
    if n_bursts > 0:
        IBI = - np.log(rs.uniform(size=n_bursts)) / F_burst * 1e3  # [ms]
        presyn_burst_times = 2 * delay + np.cumsum(IBI)
        presyn_burst_times = presyn_burst_times[presyn_burst_times < delay + stim_dur]
    
        try:
            F = config['poisson_frequency']
            poisson = True
        except:
            poisson = False
            spike_dt = config['spike_dt']
        
        presyn_spike_times = [np.array([]) for _ in range(n_spines)]

        for t0 in presyn_burst_times:
            if poisson:
                ISI = - np.log(rs.uniform(size=n_spines*2)) / F * 1e3  # [ms]
                spks = np.cumsum(ISI)
                for i,j in enumerate(rs.permutation(n_spines)):
                    presyn_spike_times[j] = np.append(presyn_spike_times[j], t0 + spks[i])
            else:
                for i in range(n_spines):
                    presyn_spike_times[i] = np.append(presyn_spike_times[i], t0 + i * spike_dt)

        logger.info('Presynaptic spike times:')
        for i, spks in enumerate(presyn_spike_times):
            logger.info(f'Spine {i+1}: t = {spks}')

        for syn, spks in zip(synapses, presyn_spike_times):
            syn.set_presynaptic_spike_times(spks)
    else:
        logger.info('No presynaptic stimulation')
        presyn_burst_times = np.array([])
        presyn_spike_times = np.array([])

    ############################
    ### make the OU stimulus ###
    ############################

    dt = config['sim']['dt']        # [ms]
    OU = {}
    OU['t'] = np.arange(0, tstop, dt)
    OU['x'] = np.zeros(OU['t'].size)
    OU['rnd'] = rs.normal(size=OU['t'].size)
    for par in 'mean', 'stddev', 'tau':
        OU[par] = config['OU'][par]
    OU['const'] = 2 * OU['stddev']**2 / OU['tau']
    OU['mu'] = np.exp(-dt / OU['tau'])
    OU['coeff'] = np.sqrt(OU['const'] * OU['tau'] / 2 * (1 - OU['mu'] ** 2))
    idx, = np.where((OU['t'] >= delay)  & (OU['t'] <= delay + stim_dur))
    OU['x'][idx[0]] = OU['mean']
    for i in idx[1:]:
        OU['x'][i] = OU['mean'] + OU['mu'] * (OU['x'][i-1] - OU['mean']) + OU['coeff'] * OU['rnd'][i]

    vec = {key: h.Vector(OU[key]) for key in ('t','x')}

    stim = h.IClamp(cell.morpho.soma[0](0.5))
    if OU['stddev'] != 0:
        stim.dur = 10 * tstop
        vec['x'].play(stim._ref_amp, vec['t'], 1)
    else:
        stim.dur = stim_dur
        stim.delay = delay
        stim.amp = OU['mean']
        logger.info('The standard deviation of the OU process is zero: using conventional current clamp stimulus')

    ##########################
    ### make the recorders ###
    ##########################

    recorders = {}
    for lbl in 'time', 'Vsoma', 'spike_times':
        recorders[lbl] = h.Vector()
    recorders['time'].record(h._ref_t)
    recorders['Vsoma'].record(cell.morpho.soma[0](0.5)._ref_v)
    apc = h.APCount(cell.morpho.soma[0](0.5))
    apc.thresh = -20
    apc.record(recorders['spike_times'])


    ##########################
    ### run the simulation ###
    ##########################

    if OU['stddev'] != 0:
        h.cvode_active(0)
        h.dt = dt
        logger.info(f'Not using CVode: dt set to {dt:.3f} ms')
    else:
        h.cvode_active(1)
        logger.info('Using CVode')

    h.tstop = tstop
    logger.info('Running simulation')
    start = TIME()
    h.run()
    end = TIME()
    dur = int(end - start)
    hours = dur // 3600
    minutes = (dur % 3600) // 60
    secs = (dur % 60) % 60
    logger.info(f'Elapsed time: {hours:02d}:{minutes:02d}:{secs:02d}')
    

    #####################
    ### save the data ###
    #####################

    data = {
        'config': config,
        'OU_t': OU['t'],
        'OU_x': OU['x'],
        'RA': Ra,
        'presyn_burst_times': presyn_burst_times,
        'presyn_spike_times': presyn_spike_times
    }
    for key in recorders:
        data[key] = np.array(recorders[key])
    logger.info(f'Saving data to synaptic_activation_{ts}.npz')
    np.savez_compressed(f'synaptic_activation_{ts}.npz', **data)
    spike_times = np.array(recorders['spike_times'])
    ISI = np.diff(spike_times) * 1e-3
    firing_rate = len(spike_times) / stim_dur * 1e3
    CV = ISI.std() / ISI.mean()
    logger.info(f'Firing rate = {firing_rate:.2f} spike/s')
    logger.info(f'CV = {CV:.4f}')

    #####################
    ### plot a figure ### 
    #####################

    logger.info(f'Plotting simulation results to synaptic_activation_{ts}.pdf')
    fig,ax = plt.subplots(1, 1, figsize=(6,4))
    ax.plot(data['time'], data['Vsoma'], 'k', lw=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Vm (mV)')
    for side in 'top', 'right':
        ax.spines[side].set_visible(False)
    plt.savefig(f'synaptic_activation_{ts}.pdf')
    fig.tight_layout()


