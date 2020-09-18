import os
import sys
import time
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
from dlutils import synapse as su
from dlutils import cell as cu
from dlutils import utils
import pickle
import json
import tables as tbl

# the name of this script
progname = os.path.basename(sys.argv[0])

def get_segment_coords(seg):
    sec = seg.sec
    n_pts = h.n3d(sec=sec)
    return np.array([h.x3d(int(seg.x*n_pts),sec=sec),\
                     h.y3d(int(seg.x*n_pts),sec=sec),\
                     h.z3d(int(seg.x*n_pts),sec=sec)])


def make_recorders(cell, synapses=None, bifurcation=False, full=False):
    sys.stdout.write('Adding a recorder to each segment... ')
    sys.stdout.flush()
    
    recorders = {'time': h.Vector(), \
                 'spike_times': h.Vector(), \
                 'Vm': []}

    coords = {'Vm': []}
    
    recorders['time'].record(h._ref_t)
    apc = h.APCount(cell.morpho.soma[0](0.5))
    apc.thresh = -20.
    apc.record(recorders['spike_times'])

    for sec in h.allsec():
        for seg in sec:
            if full or 'soma' in sec.name():
                vec = h.Vector()
                vec.record(seg._ref_v)
                recorders['Vm'].append(vec)
                coords['Vm'].append(get_segment_coords(seg))

    if bifurcation and not full:
        for seg in cell.morpho.apic[0]:
            pass
        #seg = cell.morpho.apic[0](1)
        vec = h.Vector()
        vec.record(seg._ref_v)
        recorders['Vm'].append(vec)
        coords['Vm'].append(get_segment_coords(seg))

    if full and synapses is not None:
        recorders['I_AMPA'] = []
        recorders['I_NMDA'] = []
        coords['Isyn'] = []
        for syn in synapses.values():
            for s in syn:
                vec = h.Vector()
                vec.record(s.ampa_syn._ref_i)
                recorders['I_AMPA'].append(vec)
                coords['Isyn'].append(get_segment_coords(s.seg))
                vec = h.Vector()
                vec.record(s.nmda_syn._ref_i)
                recorders['I_NMDA'].append(vec)

    sys.stdout.write('done.\n')
    return recorders,coords,apc


def save_data(recorders, coords, synapses, args):
    class Simulation (tbl.IsDescription):
        distribution = tbl.StringCol(16)
        mean = tbl.Float64Col()
        std = tbl.Float64Col()
        scaling = tbl.Float64Col()
        rate = tbl.Float64Col()
        delay = tbl.Float64Col()
        dur = tbl.Float64Col()
        seed = tbl.Int64Col()
        
    sys.stdout.write('Saving data... ')
    sys.stdout.flush()

    now = time.localtime(time.time())
    filename = '%s/synaptic_activation_%d%02d%02d_%02d%02d%02d.h5' % \
               (args.output_dir,now.tm_year,now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec)

    h5file = tbl.open_file(filename,mode='w',title='Recorders')
    
    sim_group = h5file.create_group(h5file.root, 'Simulation')
    table = h5file.create_table(sim_group, 'Info', Simulation)
    sim = table.row
    sim['distribution'] = args.distr
    sim['mean'] = args.mean
    sim['std'] = args.std
    sim['scaling'] = args.scaling
    sim['rate'] = args.rate
    sim['delay'] = args.delay
    sim['dur'] = args.dur
    if args.seed is None:
        sim['seed'] = -1
    else:
        sim['seed'] = args.seed
    sim.append()

    data_group = h5file.create_group(h5file.root, 'Data')
    voltage_data_group = h5file.create_group(data_group, 'Vm')
    presyn_spike_times_data_group = h5file.create_group(data_group, 'Presyn_spike_times')
    i_ampa_data_group = h5file.create_group(data_group, 'I_AMPA')
    i_nmda_data_group = h5file.create_group(data_group, 'I_NMDA')

    h5file.create_array(data_group, 'time', np.array(recorders['time']))
    h5file.create_array(data_group, 'spike_times', np.array(recorders['spike_times']))

    for i,(rec,coord) in enumerate(zip(recorders['Vm'],coords['Vm'])):
        array = h5file.create_array(voltage_data_group, 'Vm_%04d'%i, np.array(rec))
        array.attrs.coord = coord

    for name,syn_group in synapses.items():
        grp = h5file.create_group(presyn_spike_times_data_group, name)
        for i,syn in enumerate(syn_group):
            array = h5file.create_array(grp, 'syn_%04d'%i, syn.get_presynaptic_spike_times())
            array.attrs.coord = get_segment_coords(syn.seg)

    try:
        for i,(rec_ampa,rec_nmda,coord) in enumerate(zip(recorders['I_AMPA'],recorders['I_NMDA'],coords['Isyn'])):
            array = h5file.create_array(i_ampa_data_group, 'I_AMPA_%04d'%i, np.array(rec_ampa))
            array.attrs.coord = coord
            array = h5file.create_array(i_nmda_data_group, 'I_NMDA_%04d'%i, np.array(rec_nmda))
            array.attrs.coord = coord
    except:
        pass

    sys.stdout.write('done.\n')


def set_presynaptic_spike_times(synapses, rate, duration, delay, spike_times_file=None, seed=None, verbose=False):

    np.random.seed(seed)

    total_number_of_synapses = np.sum(list(map(len,synapses.values())))

    if spike_times_file is not None:
        data = np.loadtxt(spike_times_file)
        num = np.unique(data[:,1])
        N = len(num)
        idx = np.array([])
        while len(idx) < N:
            idx = np.unique(np.append(idx,np.random.randint(1,total_number_of_synapses+1)))
        spike_times = {i: np.unique(data[data[:,1]==n,0])+delay for i,n in zip(idx,num)}
    else:
        spike_times = {}

    Nev = np.max([1, int(duration * rate * 1e-3) * 2])  # duration is in ms
    cnt = 0
    for synapse_group in synapses.values():
        for syn in synapse_group:
            try:
                spks = spike_times[cnt]
                if verbose:
                    print('{:03d} > setting presynaptic spike times from file.'.format(cnt))
            except:
                rnd = np.random.uniform(size=Nev)
                ISIs = -np.log(rnd) / rate
                spks = delay + np.cumsum(ISIs) * 1e3
                if verbose:
                    print('{:03d} > generated presynaptic spike times from scratch.'.format(cnt))
            spks = spks[spks < delay + duration]
            cnt += len(spks)
            if len(spks) > 0:
                syn.set_presynaptic_spike_times(spks)
                print(spks)
    return cnt


def simulate_synaptic_activation(swc_file, cell_parameters, mechanisms, replace_axon, add_axon_if_missing, passive_cell, \
                                 exc_syn_parameters, inh_syn_parameters, exc_rate, inh_rate, delay, dur, rnd_seed=None, \
                                 do_plot=False, verbose=False):

    cell = cu.Cell('CA3_cell_%d' % int(np.random.uniform()*1e5), swc_file, cell_parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive_cell)

    segments = {'basal': cell.basal_segments, 'apical': cell.apical_segments}

    synapses = {'excitatory': {}, 'inhibitory': {}}

    n_basal = len(exc_syn_parameters['segments_index']['EPSP']['basal'])
    n_apical = len(exc_syn_parameters['segments_index']['EPSP']['apical'])
    print('There are {} basal and {} apical excitatory synapses, for a total of {} synapses.'.format(n_basal, n_apical, n_basal+n_apical))

    synapse_parameters = exc_syn_parameters['synapse_parameters']['EPSP']
    synapse_parameters.pop('AMPA_NMDA_ratio')
    ampa_nmda_ratio = exc_syn_parameters['ampa_nmda_ratio']
    for dend_type in segments:
        synapses['excitatory'][dend_type] = []
        for index,weight in zip(exc_syn_parameters['segments_index']['EPSP'][dend_type],
                                exc_syn_parameters['weights']['EPSP'][dend_type]):
            segment = segments[dend_type][index]
            syn = su.AMPANMDASynapse(segment['sec'], segment['seg'].x, 0,
                                     [weight, weight*ampa_nmda_ratio], **synapse_parameters)
            synapses['excitatory'][dend_type].append(syn)

    n_basal = len(inh_syn_parameters['segments_index']['IPSP']['basal'])
    n_apical = len(inh_syn_parameters['segments_index']['IPSP']['apical'])
    print('There are {} basal and {} apical inhibitory synapses, for a total of {} synapses.'.format(n_basal, n_apical, n_basal+n_apical))

    synapse_parameters = inh_syn_parameters['synapse_parameters']['IPSP']
    for dend_type in segments:
        synapses['inhibitory'][dend_type] = []
        for index,weight in zip(inh_syn_parameters['segments_index']['IPSP'][dend_type],
                                inh_syn_parameters['weights']['IPSP'][dend_type]):
            segment = segments[dend_type][index]
            syn = su.GABAASynapse(segment['sec'], segment['seg'].x, -73, weight, **synapse_parameters)
            synapses['inhibitory'][dend_type].append(syn)

    recorders,coords,apc = make_recorders(cell, synapses=None, bifurcation=False, full=False)

    n_exc_spikes = set_presynaptic_spike_times(synapses['excitatory'], exc_rate, dur, delay, None, rnd_seed, verbose)
    print('Added a total of {} excitatory presynaptic spikes.'.format(n_exc_spikes))
    n_inh_spikes = set_presynaptic_spike_times(synapses['inhibitory'], inh_rate, dur, delay, None, rnd_seed+1, verbose)
    print('Added a total of {} inhibitory presynaptic spikes.'.format(n_inh_spikes))

    # run the simulation
    start = time.time()
    now = time.localtime(start)
    sys.stdout.write('%02d/%02d %02d:%02d:%02d >> ' % (now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec))
    sys.stdout.flush()
    h.cvode_active(1)
    h.tstop = dur + 2 * delay
    h.run()
    sys.stdout.write('elapsed time: %d seconds.\n' % (time.time()-start))

    if do_plot:
        plt.figure()
        t = np.array(recorders['time'])
        for rec in recorders['Vm']:
            plt.plot(t,rec)
        plt.ylabel(r'$V_m$ (mV)')
        plt.xlabel('Time (ms)')
        plt.show()
        
    return recorders,coords,synapses


def main():
    parser = arg.ArgumentParser(description='Simulate synaptic activation in a neuron model.', prog=progname)
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--cell-params', type=str, default=None, required=True,
                        help='JSON file containing the parameters of the cell')
    parser.add_argument('--exc-syn-params', type=str, default=None, required=True,
                        help='PKL file containing the parameters of the excitatory synapses')
    parser.add_argument('--inh-syn-params', type=str, default=None, required=True,
                        help='PKL file containing the parameters of the inhibitory synapses')
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
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory (default: .)')
    parser.add_argument('--exc-rate', type=float, help='Firing rate of the excitatory presynaptic cells')
    parser.add_argument('--inh-rate', type=float, help='Firing rate of the inhibitory presynaptic cells')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=2000., type=float, help='stimulation duration (default: 2000 ms)')
    parser.add_argument('--seed', default=None, type=str, help='The seed of the random number generator')
    parser.add_argument('--plot', action='store_true', help='show a plot (default: no)')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose (default: no)')
    args = parser.parse_args(args=sys.argv[1:])

    if not os.path.isfile(args.swc_file):
        print('{}: {}: no such file.'.format(progname,args.swc_file))
        sys.exit(1)

    if not os.path.isfile(args.cell_params):
        print('{}: {}: no such file.'.format(progname,args.cell_params))
        sys.exit(1)
    cell_parameters = json.load(open(args.cell_params, 'r'))

    if not os.path.isfile(args.exc_syn_params):
        print('{}: {}: no such file.'.format(progname,args.exc_syn_params))
        sys.exit(1)
    exc_synapse_parameters = pickle.load(open(args.exc_syn_params, 'rb'))

    if not os.path.isfile(args.inh_syn_params):
        print('{}: {}: no such file.'.format(progname,args.inh_syn_params))
        sys.exit(1)
    inh_synapse_parameters = pickle.load(open(args.inh_syn_params, 'rb'))

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

    seed = int(args.seed) if args.seed is not None else None

    recorders,coords,synapses = simulate_synaptic_activation(args.swc_file, cell_parameters, mechanisms, \
                                                             replace_axon, add_axon_if_missing, passive_cell, \
                                                             exc_synapse_parameters, inh_synapse_parameters, \
                                                             args.exc_rate, args.inh_rate, args.delay, args.dur, \
                                                             seed, args.plot, args.verbose)
    
    # this needs to be fixed (potentially)
    save_data(recorders, coords, synapses, args)
    

if __name__ == '__main__':
    main()
