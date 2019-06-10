import os
import sys
import time
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
import synapse_utils as su
import utils
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


def make_recorders(cell,synapses=None,bifurcation=False,full=False):
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

def save_data(recorders,coords,synapses,args):
    class Simulation (tbl.IsDescription):
        distribution = tbl.StringCol(16)
        mu = tbl.Float64Col()
        sigma = tbl.Float64Col()
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
    sim['mu'] = args.mu
    sim['sigma'] = args.sigma
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

def set_presynaptic_spike_times(synapses, rate, duration, delay, spike_times_file=None, seed=None):

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

    Nev = int(duration*rate*1e-3)*2  # dur is in ms
    cnt = 1
    for synapse_group in synapses.values():
        for syn in synapse_group:
            try:
                spks = spike_times[cnt]
                print('%03d > setting presynaptic spike times from file.' % cnt)
            except:
                ISIs = -np.log(np.random.uniform(size=Nev))/rate
                spks = delay + np.cumsum(ISIs)*1e3
                #print('%03d > generated presynaptic spike times from scratch.' % cnt)
            syn.set_presynaptic_spike_times(spks[spks < delay+duration])
            cnt += 1

def simulate_synaptic_activation(swc_file, parameters, mechanisms, distr_name, mu, sigma, scaling, rate, \
                                 delay, dur, rnd_seed=None, spikes_file=None, do_plot=False):

    cell,synapses = su.build_cell_with_synapses(swc_file, parameters, mechanisms, distr_name, \
                                                mu, sigma, scaling, slm_border=100.)

    recorders,coords,apc = make_recorders(cell, synapses, bifurcation=True, full=False)

    set_presynaptic_spike_times(synapses, rate, dur, delay, spikes_file, rnd_seed)

    # run the simulation
    start = time.time()
    now = time.localtime(start)
    sys.stdout.write('%02d/%02d %02d:%02d:%02d >> ' % (now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec))
    sys.stdout.flush()
    h.cvode_active(1)
    h.tstop = dur + delay*2
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
    parser.add_argument('-p','--params-file', type=str, default=None, required=True,
                        help='JSON file containing the parameters of the model')
    parser.add_argument('-m','--mech-file', type=str, default=None,
                        help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-c','--config-file', type=str, default=None,
                        help='JSON file containing the configuration of the model')
    parser.add_argument('-n','--cell-name', type=str, default=None,
                        help='name of the cell as it appears in the configuration file')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory (default: .)')
    parser.add_argument('--distr', default=None, type=str, help='type of distribution of the synaptic weights (accepted values are normal or lognormal)')
    parser.add_argument('--mu', type=float, help='Mean of the distribution of synaptic weights')
    parser.add_argument('--sigma', type=float, help='Standard deviation of the distribution of synaptic weights')
    parser.add_argument('--scaling', default=1., type=float, help='AMPA/NMDA scaling')
    parser.add_argument('--rate', type=float, help='Firing rate of the presynaptic cells')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=2000., type=float, help='stimulation duration (default: 2000 ms)')
    parser.add_argument('--seed', default=None, type=str, help='The seed of the random number generator')
    parser.add_argument('--spikes-file', default=None, type=str, help='File containing presynaptic spike times')
    parser.add_argument('--plot', action='store_true', help='show a plot (default: no)')
    args = parser.parse_args(args=sys.argv[1:])

    if not os.path.isfile(args.swc_file):
        print('%s: %s: no such file.' % (progname,args.swc_file))
        sys.exit(1)

    if not os.path.isfile(args.params_file):
        print('%s: %s: no such file.' % (progname,args.params_file))
        sys.exit(1)

    if args.mech_file is not None:
        if not os.path.isfile(args.mech_file):
            print('%s: %s: no such file.' % (progname,args.mech_file))
            sys.exit(1)
        mechanisms = json.load(open(args.mech_file,'r'))
    elif args.config_file is not None:
        if not os.path.isfile(args.config_file):
            print('%s: %s: no such file.' % (progname,args.config_file))
            sys.exit(1)
        if args.cell_name is None:
            print('--cell-name must be present with --config-file option.')
            sys.exit(1)
        mechanisms = utils.extract_mechanisms(args.config_file, args.cell_name)

    parameters = json.load(open(args.params_file,'r'))

    recorders,coords,synapses = simulate_synaptic_activation(args.swc_file, parameters, mechanisms, \
                                                             args.distr, args.mu, args.sigma, args.scaling, \
                                                             args.rate, args.delay, args.dur, args.seed, \
                                                             args.spikes_file, args.plot)
    
    save_data(recorders,coords,synapses,args)
    

if __name__ == '__main__':
    main()
