#!/usr/bin/env python

import os
import sys
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
import cell_utils as cu
import synapse_utils as su
import pickle
import json

make_event_train = lambda rate,N,offset: np.arange(N)/float(rate) + offset

DEBUG = False


def get_segment_coords(seg):
    sec = seg.sec
    n_pts = h.n3d(sec=sec)
    return np.array([h.x3d(int(seg.x*n_pts),sec=sec),\
                     h.y3d(int(seg.x*n_pts),sec=sec),\
                     h.z3d(int(seg.x*n_pts),sec=sec)])


def make_recorders(cell,synapses=None):
    sys.stdout.write('Adding a recorder to each segment... ')
    sys.stdout.flush()
    
    recorders = {'time': h.Vector(), \
                 'spike_times': h.Vector(), \
                 'Vm': [], 'I_AMPA': [], 'I_NMDA': []}

    coords = {'Vm': [], 'Isyn': []}
    
    recorders['time'].record(h._ref_t)
    apc = h.APCount(cell.morpho.soma[0](0.5))
    apc.thresh = -20.
    apc.record(recorders['spike_times'])

    for sec in h.allsec():
        for seg in sec:
            vec = h.Vector()
            vec.record(seg._ref_v)
            recorders['Vm'].append(vec)
            coords['Vm'].append(get_segment_coords(seg))

    if synapses is not None:
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

def save_recorders(recorders,coords,args):
    import tables as tbl
    
    class Simulation (tbl.IsDescription):
        mu = tbl.Float64Col()
        sigma = tbl.Float64Col()
        rate = tbl.Float64Col()
        delay = tbl.Float64Col()
        dur = tbl.Float64Col()
        
    class Coord (tbl.IsDescription):
        x = tbl.Float64Col()
        y = tbl.Float64Col()
        z = tbl.Float64Col()

    sys.stdout.write('Saving data... ')
    sys.stdout.flush()

    h5file = tbl.open_file(args.output,mode='w',title='Recorders')
    
    sim_group = h5file.create_group(h5file.root, 'Simulation')
    table = h5file.create_table(sim_group, 'Info', Simulation)
    sim = table.row
    sim['mu'] = args.mu
    sim['sigma'] = args.sigma
    sim['rate'] = args.rate
    sim['delay'] = args.delay
    sim['dur'] = args.dur
    sim.append()

    data_group = h5file.create_group(h5file.root, 'Data')
    voltage_data_group = h5file.create_group(data_group, 'Vm')
    i_ampa_data_group = h5file.create_group(data_group, 'I_AMPA')
    i_nmda_data_group = h5file.create_group(data_group, 'I_NMDA')

    h5file.create_array(data_group, 'time', np.array(recorders['time']))
    h5file.create_array(data_group, 'spike_times', np.array(recorders['spike_times']))

    for i,(rec,coord) in enumerate(zip(recorders['Vm'],coords['Vm'])):
        array = h5file.create_array(voltage_data_group, 'Vm_%04d'%i, np.array(rec))
        array.attrs.coord = coord
        
    for i,(rec_ampa,rec_nmda,coord) in enumerate(zip(recorders['I_AMPA'],recorders['I_NMDA'],coords['Isyn'])):
        array = h5file.create_array(i_ampa_data_group, 'I_AMPA_%04d'%i, np.array(rec_ampa))
        array.attrs.coord = coord
        array = h5file.create_array(i_nmda_data_group, 'I_NMDA_%04d'%i, np.array(rec_nmda))
        array.attrs.coord = coord

    sys.stdout.write('done.\n')

def simulate_synaptic_activation(swc_file, mech_file, params_file, distr_name, mu, sigma, rate, delay, dur, do_plot=False):

    cell,synapses = su.build_cell_with_synapses(swc_file, mech_file, params_file, distr_name, \
                                                mu, sigma, scaling=1., slm_border=100.)

    recorders,coords,apc = make_recorders(cell,synapses)
    
    Nev = int(dur*rate*1e-3)*2  # dur is in ms

    for syn in synapses.values():
        for s in syn:
            ISIs = -np.log(np.random.uniform(size=Nev))/rate
            spike_times = delay + np.cumsum(ISIs)*1e3
            s.set_presynaptic_spike_times(spike_times[spike_times < delay+dur])

    h.cvode_active(1)
    h.tstop = dur + delay*2

    sys.stdout.write('Running simulation... ')
    sys.stdout.flush()
    h.run()
    sys.stdout.write('done.\n')

    if do_plot:
        plt.figure()
        t = np.array(recorders['time'])
        for rec in recorders:
            if 'soma' in rec:
                v = np.array(recorders[rec]['vec'])
                break
        plt.plot(t,v,'k')
        plt.ylabel(r'$V_m$ (mV)')
        plt.xlabel('Time (ms)')
        plt.show()
        
    return recorders,coords


def main():
    parser = arg.ArgumentParser(description='Simulate synaptic activation in a neuron model.')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-m','--mech-file', type=str, default='mechanisms.json', help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-p','--params-file', type=str, help='JSON file containing the parameters of the model', required=True)
    parser.add_argument('-o','--output', type=str, default='sim.h5', help='Output file name (default: sim.pkl)')
    parser.add_argument('--mu', type=float, help='Mean of the distribution of synaptic weights')
    parser.add_argument('--sigma', type=float, help='Standard deviation of the distribution of synaptic weights')
    parser.add_argument('--rate', type=float, help='Firing rate of the presynaptic cells')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=2000., type=float, help='stimulation duration (default: 2000 ms)')
    parser.add_argument('--plot', action='store_true', help='show a plot (default: no)')
    args = parser.parse_args(args=sys.argv[1:])

    recorders,coords = simulate_synaptic_activation(args.swc_file, args.mech_file, args.params_file,\
                                                    'lognormal', args.mu, args.sigma, \
                                                    args.rate, args.delay, args.dur, args.plot)
    
    save_recorders(recorders,coords,args)
    

if __name__ == '__main__':
    main()
