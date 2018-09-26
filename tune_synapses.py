#!/usr/bin/env python

import os
import sys
import time
import glob
import pickle
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from neuron import h
import cell_utils as cu
import synapse_utils as su

# the name of this script
progname = os.path.basename(sys.argv[0])

# definitions of normal and log-normal distribution, for the fit and plot
normal = lambda x,m,s: 1./(np.sqrt(2*np.pi*s**2)) * np.exp(-(x-m)**2/(2*s**2))
lognormal = lambda x,m,s: 1./(s*x*np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-m)**2/(2*s**2))


def build_cell_with_synapses(swc_file, mech_file, params_file, distr_name, mu, sigma, delay, dt, scaling=1., slm_border=100.):
    """
    Builds a cell and inserts one synapse per segment. Each synapse is activated sequentially.
    """
    
    cell = cu.Cell('CA3_cell_%d' % int(np.random.uniform()*1e5),{'morphology': swc_file,
                                                                 'mechanisms': mech_file,
                                                                 'parameters': params_file})

    cell.instantiate()

    if distr_name == 'normal':
        rand_func = np.random.normal
    else:
        rand_func = np.random.lognormal
        mu = np.log(mu)
        
    # one synapse in each basal segment
    Nsyn = {'basal': len(cell.basal_segments)}
    weights = {'basal': [x if x > 0 else 0 for x in rand_func(mu,sigma,size=Nsyn['basal'])]}
    synapses = {}
    synapses['basal'] = [su.AMPANMDASynapse(basal_segment['sec'], basal_segment['seg'].x, 0, [w,scaling*w]) \
                         for basal_segment,w in zip(cell.basal_segments,weights['basal'])]
    # one synapse in each apical segment that is within slm_border um from the tip of the apical dendrites
    y_coord = np.array([h.y3d(round(h.n3d(sec=segment['sec'])*segment['seg'].x),sec=segment['sec']) \
                        for segment in cell.apical_segments])
    max_y_coord = max(y_coord) - slm_border
    idx, = np.where(y_coord<max_y_coord)
    Nsyn['apical'] = len(idx)
    weights['apical'] = [x if x > 0 else 0 for x in rand_func(mu,sigma,size=Nsyn['apical'])]
    synapses['apical']  = [su.AMPANMDASynapse(cell.apical_segments[i]['sec'], cell.apical_segments[i]['seg'].x, 0, [w,scaling*w]) \
                           for i,w in zip(idx,weights['apical'])]

    # activate each synapse in a sequential fashion
    t_event = delay
    for synapse_group in synapses.values():
        for syn in synapse_group:
            syn.set_presynaptic_spike_times([t_event])
            t_event += dt

    return cell,synapses


def simulate_synaptic_activation(swc_file, mech_file, params_file, distr_name, mu, sigma, scaling, reps, output_folder='.', do_plot=True):
    """
    Instantiates reps cells and simulates them while recording the membrane potential. EPSPs are then extracted and their
    distribution is computed and fit with an appropriate distribution. Also saves the results to disk.
    """
    # do not insert synapses into the apical dendrites that are in SLM: these are the segments that lie within slm_border
    # microns from the distal tip of the dendrites
    slm_border = 100.
    # beginning of the synaptic activation
    delay = 1000.
    # interval between consecutive synapse being activated
    dt = 250.

    print('(mu,sigma) = (%g,%g): ' % (mu,sigma))

    # instantiate cells, synapses and recorders
    cells = []
    synapses = []
    recorders = [h.Vector() for rep in range(reps)]
    for rep in range(reps):
        cell,syn = build_cell_with_synapses(swc_file, mech_file, params_file, distr_name, \
                                            mu, sigma, delay, dt, scaling, slm_border)
        cells.append(cell)
        synapses.append(syn)
        recorders[rep].record(cell.morpho.soma[0](0.5)._ref_v)

    # total duration of the simulation
    tend = delay + sum(map(len,syn.values()))*dt
    
    # run the simulation
    start = time.time()
    now = time.localtime(start)
    sys.stdout.write('%02d/%02d %02d:%02d:%02d >> ' % (now.tm_mon,now.tm_mday,now.tm_hour,now.tm_min,now.tm_sec))
    sys.stdout.flush()
    h.cvode_active(1)
    h.tstop = tend
    h.run()
    sys.stdout.write('elapsed time: %d seconds.\n' % (time.time()-start))

    # extract the EPSPs from all the recordings
    EPSP_amplitudes = np.array([])
    for rec in recorders:
        V = np.array(rec)
        Vrest = V[-1]
        pks = find_peaks(V,height=Vrest+0.1)[0]
        EPSP_amplitudes = np.concatenate((EPSP_amplitudes, V[pks]-Vrest))
        
    print('Mean EPSP amplitude: %g mV.' % np.mean(EPSP_amplitudes))

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

    data = {'EPSP_amplitudes': EPSP_amplitudes, 'swc_file': swc_file, 'mech_file': mech_file, \
            'params_file': params_file, 'distr_name': distr_name, 'mu': mu, 'sigma': sigma, \
            'scaling': scaling, 'slm_border': slm_border, 'hist': hist, 'binwidth': binwidth, \
            'edges': edges, 'popt': popt}
    pickle.dump(data,open(output_folder + '/' + filename + '.pkl','w'))


############################################################
###                       SIMULATE                       ###
############################################################

def simulate():
    parser = arg.ArgumentParser(description='Simulate synaptic activation in a neuron model.')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-m','--mech-file', type=str, default='mechanisms.json', help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-p','--params-file', type=str, help='JSON file containing the parameters of the model', required=True)
    parser.add_argument('--mean', default=None, type=float, help='mean of the distribution of the synaptic weights')
    parser.add_argument('--std', default=None, type=float, help='standard deviation of the distribution of the synaptic weights')
    parser.add_argument('--distr', default=None, type=str, help='type of distribution of the synaptic weights (accepted values are normal or lognormal)')
    parser.add_argument('--reps', default=10, type=int, help='number of repetitions')
    parser.add_argument('--scaling', default=1., type=float, help='AMPA/NMDA scaling')
    parser.add_argument('--plot', action='store_true', help='show a plot (default: no)')
    parser.add_argument('--output-dir', default='.', type=str, help='output folder')
    
    args = parser.parse_args(args=sys.argv[2:])

    if args.mean is None:
        raise 'You must specify the mean of the distribution of synaptic weights'

    if args.std is None:
        raise 'You must specify the standard deviation of the distribution of synaptic weights'

    if args.std < 0:
        raise 'The standard deviation of the distribution of synaptic weights must be non-negative'

    if not args.distr in ('normal','lognormal'):
        raise 'The distribution of synaptic weights must either be "normal" or "lognormal"'

    if args.scaling < 0:
        raise 'The AMPA/NMDA scaling must be non-negative'

    if args.reps <= 0:
        raise 'The number of repetitions must be positive'
    
    simulate_synaptic_activation(args.swc_file, args.mech_file, args.params_file, args.distr, \
                                 args.mean, args.std, args.scaling, args.reps, args.output_folder, do_plot=args.plot)


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
        print('%s: %s: no such file.' % (progname,f_in))
        sys.exit(0)

    data = pickle.load(open(f_in,'r'))

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
        pickle.dump(data,open(f_in,'w'))

    if args.output is None:
        f_out = f_in.split('.pkl')[0] + '.pdf'
    else:
        f_out = args.output

    x = data['edges'][:-1] + data['binwidth']/2
    plt.figure(figsize=(6,5))
    plt.bar(x,data['hist'],width=data['binwidth'])
    if data['distr_name'] == 'normal':
        plt.plot(data['edges'],normal(data['edges'],data['popt'][0],data['popt'][1]),'r',lw=2)
    else:
        plt.plot(data['edges'],lognormal(data['edges'],data['popt'][0],data['popt'][1]),'r',lw=2)
    plt.xlabel('EPSP amplitude (mV)')
    plt.ylabel('PDF')
    plt.title('mu,sigma = %g,%g' % (data['mu'],data['sigma']))
    plt.xlim([0,6])
    plt.savefig(f_out)


############################################################
###                         HELP                         ###
############################################################

def help():
    if len(sys.argv) > 2 and sys.argv[2] in commands:
        cmd = sys.argv[2]
        sys.argv = [sys.argv[0], cmd, '-h']
        commands[cmd]()
    else:
        print('Usage: %s <command> [<args>]' % progname)
        print('')
        print('Available commands are:')
        print('   simulate       Simulate synaptic activation')
        print('   plot           Plot the results')
        print('')
        print('Type \'%s help <command>\' for help about a specific command.' % progname)


############################################################
###                         MAIN                         ###
############################################################

# all the commands currently implemented
commands = {'help': help, 'simulate': simulate, 'plot': plot}

def main():
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        commands['help']()
        sys.exit(0)
    if not sys.argv[1] in commands:
        print('%s: %s is not a recognized command. See \'%s --help\'.' % (progname,sys.argv[1],progname))
        sys.exit(1)
    commands[sys.argv[1]]()


if __name__ == '__main__':
    main()
    


