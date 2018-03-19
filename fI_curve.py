#!/usr/bin/env python

import os
import sys
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
import cell_utils as cu
import pickle
import json

make_suffix = lambda params_files:  '--'.join(map(lambda f: '_'.join(f.split('/')[-2:]).split('.json')[0], params_files))

def plot_means_with_sem(x,y,color='k',label=''):
    Ym = np.mean(y,axis=0)
    Ys = np.std(y,axis=0) / np.sqrt(y.shape[0])
    for i,ym,ys in zip(x,Ym,Ys):
        plt.plot([i,i],[ym-ys,ym+ys],color=color,lw=1)
    plt.plot(x,Ym,'o-',color=color,lw=1,label=label)


def compute_fI_curve_hall_of_fame(I, swc_file, mech_file, hof_file='hall_of_fame.pkl', \
                                  evaluator_file='evaluator.pkl', parameters_file='parameters.json',\
                                  delay=500., dur=2000., tran=200.):
    hall_of_fame = pickle.load(open(hof_file,'r'))
    evaluator = pickle.load(open(evaluator_file,'r'))
    for i,individual in enumerate(hall_of_fame):
        parameters = json.load(open(parameters_file,'r'))
        param_dict = evaluator.param_dict(individual)
        for par in parameters:
            if 'value' not in par:
                par['value'] = param_dict[par['param_name'] + '.' + par['sectionlist']]
                par.pop('bounds')
        params_file = 'individual_%d.json' % i
        json.dump(parameters,open(params_file,'w'),indent=4)
        parameters_files.append(params_file)
    return compute_fI_curves(I, swc_file, mech_file, parameters_files, delay, dur, tran)


def compute_fI_curves(I, swc_file, mech_file, parameters_files, delay=500., dur=2000., tran=200.):
    N = len(parameters_files)
    f = np.zeros((N,len(I)))
    no_spikes = np.zeros((N,len(I)))
    inverse_first_isi = np.zeros((N,len(I)))
    inverse_last_isi = np.zeros((N,len(I)))
    for i,params_file in enumerate(parameters_files):
        f[i,:],no_spikes[i,:],inverse_first_isi[i,:],inverse_last_isi[i,:] = \
                compute_fI_curve(I, swc_file, mech_file, params_file, delay, dur, tran, cell_name='individual_%d'%i, do_plot=False)

    plt.figure()
    plot_means_with_sem(I,no_spikes,color='r',label='All spikes')
    plot_means_with_sem(I,f,color='k',label='With transient removed')
    plot_means_with_sem(I,inverse_first_isi,color='b',label='Inverse first ISI')
    plot_means_with_sem(I,inverse_last_isi,color='m',label='Inverse last ISI')
    plt.xlabel('Current (nA)')
    plt.ylabel(r'$f$ (spikes/s)')
    plt.legend(loc='best')
    plt.savefig('fI_curve_%s.pdf' % make_suffix(parameters_files))
    plt.show()

    return f,no_spikes,inverse_first_isi,inverse_last_isi


def compute_fI_curve(I, swc_file, mech_file, params_file, delay=500., dur=2000., tran=200., cell_name='MyCell', do_plot=False):
    cell = cu.Cell(cell_name,{'morphology': swc_file,
                              'mechanisms': mech_file,
                              'parameters': params_file})
    cell.instantiate()

    stim = h.IClamp(cell.morpho.soma[0](0.5))
    stim.delay = delay
    stim.dur = dur

    recorders = {'spike_times': h.Vector()}
    apc = h.APCount(cell.morpho.soma[0](0.5))
    apc.thresh = -20.
    apc.record(recorders['spike_times'])
    spike_times = []

    if do_plot and len(I) == 1:
        for lbl in 't','Vsoma','Vaxon','Vapic','Vbasal','ICa':
            recorders[lbl] = h.Vector()
        recorders['t'].record(h._ref_t)
        recorders['Vsoma'].record(cell.morpho.soma[0](0.5)._ref_v)
        recorders['Vaxon'].record(cell.morpho.axon[4](0.5)._ref_v)
        for sec,dst in zip(cell.morpho.apic,cell.apical_path_lengths):
            if dst[0] >= 100:
                recorders['Vapic'].record(sec(0.5)._ref_v)
                break
        for sec,dst in zip(cell.morpho.dend,cell.basal_path_lengths):
            if dst[0] >= 100:
                recorders['Vbasal'].record(sec(0.5)._ref_v)
                break
        recorders['ICa'].record(cell.morpho.soma[0](0.5)._ref_ica)

    h.cvode_active(1)
    h.tstop = stim.dur + stim.delay + 100

    for amp in I:
        print('Simulating I = %g pA.' % (amp*1e3))
        stim.amp = amp
        apc.n = 0
        h.t = 0
        h.run()
        spike_times.append(np.array(recorders['spike_times']))
        if do_plot and len(I) == 1:
            plt.figure()
            t = np.array(recorders['t'])
            plt.plot(t,recorders['Vaxon'],'r',label='Axon')
            plt.plot(t,recorders['Vbasal'],'g',label='Basal')
            plt.plot(t,recorders['Vapic'],'b',label='Apical')
            plt.plot(t,recorders['Vsoma'],'k',label='Soma')
            plt.legend(loc='best')
            plt.ylabel(r'$V_m$ (mV)')
            plt.xlabel('Time (ms)')

    no_spikes = np.array(map(lambda x: x.shape[0]/dur*1e3, spike_times)).squeeze()
    f = np.array(map(lambda x: len(np.where(x > delay+tran)[0])/(dur-tran)*1e3, spike_times)).squeeze()
    inverse_first_isi = np.array([1e3/np.diff(t[:2]) if len(t) > 1 else 0 for t in spike_times]).squeeze()
    inverse_last_isi = np.array([1e3/np.diff(t[-2:]) if len(t) > 1 else 0 for t in spike_times]).squeeze()

    if do_plot:
        plt.figure()
        plt.plot(I,no_spikes,'ro-',label='All spikes')
        plt.plot(I,f,'ko-',label='With transient removed')
        plt.plot(I,inverse_first_isi,'bo-',label='Inverse first ISI')
        plt.plot(I,inverse_last_isi,'mo-',label='Inverse last ISI')
        plt.xlabel('Current (nA)')
        plt.ylabel(r'$f$ (spikes/s)')
        plt.legend(loc='best')
        plt.savefig('fI_curve.pdf')
        plt.show()

    return f,no_spikes,inverse_first_isi,inverse_last_isi


def main():
    parser = arg.ArgumentParser(description='Compute the f-I curve of a neuron model.')
    parser.add_argument('I', type=str, action='store', help='current values in pA, either comma separated or interval and steps, as in 100:300:50')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-m','--mech-file', type=str, help='JSON file containing the mechanisms to be inserted into the cell', required=True)
    parser.add_argument('-p','--params-files', type=str, help='JSON file(s) containing the parameters of the model (comma separated)')
    parser.add_argument('--hall-of-fame', action='store_true', help='compute population f-I curve')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=2000., type=float, help='stimulation duration (default: 2000 ms)')
    parser.add_argument('--tran', default=200., type=float, help='transient to be discard after stimulation onset (default: 200 ms)')
    args = parser.parse_args(args=sys.argv[1:])

    if args.hall_of_fame and not args.params_files is None:
        print('--hall-of-fame option has precedence over -p: ignoring parameters files %s.' % args.params_files)
    elif not args.hall_of_fame is None:
        params_files = args.params_files.split(',')

    try:
        I = np.array([float(args.I)])
    except:
        if ',' in args.I:
            I = np.sort(np.array(map(lambda x: float(x), args.I.split(','))))
        elif ':' in args.I:
            tmp = np.array(map(lambda x: float(x), args.I.split(':')))
            I = np.arange(tmp[0],tmp[1]+tmp[2]/2,tmp[2])
        else:
            print('Unknown current definition: %s.' % args.I)
            sys.exit(1)

    if args.hall_of_fame:
        f,no_spikes,inverse_first_isi,inverse_last_isi = compute_fI_curve_hall_of_fame(I*1e-3, args.swc_file, args.mech_file, \
                                                                                       delay=args.delay, dur=args.dur, \
                                                                                       tran=args.tran)
        suffix = 'hall_of_fame'
    elif len(params_files) > 1:
        f,no_spikes,inverse_first_isi,inverse_last_isi = compute_fI_curves(I*1e-3, args.swc_file, args.mech_file, \
                                                                           params_files, args.delay, args.dur, args.tran)
        suffix = make_suffix(params_files)
    else:
        f,no_spikes,inverse_first_isi,inverse_last_isi = compute_fI_curve(I*1e-3, args.swc_file, args.mech_file, \
                                                                          params_files[0], args.delay, args.dur, \
                                                                          args.tran, 'MyCell', do_plot=True)
        suffix = args.params_file.split('.')[0]

    fI_curve = {'I': I, 'f': f,
                'no_spikes': no_spikes,
                'inverse_first_isi': inverse_first_isi,
                'inverse_last_isi': inverse_last_isi}

    pickle.dump(fI_curve, open('fI_curve_'+suffix+'.pkl','w'))


if __name__ == '__main__':
    main()
