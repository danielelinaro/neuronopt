


import os
import sys
import json
import pickle
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from random import randint

import neuron
from current_step import inject_current_step

use_scoop = True
if use_scoop:
    try:
        from scoop import futures
        map_fun = futures.map
    except:
        map_fun = map
else:
    map_fun = map

def params_files_from_pickle(pkl_file, parameters_file='parameters.json', evaluator_file='evaluator.pkl'):

    try:
        data = pickle.load(open(pkl_file,'rb'))
        population = data['good_population']
    except:
        population = np.array(pickle.load(open(pkl_file,'rb'), encoding='latin1'))

    evaluator = pickle.load(open(evaluator_file,'rb'))
    parameters_files = []

    for i,individual in enumerate(population):
        parameters = json.load(open(parameters_file,'r'))
        param_dict = evaluator.param_dict(individual)
        for par in parameters:
            if 'value' not in par:
                par['value'] = param_dict[par['param_name'] + '.' + par['sectionlist']]
                par.pop('bounds')
        params_file = '/tmp/individual_%d.json' % i
        json.dump(parameters,open(params_file,'w'),indent=4)
        parameters_files.append(params_file)

    return parameters_files


def plot_means_with_sem(x,y,color='k',label=''):
    Ym = np.mean(y,axis=0)
    Ys = np.std(y,axis=0) / np.sqrt(y.shape[0])
    for i,ym,ys in zip(x,Ym,Ys):
        plt.plot([i,i],[ym-ys,ym+ys],color=color,lw=1)
    plt.plot(x,Ym,'o-',color=color,lw=1,label=label)


if __name__ == '__main__':

    parser = arg.ArgumentParser(description='Compute the f-I curve of a neuron model.')
    parser.add_argument('I', type=str, action='store', help='current values in pA, either comma separated or interval and steps, as in 100:300:50')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-m','--mech-file', type=str, default='mechanisms.json', help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-p','--params-files', type=str, help='JSON file(s) containing the parameters of the model (comma separated)')
    parser.add_argument('-o','--output', type=str, default='fI_curve.pkl', help='Output file name')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=2000., type=float, help='stimulation duration (default: 2000 ms)')
    parser.add_argument('--tran', default=200., type=float, help='transient to be discard after stimulation onset (default: 200 ms)')
    args = parser.parse_args(args=sys.argv[1:])

    try:
        I = np.array([float(args.I)])
    except:
        if ',' in args.I:
            I = np.sort([float(x) for x in args.I.split(',')])
        elif ':' in args.I:
            tmp = [float(x) for x in args.I.split(':')]
            I = np.arange(tmp[0],tmp[1]+tmp[2]/2,tmp[2])
        else:
            print('Unknown current definition: %s.' % args.I)
            sys.exit(1)

    params_files = args.params_files.split(',')
    if params_files[0][-3:] == 'pkl':
        params_files = params_files_from_pickle(params_files[0])

    dur = args.dur
    delay = args.delay
    tran = args.tran

    N = len(params_files)
    f = np.zeros((N,len(I)))
    no_spikes = np.zeros((N,len(I)))
    inverse_first_isi = np.zeros((N,len(I)))
    inverse_last_isi = np.zeros((N,len(I)))
    spike_times = []

    for i,params_file in enumerate(params_files):

        worker = lambda Idc: inject_current_step(Idc, args.swc_file, args.mech_file,
                                                 params_file, delay, dur,
                                                 None, neuron, False, True)

        curve = list(map_fun(worker, I))
        neuron.h('forall delete_section()')
        
        spks = [np.array(point['spike_times']) for point in curve]
        no_spikes[i,:] = [x.shape[0]/dur*1e3 for x in spks]
        f[i,:] = [len(x[(x>delay+tran) & (x<delay+dur)])/(dur-tran)*1e3 for x in spks]
        inverse_first_isi[i,:] = [1e3/np.diff(t[:2]) if len(t) > 1 else 0 for t in spks]
        inverse_last_isi[i,:] = [1e3/np.diff(t[-2:]) if len(t) > 1 else 0 for t in spks]
        spike_times.append(spks)

    data = {'delay': delay, 'dur': dur, 'tran': tran,
            'I': I, 'spike_times': spike_times,
            'f': f, 'no_spikes': no_spikes,
            'inverse_first_isi': inverse_first_isi,
            'inverse_last_isi': inverse_last_isi}
    pickle.dump(data, open(args.output,'wb'))
    
    plt.figure()
    plot_means_with_sem(I*1e-3,no_spikes,color='r',label='All spikes')
    plot_means_with_sem(I*1e-3,f,color='k',label='With transient removed')
    plot_means_with_sem(I*1e-3,inverse_first_isi,color='b',label='Inverse first ISI')
    plot_means_with_sem(I*1e-3,inverse_last_isi,color='m',label='Inverse last ISI')
    plt.xlabel('Current (nA)')
    plt.ylabel(r'$f$ (spikes/s)')
    plt.legend(loc='best')
    folder = os.path.dirname(args.output)
    if folder == '':
        folder = '.'
    plt.savefig(folder + '/' + os.path.basename(args.output).split('.')[0] + '.pdf')
    plt.show()

