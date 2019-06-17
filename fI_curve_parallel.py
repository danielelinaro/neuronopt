
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


def individuals_from_pickle(pkl_file, config_file, cell_name=None, evaluator_file='evaluator.pkl'):
    try:
        data = pickle.load(open(pkl_file,'rb'))
        population = data['good_population']
    except:
        population = np.array(pickle.load(open(pkl_file,'rb'), encoding='latin1'))

    evaluator = pickle.load(open(evaluator_file,'rb'))

    if cell_name is None:
        default_parameters = json.load(open(parameters_file,'r'))
        config = None
    else:
        default_parameters = None
        config = json.load(open(config_file,'r'))[cell_name]

    import utils
    return utils.build_parameters_dict(population, evaluator, config, default_parameters)


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
    parser.add_argument('-p','--params-files', type=str, default='', help='JSON file(s) containing the parameters of the model (comma separated)')
    parser.add_argument('-m','--mech-file', type=str, default='', help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-c','--config-file', type=str, default='', help='JSON file(s) containing the configuration')
    parser.add_argument('-n','--cell-name', default='', type=str, help='cell name, if the mechanisms are stored in new style format')
    parser.add_argument('-P','--pickle-file', type=str, default='', help='Pickle file containing the parameters of a population of individuals')
    parser.add_argument('-e','--evaluator-file', type=str, default='evaluator.pkl', help='Pickle file containing the evaluator')
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

    new_config_style = False
    if args.config_file != '':
        new_config_style = True

    if new_config_style and args.cell_name == '':
        print('You must provide the --cell-name option along with the --config-file option.')
        sys.exit(1)

    if '*' in args.params_files:
        import glob
        params_files = glob.glob(args.params_files)
    else:
        params_files = args.params_files.split(',')

    if args.mech_file == '':
        if not new_config_style:
            print('You must provide the --mech-file option if no configuration file is specified.')
            sys.exit(1)
        import utils
        cell_name = args.cell_name
        mechanisms = utils.extract_mechanisms(args.config_file, cell_name)
    else:
        cell_name = None
        mechanisms = json.load(open(args.mech_file,'r'))

    if args.pickle_file == '':
        population = [json.load(open(params_file,'r')) for params_file in params_files]
    else:
        if len(params_files) > 1:
            print('You cannot specify multiple parameter files and one pickle file.')
            sys.exit(1)
        population = individuals_from_pickle(args.pickle_file, args.config_file, cell_name, args.evaluator_file)

    dur = args.dur
    delay = args.delay
    tran = args.tran

    N = len(population)
    f = np.zeros((N,len(I)))
    no_spikes = np.zeros((N,len(I)))
    inverse_first_isi = np.zeros((N,len(I)))
    inverse_last_isi = np.zeros((N,len(I)))
    spike_times = []

    for i,individual in enumerate(population):

        worker = lambda Idc: inject_current_step(Idc, delay, dur, args.swc_file, individual, mechanisms,
                                                 cell_name=None, neuron=neuron, do_plot=False, verbose=False)

        curve = list(map_fun(worker, I))
        neuron.h('forall delete_section()')

        spks = [np.array(point['spike_times']) for point in curve]
        no_spikes[i,:] = [len(x)/dur*1e3 for x in spks]
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

