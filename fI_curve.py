import os
import sys
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from neuron import h
import cell_utils as cu
from utils import *
import pickle
import json

make_suffix = lambda params_files:  '--'.join(['_'.join(f.split('/')[-2:]).split('.json')[0] for f in params_files])

def plot_means_with_sem(x,y,color='k',label=''):
    Ym = np.mean(y,axis=0)
    Ys = np.std(y,axis=0) / np.sqrt(y.shape[0])
    for i,ym,ys in zip(x,Ym,Ys):
        plt.plot([i,i],[ym-ys,ym+ys],color=color,lw=1)
    plt.plot(x,Ym,'o-',color=color,lw=1,label=label)


def compute_fI_curves(I, swc_file, parameters, mechanisms, delay=500., dur=2000., tran=200., do_plot=True):
    N = len(parameters)
    f = np.zeros((N,len(I)))
    no_spikes = np.zeros((N,len(I)))
    inverse_first_isi = np.zeros((N,len(I)))
    inverse_last_isi = np.zeros((N,len(I)))
    for i,params in enumerate(parameters):
        f[i,:],no_spikes[i,:],inverse_first_isi[i,:],inverse_last_isi[i,:] = \
                compute_fI_curve(I, swc_file, params, mechanisms, delay, dur, tran, cell_name='individual_%d'%i, do_plot=False)

    if do_plot:
        plot_means_with_sem(I,no_spikes,color='r',label='All spikes')
        plot_means_with_sem(I,f,color='k',label='With transient removed')
        plot_means_with_sem(I,inverse_first_isi,color='b',label='Inverse first ISI')
        plot_means_with_sem(I,inverse_last_isi,color='m',label='Inverse last ISI')
        plt.xlabel('Current (nA)')
        plt.ylabel(r'$f$ (spikes/s)')
        plt.legend(loc='best')

    return f,no_spikes,inverse_first_isi,inverse_last_isi


def compute_fI_curve(I, swc_file, parameters, mechanisms, delay=500., dur=2000., tran=200., cell_name='MyCell', do_plot=False):

    cell = cu.Cell(cell_name, swc_file, parameters, mechanisms)
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
        if cell.n_axonal_sections > 0:
            recorders['Vaxon'].record(cell.morpho.axon[0](0.5)._ref_v)
        else:
            print('The cell has no axon.')
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
        if len(I) == 1 and do_plot:
            plt.figure()
            plt.plot(recorders['t'],recorders['Vsoma'],'k',label='Soma')
            plt.xlabel('Time (ms)')
            plt.ylabel(r'$V_m$ (mV)')

    no_spikes = np.array([x.shape[0]/dur*1e3 for x in spike_times])
    f = np.array([len(x[(x>delay+tran) & (x<delay+dur)])/(dur-tran)*1e3 for x in spike_times])
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

    return f,no_spikes,inverse_first_isi,inverse_last_isi


def main():
    parser = arg.ArgumentParser(description='Compute the f-I curve of a neuron model.')
    parser.add_argument('I', type=str, action='store', help='current values in pA, either comma separated or interval and steps, as in 100:300:50')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--params-files', type=str, default=None,
                        help='JSON file containing the parameters of the model')
    parser.add_argument('-m','--mech-file', type=str, default=None,
                        help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-c','--config-file', type=str, default=None,
                        help='JSON file containing the configuration of the model')
    parser.add_argument('-n','--cell-name', type=str, default=None,
                        help='name of the cell as it appears in the configuration file')
    parser.add_argument('--hall-of-fame', action='store_true', help='compute population f-I curve')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=2000., type=float, help='stimulation duration (default: 2000 ms)')
    parser.add_argument('--tran', default=200., type=float, help='transient to be discard after stimulation onset (default: 200 ms)')
    args = parser.parse_args(args=sys.argv[1:])

    if args.mech_file is not None and args.config_file is not None:
        print('--mech-file and --config-file cannot both be present.')
        sys.exit(1)

    if args.config_file is not None and args.cell_name is None:
        print('You must specify --cell-name with --config-file.')
        sys.exit(1)

    if args.config_file is not None or args.cell_name is not None:
        import utils
        mechanisms = utils.extract_mechanisms(args.config_file, args.cell_name)
    else:
        mechanisms = json.load(open(args.mech_file,'r'))

    try:
        I = np.array([float(args.I)])
    except:
        if ',' in args.I:
            I = np.sort(np.array([float(x) for x in args.I.split(',')]))
        elif ':' in args.I:
            tmp = np.array([float(x) for x in args.I.split(':')])
            I = np.arange(tmp[0],tmp[1]+tmp[2]/2,tmp[2])
        else:
            print('Unknown current definition: %s.' % args.I)
            sys.exit(1)

    params_files = None
    if args.params_files is not None:
        params_files = args.params_files.split(',')

    if args.hall_of_fame:
        hall_of_fame = pickle.load(open('hall_of_fame.pkl','rb'))
        evaluator = pickle.load(open('evaluator.pkl','rb'))
        if args.config_file is not None:
            config = json.load(open(args.config_file,'r'))[args.cell_name]
            default_parameters = None
        elif params_files is not None:
            config = None
            default_parameters = json.load(open(params_files[0],'r'))
        else:
            print('If you do not specify a configuration file, the option --params-files ' + \
                  'indicates the name of the file where the default parameters are stored.')
            sys.exit(0)
        parameters = build_parameters_dict(hall_of_fame, evaluator, config, default_parameters)
        suffix = 'hall_of_fame'
    else:
        parameters = [json.load(open(f,'r')) for f in params_files]

    if len(parameters) > 1:
        f,no_spikes,inverse_first_isi,inverse_last_isi = compute_fI_curves(I*1e-3, args.swc_file, parameters, mechanisms, \
                                                                           args.delay, args.dur, args.tran, True)
        if args.hall_of_fame:
            suffix = 'hall_of_fame'
        else:
            suffix = make_suffix(params_files)
    else:
        f,no_spikes,inverse_first_isi,inverse_last_isi = compute_fI_curve(I*1e-3, args.swc_file, parameters[0], \
                                                                          mechanisms, args.delay, args.dur, \
                                                                          args.tran, 'MyCell', do_plot=True)
        suffix = os.path.basename(params_files[0]).split('.')[0]

    plt.savefig('fI_curve_%s.pdf' % suffix)
    plt.show()

    fI_curve = {'I': I, 'f': f,
                'no_spikes': no_spikes,
                'inverse_first_isi': inverse_first_isi,
                'inverse_last_isi': inverse_last_isi}

    pickle.dump(fI_curve, open('fI_curve_'+suffix+'.pkl','wb'))


if __name__ == '__main__':
    main()
