import os
import sys
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from dlutils import cell as cu
from dlutils import utils as dlu
import pickle
import json
import time


def inject_current_step(I, delay, dur, swc_file, parameters, mechanisms, replace_axon=False, \
                        add_axon_if_missing=True, cell_name=None, neuron=None, do_plot=False, verbose=False):

    if cell_name is None:
        import random
        cell_name = 'cell_%06d' % random.randint(0,999999)

    cell = cu.Cell(cell_name, swc_file, parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing)

    if neuron is None:
        h = cu.h
    else:
        h = neuron.h

    stim = h.IClamp(cell.morpho.soma[0](0.5))
    stim.delay = delay
    stim.dur = dur
    stim.amp = I*1e-3
    
    recorders = {'spike_times': h.Vector()}
    apc = h.APCount(cell.morpho.soma[0](0.5))
    apc.thresh = -20.
    apc.record(recorders['spike_times'])
    spike_times = []

    CA3_mech_vars = {'kca': 'gk', 'kap': 'gka', 'cat': 'gcat', 'cal': 'gcal', 'can': 'gcan', \
                     'cagk': 'gkca', 'kad': 'gka'}

    for lbl in 't','soma.v','soma.cai','soma.ica','axon.v','apic.v','basal.v':
        recorders[lbl] = h.Vector()
    recorders['t'].record(h._ref_t)
    recorders['soma.v'].record(cell.morpho.soma[0](0.5)._ref_v)
    recorders['soma.cai'].record(cell.morpho.soma[0](0.5)._ref_cai)
    recorders['soma.ica'].record(cell.morpho.soma[0](0.5)._ref_ica)

    if cell.n_axonal_sections > 0:
        recorders['axon.v'].record(cell.morpho.axon[0](0.5)._ref_v)
    else:
        print('The cell has no axon.')
    if cell.n_apical_sections > 0:
        for sec,dst in zip(cell.morpho.apic,cell.apical_path_lengths):
            if dst[0] >= 200:
                recorders['apic.v'].record(sec(0.5)._ref_v)
                break
    if cell.n_basal_sections > 0:
        for sec,dst in zip(cell.morpho.dend,cell.basal_path_lengths):
            if dst[0] >= 100:
                recorders['basal.v'].record(sec(0.5)._ref_v)
                break

    gbars = {}
    for mech in cell.morpho.soma[0](0.5):
        name = mech.name()
        if name in CA3_mech_vars:
            key = 'soma.' + CA3_mech_vars[name] + '_' + name
            recorders[key] = h.Vector()
            recorders[key].record(getattr(cell.morpho.soma[0](0.5), '_ref_' + CA3_mech_vars[name] + '_' + name))
            try:
                gbars[key] = getattr(cell.morpho.soma[0](0.5), 'gbar_' + name)
            except:
                try:
                    gbars[key] = getattr(cell.morpho.soma[0](0.5), 'g' + name + 'bar_' + name)
                except:
                    gbars[key] = getattr(cell.morpho.soma[0](0.5), 'g' + name[:2] + 'bar_' + name)

    try:
        rec = h.Vector()
        rec.record(cell.morpho.soma[0](0.5)._ref_m_kmb)
        recorders['soma.m_kmb'] = rec
        gbars['soma.m_kmb'] = 1
    except:
        pass

    h.cvode_active(1)
    h.tstop = stim.dur + stim.delay + 100

    fmt = lambda now: '%02d:%02d:%02d' % (now.tm_hour,now.tm_min,now.tm_sec)

    start = time.time()
    if verbose:
        print('{}>> I = {} pA started @ {}.'.format(cell_name,stim.amp*1e3,fmt(time.localtime(start))))

    h.run()
    stop = time.time()

    if verbose:
        print('{}>> I = {} pA finished @ {}, f = {} spikes/s elapsed time = {} seconds.'.format(
            cell_name,stim.amp*1e3,fmt(time.localtime(stop)),
            len(recorders['spike_times'])/dur*1e3,stop-start))

    if do_plot:
        fig,ax = plt.subplots(3, 1, sharex=True, figsize=(6,4))
        t = np.array(recorders['t'])
        ax[0].plot(t,recorders['soma.v'],'k',label='Soma')
        ax[0].plot(t,recorders['apic.v'],'r',label='Apical')
        ax[0].plot(t,recorders['basal.v'],'g',label='Basal')
        ax[0].set_ylabel(r'$V_m$ (mV)')
        ax[0].legend(loc='best')
        ax[1].plot(t,np.array(recorders['soma.cai'])*1e3,'k')
        ax[1].set_ylabel(r'$Ca_i$ ($\mu$M)')
        ax[2].plot(t,np.array(recorders['soma.ica'])*1e6,'k')
        ax[2].set_ylabel(r'$I_{Ca}$ (pA)')
        ax[2].set_xlabel('Time (ms)')
        ax[2].set_xlim([stim.delay - 50, stim.delay + stim.dur + 100])
        if len(gbars) > 0:
            fig,ax = plt.subplots(1, 1, figsize=(6,4))
            for name,rec in recorders.items():
                if 'soma' in name and not name.split('.')[1] in ('v','cai','ica'):
                    ax.plot(recorders['t'], np.array(rec)/gbars[name], label=name)
            ax.legend(loc='best')
            ax.set_xlim([stim.delay - 50, stim.delay + stim.dur + 100])
            ## phase-space plot: to be perfected...
            #fig,ax = plt.subplots(1, 1, figsize=(6,4))
            #idx, = np.where(t > 1000)
            #v = np.array(recorders['soma.v'])[idx]
            #m = np.array(recorders['soma.m_kmb'])[idx]
            #ax.plot(v,m,'k')
        plt.show()
        
    h('forall delete_section()')
    return recorders


def main():
    parser = arg.ArgumentParser(description='Compute the f-I curve of a neuron model.')
    parser.add_argument('I', type=float, action='store', help='current value in pA')
    parser.add_argument('--delay', default=500., type=float, help='delay before stimulation onset (default: 500 ms)')
    parser.add_argument('--dur', default=2000., type=float, help='stimulation duration (default: 2000 ms)')
    parser.add_argument('-f','--swc-file', type=str, help='SWC file defining the cell morphology', required=True)
    parser.add_argument('-p','--params-file', type=str, default=None,
                        help='JSON file containing the parameters of the model', required=True)
    parser.add_argument('-m','--mech-file', type=str, default=None,
                        help='JSON file containing the mechanisms to be inserted into the cell')
    parser.add_argument('-c','--config-file', type=str, default=None,
                        help='JSON file containing the configuration of the model')
    parser.add_argument('-n','--cell-name', type=str, default=None,
                        help='name of the cell as it appears in the configuration file')
    parser.add_argument('-R','--replace-axon', type=str, default=None,
                        help='whether to replace the axon (accepted values: "yes" or "no")')
    parser.add_argument('-A', '--add-axon-if-missing', type=str, default=None,
                        help='whether add an axon if the cell does not have one (accepted values: "yes" or "no")')
    parser.add_argument('-o','--output', type=str, default='step.pkl', help='output file name (default: step.pkl)')
    parser.add_argument('--plot', action='store_true', help='show a plot (default: no)')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose (default: no)')
    args = parser.parse_args(args=sys.argv[1:])

    if args.mech_file is not None and args.config_file is not None:
        print('--mech-file and --config-file cannot both be present.')
        sys.exit(1)

    if args.config_file is not None and args.cell_name is None:
        print('You must specify --cell-name with --config-file.')
        sys.exit(2)

    parameters = json.load(open(args.params_file,'r'))

    if args.config_file is not None or args.cell_name is not None:
        import dlutils
        mechanisms = dlu.extract_mechanisms(args.config_file, args.cell_name)
    else:
        mechanisms = json.load(open(args.mech_file,'r'))

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
            sys.exit(3)

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
            sys.exit(4)

    rec = inject_current_step(args.I, args.delay, args.dur, args.swc_file, parameters, mechanisms, \
                              replace_axon, add_axon_if_missing, do_plot=args.plot, verbose=args.verbose)
        
    step = {'I': args.I, 'time': np.array(rec['t']), 'voltage': np.array(rec['soma.v']), 'spike_times': np.array(rec['spike_times'])}

    pickle.dump(step, open(args.output,'wb'))


if __name__ == '__main__':
    main()
