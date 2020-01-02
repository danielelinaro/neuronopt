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



def make_cell(swc_file, parameters, mechanisms, cell_name=None, replace_axon=False, add_axon_if_missing=True):
    if cell_name is None:
        import random
        cell_name = 'cell_%06d' % random.randint(0,999999)

    cell = cu.Cell(cell_name, swc_file, parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing)
    return cell


def make_recorders(cell, h, apical_dst=None, basal_dst=None, mech_vars=None):
    recorders = {'spike_times': h.Vector()}
    apc = h.APCount(cell.morpho.soma[0](0.5))
    apc.thresh = -20.
    apc.record(recorders['spike_times'])

    for lbl in 't','soma.v','soma.cai','soma.ica':
        recorders[lbl] = h.Vector()
    recorders['t'].record(h._ref_t)
    recorders['soma.v'].record(cell.morpho.soma[0](0.5)._ref_v)
    recorders['soma.cai'].record(cell.morpho.soma[0](0.5)._ref_cai)
    recorders['soma.ica'].record(cell.morpho.soma[0](0.5)._ref_ica)

    if cell.n_axonal_sections > 0:
        recorders['axon.v'] = h.Vector()
        recorders['axon.v'].record(cell.morpho.axon[0](0.5)._ref_v)
    else:
        print('The cell has no axon.')

    if apical_dst is not None and cell.n_apical_sections > 0:
        h.distance(0, 0.5, sec=cell.morpho.soma[0])
        for sec in cell.morpho.apic:
            for seg in sec:
                for k,v in apical_dst.items():
                    if not k+'.v' in recorders and h.distance(1, seg.x, sec=sec) >= v:
                        recorders[k+'.v'] = h.Vector()
                        recorders[k+'.v'].record(seg._ref_v)
                        print('Adding recorder on the apical dendrite at a distance of {:.2f} um.'\
                              .format(h.distance(1, seg.x, sec=sec)))
    if basal_dst is not None and cell.n_basal_sections > 0:
        h.distance(0, 0.5, sec=cell.morpho.soma[0])
        for sec in cell.morpho.basal:
            for seg in sec:
                for k,v in basal_dst.items():
                    if not k+'.v' in recorders and h.distance(1, seg.x, sec=sec) >= v:
                        recorders[k+'.v'] = h.Vector()
                        recorders[k+'.v'].record(seg._ref_v)
                        print('Adding recorder on the basal dendrite at a distance of {:.2f} um.'\
                              .format(h.distance(1, seg.x, sec=sec)))

    if mech_vars is None:
        return recorders

    gbars = {}
    for mech in cell.morpho.soma[0](0.5):
        name = mech.name()
        if name in mech_vars:
            key = 'soma.' + mech_vars[name] + '_' + name
            recorders[key] = h.Vector()
            recorders[key].record(getattr(cell.morpho.soma[0](0.5), '_ref_' + mech_vars[name] + '_' + name))
            try:
                gbars[key] = getattr(cell.morpho.soma[0](0.5), 'gbar_' + name)
            except:
                try:
                    gbars[key] = getattr(cell.morpho.soma[0](0.5), 'g' + name + 'bar_' + name)
                except:
                    gbars[key] = getattr(cell.morpho.soma[0](0.5), 'g' + name[:2] + 'bar_' + name)

    return recorders,gbars


def run_simulation(tstop, h, verbose=False):
    h.cvode_active(1)
    h.tstop = tstop
    fmt = lambda now: '%02d:%02d:%02d' % (now.tm_hour,now.tm_min,now.tm_sec)
    start = time.time()
    if verbose:
        print('{}>> simulation started @ {}.'.format(cell_name,fmt(time.localtime(start))))
    h.run()
    stop = time.time()
    if verbose:
        print('{}>> simulation finished @ {}, elapsed time = {} seconds.'.format(
            cell_name,fmt(time.localtime(stop)),stop-start))


def plot_results(recorders, apical_dst=None, basal_dst=None, gbars=None, x_lim=None):
    apical_col = 'rm'
    basal_col = 'gc'
    fig,ax = plt.subplots(3, 1, sharex=True, figsize=(6,4))
    t = np.array(recorders['t'])
    if apical_dst is not None:
        for i,(k,v) in enumerate(apical_dst.items()):
            if k+'.v' in recorders:
                ax[0].plot(t,recorders[k+'.v'],apical_col[i],label='Apical - {:.0f} um'.format(v))
    if basal_dst is not None:
        for i,(k,v) in enumerate(basal_dst.items()):
            if k+'.v' in recorders:
                ax[0].plot(t,recorders[k+'.v'],basal_col[i],label='Basal - {:.0f} um'.format(v))
    ax[0].plot(t,recorders['soma.v'],'k',label='Soma')
    ax[0].set_ylabel(r'$V_m$ (mV)')
    ax[0].legend(loc='best')
    ax[1].plot(t,np.array(recorders['soma.cai'])*1e3,'k')
    ax[1].set_ylabel(r'$Ca_i$ ($\mu$M)')
    ax[2].plot(t,np.array(recorders['soma.ica'])*1e6,'k')
    ax[2].set_ylabel(r'$I_{Ca}$ (pA)')
    ax[2].set_xlabel('Time (ms)')
    if x_lim is not None:
        ax[2].set_xlim(x_lim)
    plt.savefig('step.pdf')
    if gbars is not None:
        fig,ax = plt.subplots(1, 1, figsize=(6,4))
        for name,rec in recorders.items():
            if 'soma' in name and not name.split('.')[1] in ('v','cai','ica'):
                ax.plot(recorders['t'], np.array(rec)/gbars[name], label=name)
        ax.legend(loc='best')
        ax.set_xlim([delay - 50, delay + dur + 100])
    plt.show()


def inject_current_step(I, delay, dur, swc_file, inj_loc, inj_dist, parameters, mechanisms, N=1, freq=np.inf, replace_axon=False, \
                        add_axon_if_missing=True, cell_name=None, neuron=None, do_plot=False, verbose=False):

    if neuron is not None:
        h = neuron.h
    else:
        from neuron import h

    cell = make_cell(swc_file, parameters, mechanisms, cell_name, replace_axon, add_axon_if_missing)

    if not isinstance(I,list):
        I = [I for _ in range(N)]
    elif len(I) == 1:
        I = [I[0] for _ in range(N)]

    if len(I) != N:
        raise Exception('Number of stimuli does not agree with number of current values')


    if inj_loc == 'soma':
        inj_seg = cell.morpho.soma[0](0.5)
    else:
        inj_seg = None
        h.distance(0, 0.5, sec=cell.morpho.soma[0])
        if inj_loc == 'apical':
            sections = cell.morpho.apic
        elif inj_loc == 'basal':
            sections = cell.morpho.basal
        elif inj_loc == 'axon':
            sections = cell.morpho.axon
        else:
            raise Exception('Unknown group "{}"'.format(inj_loc))
        for sec in sections:
            for seg in sec:
                if h.distance(1, seg.x, sec=sec) >= inj_dist:
                    inj_seg = seg
                    print('Adding stimulus on "{}" at a distance of {:.2f} um.'\
                          .format(inj_loc, h.distance(1, seg.x, sec=sec)))
                    break
            if inj_seg is not None:
                break
        if inj_seg is None:
            raise Exception('No segment on "{}" at a distance of {:.2f} um'.format(inj_loc, inj_dist))

    stimuli = [h.IClamp(inj_seg) for _ in range(N)]
    for i,(stim,amp) in enumerate(zip(stimuli,I)):
        stim.delay = delay + i/freq*1000
        stim.dur = dur
        stim.amp = amp*1e-3

    apical_dst = {'apic1': 500, 'apic2': 600}
    basal_dst = {'basal1': 100, 'basal2': 200}
    recorders = make_recorders(cell, h, apical_dst, basal_dst)

    #CA3_mech_vars = {'kca': 'gk', 'kap': 'gka', 'cat': 'gcat', 'cal': 'gcal', 'can': 'gcan', \
    #                 'cagk': 'gkca', 'kad': 'gka'}
    #recorders,gbars = make_recorders(cell, apical_dst, basal_dst, CA3_mech_vars)
    #try:
    #    rec = h.Vector()
    #    rec.record(cell.morpho.soma[0](0.5)._ref_m_kmb)
    #    recorders['soma.m_kmb'] = rec
    #    gbars['soma.m_kmb'] = 1
    #except:
    #    pass

    t_end = dur + (N-1) / freq * 1000 + delay + 100
    run_simulation(t_end, h, verbose)

    if do_plot:
        plot_results(recorders, apical_dst, basal_dst, x_lim=[delay-50, t_end])

    h('forall delete_section()')
    return recorders


def main():
    parser = arg.ArgumentParser(description='Compute the f-I curve of a neuron model.')
    parser.add_argument('I', type=float, action='store', nargs='+', help='current value in pA')
    parser.add_argument('--dur', required=True, type=float, help='stimulus duration')
    parser.add_argument('--delay', default=1000., type=float, help='delay before stimulus onset (default: 1000 ms)')
    parser.add_argument('-N','--number', default=1, type=int, help='number of current steps (default 1)')
    parser.add_argument('-F','--frequency', default=np.inf, type=float, help='frequency of current steps (default +inf)')
    parser.add_argument('--injection-site', default='soma', type=str, help='injection site (default soma)')
    parser.add_argument('--injection-site-distance', default=0, type=float, help='injection distance (default 0 um)')
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

    if args.number < 1:
        print('The number of current steps must be >= 1')
        sys.exit(1)

    if args.frequency <= 0:
        print('The frequency of the current steps must be > 0')
        sys.exit(2)

    if not args.injection_site in ('soma','apical','basal','axon'):
        print('Unknown injection site: accepted values are "soma", "apical", "basal" and "axon".')
        sys.exit(3)

    if args.injection_site_distance < 0:
        print('Injection distance must be > 0.')
        sys.exit(4)

    if args.mech_file is not None and args.config_file is not None:
        print('--mech-file and --config-file cannot both be present.')
        sys.exit(5)

    if args.config_file is not None and args.cell_name is None:
        print('You must specify --cell-name with --config-file.')
        sys.exit(6)

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

    rec = inject_current_step(args.I, args.delay, args.dur, args.swc_file, args.injection_site, \
                              args.injection_site_distance, parameters, mechanisms, \
                              args.number, args.frequency, replace_axon, add_axon_if_missing, \
                              do_plot=args.plot, verbose=args.verbose)
        
    step = {'I': args.I, 'time': np.array(rec['t']), 'voltage': np.array(rec['soma.v']), 'spike_times': np.array(rec['spike_times'])}

    pickle.dump(step, open(args.output,'wb'))


if __name__ == '__main__':
    main()
