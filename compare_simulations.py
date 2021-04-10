
import os
import sys
import pickle
import json
import glob
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt

from bluepyopt.ephys.simulators import NrnSimulator

from dlutils import cell as cu

progname = os.path.basename(sys.argv[0])

RED = '\033[91m'
GREEN = '\033[92m'
STOP = '\033[0m'


def equal_sections(sec_a, sec_b, h, soma_a=None, soma_b=None):
    if sec_a.nseg != sec_b.nseg:
        print('{:>25s} <> {:>17s} - {} <> {} segments' \
              .format(sec_a.name(), sec_b.name(), RED+str(sec_a.nseg)+STOP, RED+str(sec_b.nseg)+STOP))
        return False
    try:
        n3d_a = sec_a.n3d()
        n3d_b = sec_b.n3d()
    except:
        n3d_a = int(h.n3d(sec=sec_a))
        n3d_b =  int(h.n3d(sec=sec_b))
    if n3d_a != n3d_b:
        print('{} <> {}'.format(sec_a.name(), sec_b.name()))
        print('{} <> {} points.'.format(n3d_a, n3d_b))
        return False
    if n3d_a == 0:
        good = True
        if sec_a.L != sec_b.L:
            good = False
            print('{:>25s} <> {:>17s} - L = {} <> {}' \
                  .format(sec_a.name(), sec_b.name(), RED+str(sec_a.L)+STOP, RED+str(sec_b.L)+STOP))
        if sec_a.diam != sec_b.diam:
            good = False
            print('{:>25s} <> {:>17s} - diam = {} <> {}' \
                  .format(sec_a.name(), sec_b.name(), RED+str(sec_a.diam)+STOP, RED+str(sec_b.diam)+STOP))
        if not good:
            return False
    else:
        for i in range(n3d_a):
            try:
                pt_a = np.array([h.x3d(i,sec=sec_a),
                                 h.y3d(i,sec=sec_a),
                                 h.z3d(i,sec=sec_a)])
                pt_b = np.array([h.x3d(i,sec=sec_b),
                                 h.y3d(i,sec=sec_b),
                                 h.z3d(i,sec=sec_b)])
            except:
                pt_a = np.array([sec_a.x3d(i),
                                 sec_a.y3d(i),
                                 sec_a.z3d(i)])
                pt_b = np.array([sec_b.x3d(i),
                                 sec_b.y3d(i),
                                 sec_b.z3d(i)])
            if np.any(pt_a != pt_b):
                print('{} <> {}'.format(sec_a.name(), sec_b.name()))
                print('{} <> {}.'.format(pt_a, pt_b))
                return False
    if sec_a.Ra != sec_b.Ra:
        print('{} <> {}'.format(sec_a.name(), sec_b.name()))
        print('Ra: {} <> {}.'.format(sec_a.Ra, sec_b.Ra))
        return False
    if sec_a.cm != sec_b.cm:
        print('{} <> {}'.format(sec_a.name(), sec_b.name()))
        print('Cm: {} <> {}.'.format(sec_a.cm, sec_b.cm))
        return False
    not_mech_names = 'v','diam','cm'
    for seg_a,seg_b in zip(sec_a,sec_b):
        if seg_a.__dict__.keys() != seg_b.__dict__.keys():
            print('{} <> {}'.format(sec_a.name(), sec_b.name()))
            print('{} <> {}.'.format(seg_a.__dict__.keys(), seg_b.__dict__.keys()))
        mech_names = [k for k in seg_a.__dict__.keys() if k not in ('v','diam','cm')]
        for mech_name in mech_names:
            mech_a = seg_a.__getattribute__(mech_name)
            mech_b = seg_b.__getattribute__(mech_name)
            if mech_a.__dict__.keys() != mech_b.__dict__.keys():
                print('{} <> {}'.format(sec_a.name(), sec_b.name()))
                print('{} <> {}.'.format(mech_a.__dict__.keys(), mech_b.__dict__.keys()))
                return False
            if mech_name in ('ca_ion','k_ion','na_ion'):
                continue
            for k in mech_a.__dict__.keys():
                attr_a = mech_a.__getattribute__(k)
                attr_b = mech_b.__getattribute__(k)
                if attr_a != attr_b: # and 'bar' in k
                    print('{} <> {}'.format(sec_a.name(), sec_b.name()))
                    print('{}: {} <> {}.'.format(k, attr_a, attr_b))
                    return False
    return True


def main():
    parser = arg.ArgumentParser(description='Compare simulations.')
    parser.add_argument('folder', type=str, action='store', help='folder where configuration  files are located')
    parser.add_argument('-i','--individual', type=int, default=0, help='index of the individual to simulate')
    parser.add_argument('-p','--protocol', type=int, default=1, help='index of the protocol to simulate')
    parser.add_argument('--compare-sections', action='store_true', help='compare all sections properties')
    parser.add_argument('--run-evaluator', action='store_true', help='run evaluator simulations')

    args = parser.parse_args(args=sys.argv[1:])
    
    folder = args.folder
    if not os.path.isdir(folder):
        print('%s: %s: no such directory.' % (progname, folder))
        sys.exit(1)
    if folder[-1] != '/':
        folder += '/'
              
    individual_id = args.individual
    if individual_id < 0:
        print('%s: individual id must be >= 0.' % progname)
        sys.exit(1)

    protocol_id = args.protocol - 1 
    if protocol_id < 0:
        print('%s: protocol id must be > 0.' % progname)
        sys.exit(1)

    swc_file = glob.glob(folder + '*.swc')[0]
    cell_name = '_'.join(os.path.split(os.path.abspath(folder))[-1].split('_')[1:])

    try:
        sim_pars = pickle.load(open('simulation_parameters.pkl','rb'))
        replace_axon = sim_pars['replace_axon']
        add_axon_if_missing = not sim_pars['no_add_axon']
    except:
        replace_axon = False
        add_axon_if_missing = False

    print('      SWC file: {}'.format(swc_file))
    print('     Cell name: {}'.format(cell_name))
    print(' Individual ID: {}'.format(individual_id))
    print('   Protocol ID: {}'.format(protocol_id))
    print('  Replace axon: {}'.format(replace_axon))
    print('      Add axon: {}'.format(add_axon_if_missing))

    # load the parameters and instantiate the cell
    parameters = json.load(open(folder + 'individual_%d.json' % individual_id,'r'))
    mechanisms = json.load(open(folder + 'parameters.json','r'))[cell_name]['mechanisms']
    if 'alldend' in mechanisms:
        mechanisms['basal'] = mechanisms['alldend']
        mechanisms['apical'] = mechanisms['alldend']
        mechanisms.pop('alldend')
    cell = cu.Cell('cell', swc_file, parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing)

    # load the protocols and pick the right one
    protocols = json.load(open(folder + 'protocols.json','r'))
    protocols_names = list(protocols.keys())
    protocol_name = protocols_names[protocol_id]
    protocol_full_name = protocol_name + '.soma.v'

    # instantiate the Neuron simulator
    sim = NrnSimulator()

    if args.run_evaluator or args.compare_sections:
        evaluator = pickle.load(open(folder + 'evaluator.pkl','rb'))
        evaluator.cell_model.morphology.morphology_path = swc_file
        hof = pickle.load(open(folder + 'hall_of_fame.pkl','rb'), encoding='latin1')

    if args.compare_sections:
        for k,v in evaluator.param_dict(hof[individual_id]).items():
            evaluator.cell_model.params[k].value = v
        evaluator.cell_model.instantiate(sim)

        sections = {key: [] for key in ('eval','sim')}
        for sec in sim.neuron.h.allsec():
            if cell_name in sec.name():
                sections['eval'].append(sec)
            else:
                sections['sim'].append(sec)

        for sec_a,sec_b in zip(sections['eval'],sections['sim']):
            flag = equal_sections(sec_a, sec_b, sim.neuron.h, sections['eval'][0], sections['sim'][0])
            if flag:
                msg = GREEN + 'OK' + STOP
            else:
                msg = RED + 'NOT OK' + STOP
            print('{:>25s} <> {:>17s} - {}'.format(sec_a.name(),sec_b.name(),msg))

    if args.run_evaluator:
        eval_response = evaluator.run_protocols(protocols=[evaluator.fitness_protocols[protocol_name]],
                                                param_values=evaluator.param_dict(hof[individual_id]))[protocol_full_name]

    # prepare the stimulus
    stim = sim.neuron.h.IClamp(cell.morpho.soma[0](0.5))
    stim.delay = protocols[protocol_name]['stimuli'][0]['delay']
    stim.amp = protocols[protocol_name]['stimuli'][0]['amp']
    stim.dur = protocols[protocol_name]['stimuli'][0]['duration']

    # create the recorders
    rec = {key: sim.neuron.h.Vector() for key in ('t','V')}
    rec['t'].record(sim.neuron.h._ref_t)
    rec['V'].record(cell.morpho.soma[0](0.5)._ref_v)

    # run the simulation
    if not args.run_evaluator:
        sim.neuron.h.load_file('stdrun.hoc')
        sim.neuron.h.celsius = 34
        sim.neuron.h.cvode_active(1)
    sim.neuron.h.tstop = protocols[protocol_name]['stimuli'][0]['totduration']
    sim.neuron.h.run()

    # load the response
    hof_response = pickle.load(open(folder + 'hall_of_fame_responses.pkl','rb'), \
                               encoding='latin1')[individual_id][protocol_full_name]

    # plot the results
    plt.plot(hof_response['time'],hof_response['voltage'],'k',label='Response')
    if args.run_evaluator:
        plt.plot(eval_response['time'],eval_response['voltage'],'r',label='Evaluator')
    plt.plot(rec['t'],rec['V'],'g',label='Simulated')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane voltage (mV)')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
