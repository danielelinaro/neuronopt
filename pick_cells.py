
import os
import sys
import copy
import json
import glob
import pickle
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt

import efel
from neuron import h
from current_injection import inject_current_step
from dlutils.utils import *

efel.setThreshold(0)

class ColorFactory:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    def __init__(self):
        pass
    def __color__(self, col, str):
        return col + str + self.ENDC
    def red(self, str):
        return self.__color__(self.RED, str)
    def green(self, str):
        return self.__color__(self.GREEN, str)
    def blue(self, str):
        return self.__color__(self.BLUE, str)
    def yellow(self, str):
        return self.__color__(self.YELLOW, str)


colors = ColorFactory()

argsort = lambda seq: sorted(list(range(len(seq))), key=seq.__getitem__)


def worker(cell_id, **kwargs):
    swc_file = kwargs['swc_file']
    final_pop = kwargs['final_pop']
    evaluator = kwargs['evaluator']
    target_features = kwargs['target_features']
    protocols = kwargs['protocols']
    err_max = kwargs['err_max']
    verbose = kwargs['verbose']
    mechanisms = kwargs['mechs']
    add_axon_if_missing = kwargs['add_axon_if_missing']
    replace_axon = kwargs['replace_axon']
    individual = final_pop[cell_id]

    config = None
    default_parameters = None
    try:
        config = kwargs['config']
    except:
        default_parameters = kwargs['default_parameters']

    parameters = build_parameters_dict([individual], evaluator, config, default_parameters)[0]

    inj_loc, inj_dist = 'soma', 0
    good = True
    n_protocols = len(protocols)

    if verbose:
        print('----------------------------------')
        print(f'Final population individual #{cell_id+1}')
        printed_header = True
    else:
        printed_header = False

    for i, (proto_name, proto) in enumerate(protocols.items()):
        if verbose:
            print(f'  {proto_name}')
            printed_proto_name = True
        else:
            printed_proto_name = False

        cell_name = f'individual_{cell_id:03d}_{i}'

        amp, dur, delay = proto['stimuli'][0]['amp'] * 1e3, proto['stimuli'][0]['duration'], proto['stimuli'][0]['delay']

        try:
            apical_rec_sites = {extra_rec['name']: extra_rec['somadistance'] for extra_rec in proto['extra_recordings'] \
                                if extra_rec['seclist_name'] == 'apical'}
            basal_rec_sites = {extra_rec['name']: extra_rec['somadistance'] for extra_rec in proto['extra_recordings'] \
                               if extra_rec['seclist_name'] == 'basal'}
        except:
            apical_rec_sites, basal_rec_sites = {}, {}

        recorders = inject_current_step(amp, delay, dur, swc_file, \
                                        inj_loc, inj_dist, parameters, mechanisms,
                                        after = proto['stimuli'][0]['totduration'] - delay - dur,
                                        cell_name = cell_name,
                                        add_axon_if_missing = add_axon_if_missing,
                                        replace_axon = replace_axon,
                                        apical_dst = apical_rec_sites,
                                        basal_dst = basal_rec_sites,
                                        do_plot = False)
        h('forall delete_section()')
        time = np.array(recorders['t'])

        for site in target_features[proto_name]:
            feature_names = list(target_features[proto_name][site].keys())
            thresh = None
            for obj in evaluator.fitness_calculator.objectives:
                if proto_name + '.' + site in obj.features[0].name:
                    thresh = obj.features[0].threshold
                    break
            if thresh is None:
                raise Exception(f'Cannot find the threshold value for protocol {proto_name} and site {site}')
            if verbose:
                print(f'Setting threshold for protocol {proto_name} and site {site} to {thresh} mV.')
            efel.setThreshold(thresh)
            feature_values = efel.getFeatureValues([
                {'T': time, 'V': np.array(recorders[site + '.v']),
                 'stim_start': [delay], 'stim_end': [delay + dur]}
            ], feature_names)
            for feature_name in feature_names:
                if feature_values[0][feature_name] is not None:
                    m = target_features[proto_name][site][feature_name][0]
                    s = target_features[proto_name][site][feature_name][1]
                    err = np.abs(np.mean(feature_values[0][feature_name]) - m) / s
                    if err > err_max and feature_name not in features_to_ignore:
                        good = False
                        if not verbose and not printed_header:
                            print('----------------------------------')
                            print(f'Final population individual #{i+1}')
                            printed_header = True
                        if not verbose and not printed_proto_name:
                            print(f'  {proto_name}.{site}')
                            printed_proto_name = True
                        print(colors.red(f'    {feature_name} = {err}'))
                    elif verbose and err > err_max and feature_name in features_to_ignore:
                        print(colors.yellow(f'    {feature_name} = {err}'))
                    elif verbose:
                        print(colors.green(f'    {feature_name} = {err}'))
                else:
                    if not printed_header:
                        print('----------------------------------')
                        print(f'Final population individual #{i+1}')
                        printed_header = True
                    if not printed_proto_name:
                        print(f'  {proto_name}.{site}')
                        printed_proto_name = True
                    print(colors.blue(f'    {feature_name} was not computed'))

        if not good:
            print('Individual ' + colors.red(f'{cell_id+1:03d}') + ' of the final population ' + \
                  colors.red('does not match') +  ' the requisites.')
            return False

    print('Individual ' + colors.green(f'{cell_id+1:03d}') + ' of the final population ' + \
          colors.green('matches') + ' the requisites.')
    return True


if __name__ == '__main__':

    parser = arg.ArgumentParser(description='Select cells that match a certain quality threshold.')
    parser.add_argument('folder', type=str, action='store', help='folder containing the results of the optimization')
    parser.add_argument('-t', '--threshold', default=5., type=float, help='threshold on the number of STDs to accept a solution')
    parser.add_argument('--features', default='all', type=str, help='comma-separated list of features to consider (default: all)')
    parser.add_argument('--ignore', default=None, type=str, help='comma-separated list of features to ignore')
    parser.add_argument('-a', '--all', action='store_true', help='process all solutions (potentially very time consuming)')
    parser.add_argument('-p', '--parallel', action='store_true', help='use SCOOP to integrate the individuals in a parallel fashion')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    args = parser.parse_args(args=sys.argv[1:])

    folder = args.folder
    if not os.path.isdir(folder):
        print(f'{os.path.basename(sys.argv[0])}: {folder}: no such directory.')
        sys.exit(1)

    err_max = args.threshold
    verbose = args.verbose

    features_to_consider = []
    if args.features != 'all':
        features_to_consider = args.features.split(',')

    features_to_ignore = []
    if args.ignore is not None:
        features_to_ignore = args.ignore.split(',')

    for feature in features_to_ignore:
        if feature in features_to_consider:
            print(f'Feature "{feature}" is both to consider and to ignore.')
            sys.exit(1)

    final_pop = np.array(pickle.load(open(folder + '/final_population.pkl', 'rb'), encoding='latin1'))
    # remove duplicate individuals
    final_pop = np.unique(final_pop, axis=0)
    hof = np.array(pickle.load(open(folder + '/hall_of_fame.pkl', 'rb'), encoding='latin1'))
    hof_responses = pickle.load(open(folder + '/hall_of_fame_responses.pkl', 'rb'), encoding='latin1')
    n_individuals,n_parameters = final_pop.shape
    evaluator = pickle.load(open(folder + '/evaluator.pkl', 'rb'))
    protocols = json.load(open(folder + '/protocols.json'))
    target_features = json.load(open(folder + '/features.json','r'))

    good_individuals_hof = []
    for i,(individual,response) in enumerate(zip(hof,hof_responses)):
        if verbose:
            print('----------------------------------')
            print(f'Hall-of-fame individual #{i+1}')
            printed_header = True
        else:
            printed_header = False
        good = True
        for step_name, trace in response.items():
            proto_name, site, _ = step_name.split('.')
            if verbose:
                print(f'  {proto_name}.{site}')
                printed_step_name = True
            else:
                printed_step_name = False
            stim_start = evaluator.fitness_protocols[proto_name].stimuli[0].step_delay
            stim_end = stim_start + evaluator.fitness_protocols[proto_name].stimuli[0].step_duration
            feature_names = list(target_features[proto_name][site].keys())
            thresh = None
            for obj in evaluator.fitness_calculator.objectives:
                if proto_name + '.' + site in obj.features[0].name:
                    thresh = obj.features[0].threshold
                    break
            if thresh is None:
                raise Exception(f'Cannot find the threshold value for protocol {proto_name} and site {site}')
            if verbose:
                print(f'Setting threshold for protocol {proto_name} and site {site} to {thresh} mV.')
            efel.setThreshold(thresh)
            feature_values = efel.getFeatureValues([
                {'T': trace['time'], 'V': trace['voltage'],
                 'stim_start': [stim_start], 'stim_end': [stim_end]}
            ], feature_names)
            for feature_name in feature_names:
                if feature_name in feature_values[0]:
                    m = target_features[proto_name][site][feature_name][0]
                    s = target_features[proto_name][site][feature_name][1]
                    err = np.abs(np.mean(feature_values[0][feature_name]) - m) / s
                    if err > err_max and feature_name not in features_to_ignore:
                        good = False
                        if not verbose and not printed_header:
                            print('----------------------------------')
                            print(f'Hall-of-fame individual #{i+1}')
                            printed_header = True
                        if not verbose and not printed_step_name:
                            print(f'  {proto_name}.{site}')
                            printed_proto_name = True
                        print(colors.red(f'    {feature_name} = {err}'))
                    elif verbose and err > err_max and feature_name in features_to_ignore:
                        print(colors.yellow(f'    {feature_name} = {err}'))
                    elif verbose:
                        print(colors.green(f'    {feature_name} = {err}'))
                else:
                    if not printed_header:
                        print('----------------------------------')
                        print(f'Final population individual #{i+1}')
                        printed_header = True
                    if not printed_proto_name:
                        print(f'  {proto_name}.{site}')
                        printed_proto_name = True
                    print(colors.blue(f'    {feature_name} was not computed'))

        if good:
            good_individuals_hof.append(i)
            print('Individual ' + colors.green(f'{i+1:03d}') + ' of the hall-of-fame ' + colors.green('matches') + ' the requisites.')
        else:
            print('Individual ' + colors.red(f'{i+1:03d}') + ' of the hall-of-fame ' + colors.red('does not match') + ' the requisites.')

    data = {'good_individuals_hof': good_individuals_hof,
            'good_population': hof[good_individuals_hof,:],
            'err_max': err_max,
            'features': features_to_consider,
            'ignored_features': features_to_ignore}

    if args.all:

        good_individuals_also_in_hof = []
        for i,individual in enumerate(final_pop):
            idx = (individual == hof).all(axis = 1).nonzero()[0]
            if len(idx) > 0:
                if idx[0] in good_individuals_hof:
                    good_individuals_also_in_hof.append(i)
                    print('Individual ' + colors.green(f'{i+1:03d}') + f' is in the hall-of-fame (#{idx[0]+1}) and ' + \
                          colors.green('matches') + ' the requisites.')
                else:
                    print('Individual ' + colors.red(f'{i+1:03d}') + f' is in the hall-of-fame (#{idx[0]+1}) and ' + \
                          colors.red('does not match') + ' the requisites.')

        swc_file = glob.glob(folder + '/*.converted.swc')[0]

        mech_file = folder + '/mechanisms.json'
        if os.path.isfile(mech_file):
            mechs = json.load(open(mech_file,'r'))
            default_parameters = json.load(open(folder + '/parameters.json','r'))
        else:
            mech_file = None
            cell_name = '_'.join(os.path.split(os.path.abspath(folder))[1].split('_')[1:])
            config_file = folder + '/parameters.json'
            config = json.load(open(config_file,'r'))[cell_name]
            mechs = extract_mechanisms(config_file, cell_name)

        sim_pars = pickle.load(open(folder + '/simulation_parameters.pkl','rb'))

        kwargs = {'swc_file': swc_file,
                  'mechs': mechs,
                  'final_pop': final_pop,
                  'evaluator': evaluator,
                  'target_features': target_features,
                  'protocols': protocols,
                  'err_max': err_max,
                  'replace_axon': sim_pars['replace_axon'],
                  'add_axon_if_missing': not sim_pars['no_add_axon'],
                  'verbose': verbose}

        if mech_file is None:
            kwargs['config'] = config
        else:
            kwargs['default_parameters'] = default_parameters

        individuals = [i for i in range(n_individuals) if i not in good_individuals_also_in_hof]

        map_func = map
        try:
            if args.parallel:
                from scoop import futures
                map_func = futures.map
        except:
            print('SCOOP not found: will run serially.')

        good = list(map_func(lambda index: worker(index, **kwargs), individuals))

        good_individuals = [ind for ind,gd in zip(individuals,good) if gd]
        data['good_individuals'] = good_individuals
        data['good_population'] = np.concatenate((data['good_population'], final_pop[good_individuals,:]))

    pickle.dump(data, open(folder + f'/good_population_{err_max:g}_STDs.pkl', 'wb'))

