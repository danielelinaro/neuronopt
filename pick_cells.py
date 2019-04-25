#!/usr/bin/env python

from __future__ import print_function

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
from current_step import inject_current_step


def equal_sections(sec_a, sec_b, h, soma_a=None, soma_b=None):
    if sec_a.nseg != sec_b.nseg:
        print('{} <> {}'.format(sec_a.name(), sec_b.name()))
        print('{} <> {} segments.'.format(sec_a.nseg, sec_b.n_seg))
        return False
    n3d_a = int(h.n3d(sec=sec_a))
    n3d_b = int(h.n3d(sec=sec_b))
    if n3d_a != n3d_b:
        print('{} <> {}'.format(sec_a.name(), sec_b.name()))
        print('{} <> {} points.'.format(n3d_a, n3d_b))
        return False
    for i in range(n3d_a):
        pt_a = np.array([h.x3d(i,sec=sec_a),
                         h.y3d(i,sec=sec_a),
                         h.z3d(i,sec=sec_a)])
        pt_b = np.array([h.x3d(i,sec=sec_b),
                         h.y3d(i,sec=sec_b),
                         h.z3d(i,sec=sec_b)])
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
    for seg_a,seg_b in zip(sec_a,sec_b):
        for mech_a,mech_b in zip(seg_a,seg_b):
            if mech_a.name() != mech_b.name():
                print('{} <> {}'.format(sec_a.name(), sec_b.name()))
                print('{} <> {}.'.format(mech_a.name(),mech_b.name()))
                return False                
            if mech_a.__dict__.keys() != mech_b.__dict__.keys():
                print('{} <> {}'.format(sec_a.name(), sec_b.name()))
                print('{} <> {}.'.format(mech_a.__dict__.keys(),mech_b.__dict__.keys()))
                return False
            if mech_a.name() in ('ca_ion','k_ion','na_ion'):
                continue
            for k in mech_a.__dict__.keys():
                attr_a = mech_a.__getattribute__(k)
                attr_b = mech_b.__getattribute__(k)
                if k == 'gIhbar' and attr_a != attr_b:
                    h.distance(sec=soma_a)
                    dst = h.distance(seg_a.x, sec=sec_a)
                    value = attr_a
                    attr_a = ( -0.8696 + 2.087 * np.exp(dst * 0.0031) ) * value
                    mech_a.gIhbar = attr_a
                if attr_a != attr_b:
                    print('{} <> {}'.format(sec_a.name(), sec_b.name()))
                    print('{}: {} <> {}.'.format(k, attr_a, attr_b))
                    return False
    return True


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


def dump_parameters(parameters,default_parameters,evaluator,filename):
    param_dict = evaluator.param_dict(parameters)
    default_parameters_copy = copy.deepcopy(default_parameters)
    for par in default_parameters_copy:
        if 'value' not in par:
            par['value'] = param_dict[par['param_name'] + '.' + par['sectionlist']]
            par.pop('bounds')
    json.dump(default_parameters_copy,open(filename,'w'),indent=4)


argsort = lambda seq: sorted(range(len(seq)), key=seq.__getitem__)


def worker(cell_id, args):
    swc_file = args['swc_file']
    mech_file = args['mech_file']
    final_pop = args['final_pop']
    default_parameters = args['default_parameters']
    evaluator = args['evaluator']
    I = args['I']
    stim_dur = args['stim_dur']
    stim_start = args['stim_start']
    feature_names = args['feature_names']
    feature_reference_values = args['feature_reference_values']
    err_max = args['err_max']
    verbose = args['verbose']
    individual = final_pop[cell_id]
    params_file = '/tmp/individual_%03d.json' % cell_id
    dump_parameters(individual, default_parameters, evaluator, params_file)
    stim_end = stim_start + stim_dur

    n_steps = len(I)
    for i in range(n_steps):
        good = True
        cell_name = 'individual_%03d_%d' % (cell_id,i)
        recorders = inject_current_step(I[i], swc_file, mech_file, params_file, stim_start, stim_dur, cell_name, do_plot=False)
        h('forall delete_section()')
        trace = {'T': recorders['t'], 'V': recorders['Vsoma'], 'stim_start': [stim_start], 'stim_end': [stim_end]}
        feature_values = efel.getFeatureValues([trace],feature_names)
        for name in feature_names:
            if feature_values[0][name] is None:
                if verbose:
                    print(colors.red('[%d] %s is None.' % (cell_id,name)))
                good = False
                continue
            m = feature_reference_values[name][i][0]
            s = feature_reference_values[name][i][1]
            err = np.abs(np.mean(feature_values[0][name]) - m) / s
            if err > err_max:
                if verbose:
                    print(colors.red('[%d] %s = %g' % (cell_id,name,err)))
                good = False
            elif verbose:
                print(colors.green('[%d] %s = %g' % (cell_id,name,err)))

        if not good:
            print('Individual ' + colors.red('%03d does not match' % (cell_id+1)) + ' the requisites')
            return False

    print('Individual ' + colors.green('%03d matches'%(cell_id+1)) + ' the requisites')
    return True


if __name__ == '__main__':

    parser = arg.ArgumentParser(description='Select cells that match a certain quality threshold.')
    parser.add_argument('folder', type=str, action='store', help='folder containing the results of the optimization')
    parser.add_argument('-t', '--threshold', default=3, type=int, help='threshold on the number of STDs to accept a solution')
    parser.add_argument('--features', default='all', type=str, help='comma-separated list of features to consider (default: all)')
    parser.add_argument('--ignore', default=None, type=str, help='comma-separated list of features to ignore')
    parser.add_argument('-a', '--all', action='store_true', help='process all solutions (potentially very time consuming)')
    parser.add_argument('-p', '--parallel', action='store_true', help='use SCOOP to integrate the individuals in a parallel fashion')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    args = parser.parse_args(args=sys.argv[1:])

    folder = args.folder
    if not os.path.isdir(folder):
        print('%s: %s: no such directory.' % (os.path.basename(sys.argv[0]),folder))
        sys.exit(1)

    err_max = args.threshold

    features_to_consider = []
    if args.features != 'all':
        features_to_consider = args.features.split(',')

    features_to_ignore = []
    if args.ignore is not None:
        features_to_ignore = args.ignore.split(',')

    for feature in features_to_ignore:
        if feature in features_to_consider:
            print('Feature "%s" is both to consider and to ignore...' % feature)
            sys.exit(1)

    do_all = args.all
    verbose = args.verbose

    map_func = map
    if args.parallel:
        if not do_all:
            print('Ignoring the --parallel option.')
        else:
            try:
                from scoop import futures
                map_func = futures.map
            except:
                print('SCOOP not found: will run serially.')

    final_pop = np.array(pickle.load(open(folder + '/final_population.pkl', 'rb'), encoding='latin1'))
    # remove duplicate individuals
    final_pop = np.unique(final_pop, axis=0)
    hof = np.array(pickle.load(open(folder + '/hall_of_fame.pkl', 'rb'), encoding='latin1'))
    hof_responses = pickle.load(open(folder + '/hall_of_fame_responses.pkl', 'rb'), encoding='latin1')
    n_individuals,n_parameters = final_pop.shape
    evaluator = pickle.load(open(folder + '/evaluator.pkl', 'rb'))
    protocols = json.load(open(folder + '/protocols.json'))
    default_parameters = json.load(open(folder + '/parameters.json','r'))
    features = json.load(open(folder + '/features.json','r'))

    good_individuals_hof = []
    for individual,(i,resp) in zip(hof,enumerate(hof_responses)):
        if verbose:
            print('----------------------------------')
            print('Individual %02d' % (i+1))
        good = True
        for step_name in resp:
            step = step_name.split('.')[0]
            if verbose:
                print('  %s' % step)
            stim_start = evaluator.fitness_protocols[step].stimuli[0].step_delay
            stim_end = stim_start + evaluator.fitness_protocols[step].stimuli[0].step_duration
            trace = {'T': resp[step+'.soma.v']['time'],
                     'V': resp[step+'.soma.v']['voltage'],
                     'stim_start': [stim_start], 'stim_end': [stim_end]}
            feature_names = [key for key in features[step]['soma']]
            feature_values = efel.getFeatureValues([trace],feature_names)
            for name in feature_names:
                m = features[step]['soma'][name][0]
                s = features[step]['soma'][name][1]
                err = np.abs(np.mean(feature_values[0][name]) - m) / s
                if err > err_max:
                    if not name in features_to_ignore:
                        good = False
                        if verbose:
                            print(colors.red('    %s = %g' % (name,err)))
                    elif verbose:
                        print(colors.yellow('    %s = %g' % (name,err)))
                elif verbose:
                    print(colors.green('    %s = %g' % (name,err)))
        if good:
            good_individuals_hof.append(i)
            print('Individual ' + colors.green('%03d'%(i+1)) + ' of the hall-of-fame ' + colors.green('matches') + ' the requisites.')
        else:
            print('Individual ' + colors.red('%03d'%(i+1)) + ' of the hall-of-fame ' + colors.red('does not match') + ' the requisites.')

    data = {'good_individuals_hof': good_individuals_hof,
            'good_population': hof[good_individuals_hof,:],
            'err_max': err_max,
            'features': features_to_consider,
            'ignored_features': features_to_ignore}

    if do_all:
        good_individuals_also_in_hof = []
        for i,individual in enumerate(final_pop):
            idx = (individual == hof).all(axis=1).nonzero()[0]
            if len(idx) > 0:
                if idx[0] in good_individuals_hof:
                    good_individuals_also_in_hof.append(i)
                    print('Individual ' + colors.green('%03d'%(i+1)) + ' is in the hall-of-fame (#{}) and '.format(idx[0]+1) + \
                          colors.green('matches') + ' the requisites.')
                else:
                    print('Individual ' + colors.red('%03d'%(i+1)) + ' is in the hall-of-fame (#{}) and '.format(idx[0]+1) + \
                          colors.red('does not match') + ' the requisites.')

        swc_file = glob.glob(folder + '/*.converted.swc')[0]
        mech_file = folder + '/mechanisms.json'

        k = list(protocols.keys())[0]
        feature_names = [key for key in features[k]['soma'] if not key in features_to_ignore and \
                         (len(features_to_consider) == 0 or key in features_to_consider)]
        feature_reference_values = {feature: [] for feature in feature_names}
        for feature in feature_names:
            for step in protocols:
                feature_reference_values[feature].append(features[step]['soma'][feature])
        stim_dur = protocols[k]['stimuli'][0]['duration']
        stim_start = protocols[k]['stimuli'][0]['delay']
        I = [proto['stimuli'][0]['amp']*1e3 for _,proto in protocols.items()]
        idx = argsort(I)
        I = [I[i] for i in idx]
        feature_reference_values = {feature: [feature_reference_values[feature][i] for i in idx] for feature in feature_names}

        args = {'swc_file': swc_file,
                'mech_file': mech_file,
                'final_pop': final_pop,
                'default_parameters': default_parameters,
                'evaluator': evaluator,
                'I': I,
                'stim_dur': stim_dur,
                'stim_start': stim_start,
                'feature_names': feature_names,
                'feature_reference_values': feature_reference_values,
                'err_max': err_max,
                'verbose': verbose}

        individuals = [i for i in range(n_individuals) if i not in good_individuals_also_in_hof]
        good = list(map_func(lambda i: worker(i,args), individuals))
        good_individuals = [ind for ind,gd in zip(individuals,good) if gd]
        data['good_individuals'] = good_individuals
        data['good_population'] = np.concatenate((data['good_population'], final_pop[good_individuals,:]))

    pickle.dump(data, open(folder + '/good_population_%d_STDs.pkl' % err_max, 'wb'))
