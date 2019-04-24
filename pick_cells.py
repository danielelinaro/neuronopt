#!/usr/bin/env python

import os
import sys
import copy
import json
import pickle
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt

import efel
from neuron import h
from scoop import futures
from current_step import inject_current_step


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


def dump_parameters(parameters,default_parameters,evaluator,filename):
    param_dict = evaluator.param_dict(parameters)
    default_parameters_copy = copy.deepcopy(default_parameters)
    for par in default_parameters_copy:
        if 'value' not in par:
            par['value'] = param_dict[par['param_name'] + '.' + par['sectionlist']]
            par.pop('bounds')
    json.dump(default_parameters_copy,open(filename,'w'),indent=4)


def main():
    parser = arg.ArgumentParser(description='Select cells that match a certain quality threshold.')
    parser.add_argument('folder', type=str, action='store', help='folder containing the results of the optimization')
    parser.add_argument('-t', '--threshold', default=3, type=int, help='threshold on the number of STDs to accept a solution')
    parser.add_argument('--features', default='all', type=str, help='comma-separated list of features to consider (default: all)')
    parser.add_argument('--ignore', default=None, type=str, help='comma-separated list of features to ignore')
    parser.add_argument('-a', '--all', action='store_true', help='process all solutions (potentially very time consuming)')
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
    
    colors = ColorFactory()

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


    if do_all:
        swc_file = folder + '/110203_b_x40_4.converted.swc'
        mech_file = folder + '/mechanisms.json'

        do_plot = False

        good_individuals = []

        for i,individual in enumerate(final_pop):
            idx = (individual == hof).all(axis=1).nonzero()[0]
            if len(idx) > 0:
                if idx[0] in good_individuals_hof:
                    good_individuals.append(idx)
                    print('Individual ' + colors.green('%03d'%(i+1)) + ' is in the hall-of-fame (#{}) and '.format(idx[0]+1) + \
                          colors.green('matches') + ' the requisites.')
                else:
                    print('Individual ' + colors.red('%03d'%(i+1)) + ' is in the hall-of-fame (#{}) and '.format(idx[0]+1) + \
                          colors.red('does not match') + ' the requisites.')
                continue
            params_file = '/tmp/individual_%03d.json' % i
            dump_parameters(individual, default_parameters, evaluator, params_file)
            if verbose:
                print('----------------------------------------')
                print('Individual %02d' % (i+1))
            for j,step in enumerate(protocols):
                cell_name = 'individual_%03d_%d' % (i,j)
                I = protocols[step]['stimuli'][0]['amp'] * 1e3
                dur = protocols[step]['stimuli'][0]['duration']
                stim_start = protocols[step]['stimuli'][0]['delay']
                stim_end = stim_start+dur
                if verbose:
                    print('  %s' % step)
                recorders = inject_current_step(I, swc_file, mech_file, params_file, stim_start, dur, cell_name, do_plot)
                h('forall delete_section()')
                trace = {'T': recorders['t'], 'V': recorders['Vsoma'], 'stim_start': [stim_start], 'stim_end': [stim_end]}
                feature_names = [key for key in features[step]['soma']]
                feature_values = efel.getFeatureValues([trace],feature_names)
                good = True
                for name in feature_names:
                    m = features[step]['soma'][name][0]
                    s = features[step]['soma'][name][1]
                    try:
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
                    except:
                        good = False
                        if verbose:
                            print(colors.red('    %s = None' % name))
                if not good:
                    break
            if good:
                good_individuals.append(i)
                print('Individual ' + colors.green('%03d'%(i+1)) + colors.green(' matches') + ' the requisites')
            else:
                print('Individual ' + colors.red('%03d'%(i+1)) + colors.red(' does not match') + ' the requisites')

        print('Good invidividuals: ', good_individuals)
    
if __name__ == '__main__':
    main()
