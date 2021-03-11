
import os
import sys
import glob
import json
import pickle
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

__all__ = ['collect_folders', 'load_population_data', 'load_backprop_data', \
          'load_fI_data', 'plot_parameters_map', 'plot_parameters_maps']

def collect_folders(folder_patterns, cells_to_exclude, fun, base_folder='.', verbose=False):
    folders = {}
    for root, dirs, _ in os.walk(base_folder):
        for d in dirs:
            pkl = glob.glob(root + os.path.sep + d + os.path.sep + 'good_population*pkl')
            if len(pkl) > 0 and \
               not os.path.isfile(root + os.path.sep + d + os.path.sep + 'DO_NOT_USE'):
                timestamp,cell = fun(d)
                if cell in cells_to_exclude:
                    continue
                if not cell in folders:
                    folders[cell] = []
                for pattern in folder_patterns:
                    if pattern in timestamp:
                        folder = root + '/' + d
                        if not os.path.isfile(folder + '/DO_NOT_USE'):
                            folders[cell].append(folder)
                            if verbose:
                                print('Adding data from folder', folders[cell][-1])
                        elif verbose:
                            print('Ignoring data in %s because of DO_NOT_USE file.' % folder)
    return folders
    

def load_population_data(folder_list, n_stds=5, flatten=True):
    populations = {}
    groups = {}
    param_names = {}
    param_bounds = {}
    for key in folder_list:
        for i,folder in enumerate(folder_list[key]):
            infile = folder + '/good_population_{}_STDs.pkl'.format(n_stds)
            print('Loading data from file {}.'.format(infile))
            data = pickle.load(open(infile, 'rb'), encoding='latin1')['good_population']
            if not key in populations:
                populations[key] = data
                groups[key] = np.zeros(data.shape[0])
                evaluator = pickle.load(open(folder + '/evaluator.pkl', 'rb'))
                param_names[key] = evaluator.param_names
                param_bounds[key] = np.array([par.bounds for par in evaluator.params])
            else:
                populations[key] = np.concatenate((populations[key],data),axis=0)
                groups[key] = np.concatenate((groups[key], i+np.zeros(data.shape[0])))
    if flatten:
        populations = np.concatenate(list(populations.values())).T
        groups = np.concatenate(list(groups.values()))
    return populations, groups, param_names, param_bounds


def load_backprop_data(folder_list):
    apical_AP_amplitudes = {}
    apical_distances = {}
    for key in folder_list:
        for i,folder in enumerate(folder_list[key]):
            data = pickle.load(open(folder + '/AP_backpropagation.pkl', 'rb'), encoding='latin1')
            apic_amp = np.array([x['AP_amplitudes']['apical']/x['AP_amplitudes']['somatic'] for x in data if x is not None])
            if len(apic_amp) == 0:
                continue
            apical_distances[key] = data[0]['distances']['apical']
            if not key in apical_AP_amplitudes:
                apical_AP_amplitudes[key] = apic_amp
            else:
                apical_AP_amplitudes[key] = np.r_[apical_AP_amplitudes[key], apic_amp]

    return apical_distances,apical_AP_amplitudes


def load_fI_data(folder_list, n_stds, verbose=False):
    I = {}
    num_spikes = {}
    inverse_first_isi = {}
    mean_frequency = {}
    for key in folder_list:
        for i,folder in enumerate(folder_list[key]):
            try:
                infile = folder + '/fI_curve_good_population_{}_STDs.pkl'.format(n_stds)
                if verbose:
                    print('Loading data from file {}.'.format(infile))
                data = pickle.load(open(infile, 'rb'), encoding='latin1')
            except:
                continue
            I[key] = data['I']
            mf = np.array([[1000 * spike_times.shape[0] / (spike_times[-1] - data['delay']) \
                            if spike_times.shape[0] > 0 else 0 for spike_times in indiv] \
                           for indiv in data['spike_times']])
            if not key in num_spikes:
                num_spikes[key] = np.reshape([len(spks) for indiv in data['spike_times'] for spks in indiv], \
                                             (len(data['spike_times']),len(data['spike_times'][0])))
                inverse_first_isi[key] = data['inverse_first_isi']
                mean_frequency[key] = mf
            else:
                num_spikes[key] = np.r_[num_spikes[key], \
                                        np.reshape([len(spks) for indiv in data['spike_times'] for spks in indiv], \
                                                   (len(data['spike_times']),len(data['spike_times'][0])))]
                inverse_first_isi[key] = np.r_[inverse_first_isi[key], data['inverse_first_isi']]
                mean_frequency[key] = np.r_[mean_frequency[key], mf]
    return I,num_spikes,mean_frequency,inverse_first_isi


def plot_parameters_map(population, evaluator, config, ax=None, groups=None, sort_parameters=True, parameter_names_on_ticks=True, sort_idx=None):
    def convert_param_names(param_names):
        np = len(param_names)
        suffix = {'somatic': 's', 'axonal': 'a', 'allnoaxon': 's,d', 'apical': 'ap', 'basal': 'b', 'alldend': 'd'}
        for i in range(np):
            name,loc = param_names[i].split('.')
            param_names[i] = name + '.' + suffix[loc]

    n_parameters,n_individuals = population.shape

    bounds = {}
    for region,params in config['optimized'].items():
        for param in params:
            param_name = param[0] + '.' + region
            bounds[param_name] = np.array([param[1],param[2]])

    normalized = np.zeros(population.shape)
    pop_maxima = np.max(population,axis=1)
    pop_minima = np.min(population,axis=1)
    for i,name in enumerate(evaluator.param_names):
        normalized[i,:] = (population[i,:] - bounds[name][0]) / (bounds[name][1] - bounds[name][0])

    if ax is None:
        ax = plt.gca()

    if sort_parameters:
        m = np.mean(normalized, axis=1)
        if sort_idx is None:
            idx = np.argsort(m)[::-1]
        else:
            idx = sort_idx
        normalized_sorted_by_mean = normalized[idx,:].copy()
        s = np.std(normalized_sorted_by_mean, axis=1)
        param_names_sorted_by_mean = [evaluator.param_names[i] for i in idx]
        for i,name in enumerate(param_names_sorted_by_mean):
            if 'bar' in name:
                param_names_sorted_by_mean[i] = name.split('bar_')[1]
        img = ax.imshow(normalized_sorted_by_mean, cmap='jet')
    else:
        s = np.std(normalized, axis=1)
        img = ax.imshow(normalized, cmap='jet')

    if groups is not None:
        borders, = np.where(groups[:-1] != groups[1:])
        borders_idx = np.where(groups[borders] == 0)[0][1:] - 1
        for i in borders:
            if i in borders[borders_idx]:
                ls = '-'
                lw = 2
            else:
                ls = '--'
                lw = 1
            ax.plot([i,i], [0,n_parameters-1], 'w'+ls, linewidth=lw)
        ax.set_xticks(np.append(borders,n_individuals-1))
        ax.set_xticklabels(np.append(borders+1,n_individuals))
    elif n_individuals < 20:
        ax.set_xticks(np.arange(n_individuals))
        ax.set_xticklabels(1+np.arange(n_individuals))

    ax.set_xlabel('Individual #')

    ax.set_yticks(np.arange(n_parameters))
    if parameter_names_on_ticks:
        if sort_parameters:
            convert_param_names(param_names_sorted_by_mean)
            ax.set_yticklabels(param_names_sorted_by_mean, fontsize=6)
        else:
            param_names = [name if not 'bar' in name else name.split('bar_')[1] \
                           for name in evaluator.param_names]
            convert_param_names(param_names)
            ax.set_yticklabels(param_names, fontsize=6)
        for i in range(n_parameters):
            if s[i] < 0.2:
                ax.get_yticklabels()[i].set_color('red')
    else:
        ax.set_yticklabels(1+np.arange(n_parameters), fontsize=6)

    if sort_parameters:
        return idx
    return None


def plot_parameters_maps(folders, n_stds, titles=None, sort_parameters=True, parameter_names_on_ticks=True):
    if titles is None:
        titles = {k: k for k in folders}
    n_rows = len(folders)
    n_cols = np.max([len(v) for v in folders.values()])
    fig,axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3,n_rows*2.5), squeeze=False)
    n_ind = np.zeros((n_rows,n_cols))
    for i,condition in enumerate(folders):
        population,groups = load_population_data(folders[condition], n_stds, flatten=False)
        sort_idx = None
        for j,cell in enumerate(population):
            folder = folders[condition][cell][0]
            evaluator = pickle.load(open(folder + '/evaluator.pkl','rb'))
            _,config = json.load(open(folder + '/parameters.json','r')).popitem()
            sort_idx = plot_parameters_map(population[cell].T, evaluator, config, axes[i,j], \
                                           groups[cell], sort_parameters, parameter_names_on_ticks, sort_idx)
            axes[i,j].set_title(cell)
            n_ind[i,j] = population[cell].shape[0]

    dx = 0.1
    if n_cols == 1:
        x_offset = [0.3,0.05]
        dy = 0.2
    else:
        x_offset = [0.1,0.05]
        dy = 0.15
    y_offset = [0.1,0.1]
    x_space = 1 - (n_cols-1) * dx - np.sum(x_offset)
    y_space = 1 - (n_rows-1) * dy - np.sum(y_offset)
    y_pos = y_offset[0]
    height = y_space / n_rows
    for i in range(n_rows):
        x_pos = x_offset[0]
        for j in range(n_cols):
            width = n_ind[i,j] / np.sum(n_ind[i,:]) * x_space
            pos = [x_pos, y_pos, width, height]
            axes[i,j].set_position(pos)
            x_pos += width + dx
        y_pos += height + dy

