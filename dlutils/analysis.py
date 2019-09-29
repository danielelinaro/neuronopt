
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
    

def load_population_data(folder_list, flatten=True):
    populations = {}
    groups = {}
    for key in folder_list:
        for i,folder in enumerate(folder_list[key]):
            data = pickle.load(open(folder + '/good_population_5_STDs.pkl', 'rb'), \
                              encoding='latin1')['good_population']
            if not key in populations:
                populations[key] = data
                groups[key] = np.zeros(data.shape[0])
            else:
                populations[key] = np.concatenate((populations[key],data),axis=0)
                groups[key] = np.concatenate((groups[key], i+np.zeros(data.shape[0])))
    if flatten:
        populations = np.concatenate(list(populations.values())).T
        groups = np.concatenate(list(groups.values()))
    return populations,groups


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


def load_fI_data(folder_list):
    I = {}
    num_spikes = {}
    inverse_first_isi = {}
    mean_frequency = {}
    for key in folder_list:
        for i,folder in enumerate(folder_list[key]):
            try:
                data = pickle.load(open(folder + '/fI_curve_good_population_5_STDs.pkl', 'rb'), encoding='latin1')
            except:
                continue
            I[key] = data['I']
            mf = np.array([[1000 * spike_times.shape[0] / (spike_times[-1] - data['delay']) \
                            if spike_times.shape[0] > 0 else 0 for spike_times in indiv] \
                           for indiv in data['spike_times']])
            if not key in num_spikes:
                num_spikes[key] = data['no_spikes']
                inverse_first_isi[key] = data['inverse_first_isi']
                mean_frequency[key] = mf
            else:
                num_spikes[key] = np.r_[num_spikes[key], data['no_spikes']]
                inverse_first_isi[key] = np.r_[inverse_first_isi[key], data['inverse_first_isi']]
                mean_frequency[key] = np.r_[mean_frequency[key], mf]

    return I,num_spikes,mean_frequency,inverse_first_isi


def plot_parameters_map(population, evaluator, config, ax=None, groups=None, sort_parameters=True, parameter_names_on_ticks=True):
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
        idx = np.argsort(m)[::-1]
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
        borders_idx = np.where(groups[borders] == 0)[0][1:]
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
            ax.set_yticklabels(param_names_sorted_by_mean, fontsize=7)
        else:
            ax.set_yticklabelss(evaluator.param_names)
        for i in range(n_parameters):
            if s[i] < 0.2:
                ax.get_yticklabels()[i].set_color('red')
    else:
        ax.set_yticklabels(1+np.arange(n_parameters))


def plot_parameters_maps(folders, titles=None):
    if titles is None:
        titles = {k: k for k in folders}
    fig,axes = plt.subplots(2,1,figsize=(10,5))
    n_ind = [0, 0]
    for i,key in enumerate(folders):
        folder = list(folders[key].values())[0][0]
        population,groups = load_population_data(folders[key], flatten=True)
        n_ind[i] = population.shape[1]
        evaluator = pickle.load(open(folder + '/evaluator.pkl','rb'))
        _,config = json.load(open(folder + '/parameters.json','r')).popitem()
        plot_parameters_map(population, evaluator, config, axes[i], groups)
        axes[i].set_title(titles[key])

    bounds = [list(ax.get_position().bounds) for ax in axes]
    m = np.argmax(n_ind)
    y_width = bounds[m][3]
    for i in range(len(axes)):
        bounds[i][0] = 0.1
        bounds[i][1] = 0.125 + y_width * 1.3 * i
        bounds[i][2] = 0.85 * n_ind[i] / n_ind[m]
        bounds[i][3] = y_width * 0.8
        axes[i].set_position(bounds[i])

    plt.savefig('parameters_maps.pdf')
    plt.show()

