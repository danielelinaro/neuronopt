import os
import sys
import argparse as arg
import json
import efel
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import bluepyopt.ephys as ephys
from neuron import h
from utils import *

files = {'results': {'hof_resp': 'hall_of_fame_responses.pkl',
                     'hof': 'hall_of_fame.pkl',
                     'final_pop': 'final_population.pkl',
                     'history': 'history.pkl',
                     'log': 'logbook.pkl',
                     'eval': 'evaluator.pkl'},
         'config': {'params': 'parameters.json',
                    'features': 'features.json',
                    'mechanisms': 'mechanisms.json'}}

def load_files():
    parameters = json.load(open(files['config']['params'],'r'))
    features = json.load(open(files['config']['features'],'r'))
    try:
        mechanisms = json.load(open(files['config']['mechanisms'],'r'))
    except:
        mechanisms = None
    
    hall_of_fame = np.array(pickle.load(open(files['results']['hof'],'rb'),encoding='latin1'))
    final_pop = np.array(pickle.load(open(files['results']['final_pop'],'rb'),encoding='latin1'))
    evaluator = pickle.load(open(files['results']['eval'],'rb'),encoding='latin1')
    responses = pickle.load(open(files['results']['hof_resp'],'rb'),encoding='latin1')

    return parameters,features,mechanisms,hall_of_fame,final_pop,evaluator,responses


def plot_summary(target_features,hall_of_fame,final_pop,evaluator,responses,individual=0,dump=False):
    pop_size = len(final_pop)
    n_params = len(evaluator.param_names)

    params = final_pop.transpose().copy()
    bounds = np.zeros((n_params,2))
    for i in range(n_params):
        bounds[i,:] = evaluator.cell_model.params[evaluator.param_names[i]].bounds
    idx = np.argsort(bounds[:,1])
    params_median = np.median(params,1)
    best_params = np.array(hall_of_fame[individual])[idx]
    params = params[idx,:]
    params_median = params_median[idx]
    bounds = bounds[idx,:]
    param_names = [evaluator.param_names[i] for i in idx]

    features = {}
    features_std_units = {}
    for proto,feature_names in target_features.items():
        stim_start = evaluator.fitness_protocols[proto].stimuli[0].step_delay
        stim_end = stim_start + evaluator.fitness_protocols[proto].stimuli[0].step_duration
        trace = {'T': responses[individual][proto+'.soma.v']['time'],
                 'V': responses[individual][proto+'.soma.v']['voltage'],
                 'stim_start': [stim_start], 'stim_end': [stim_end]}
        features[proto] = {k:[np.mean(v),np.std(v)] for k,v in efel.getFeatureValues([trace],feature_names['soma'])[0].items()}
        features_std_units[proto] = {k:np.abs(target_features[proto]['soma'][k][0]-np.mean(v))/target_features[proto]['soma'][k][1] \
                                     for k,v in efel.getFeatureValues([trace],feature_names['soma'])[0].items()}
        print('%s:' % proto)
        feature_values = efel.getFeatureValues([trace],feature_names['soma'])[0]
        for name,values in feature_names['soma'].items():
            print('\t%s: model: %g (%g std from data mean). data: %g +- %g (std/mean: %g)' %
                  (name,feature_values[name][0],np.abs(values[0]-feature_values[name][0])/values[1],
                   values[0],values[1],values[1]/np.abs(values[0])))

    if dump:
        fid = open('%s_individual_%d_errors.csv'%(os.path.basename(os.path.abspath('.')),individual),'w')
        for k in features_std_units['Step3']:
            fid.write('%s,' % k)
            for i in range(1,3):
                try:
                    fid.write('%f,' % features_std_units['Step%d'%i][k])
                except:
                    fid.write(',')
            fid.write('%f\n' % features_std_units['Step3'][k])
        fid.close()

    nsteps = len(responses[individual].keys())

    plt.figure()
    plt.axes([0.01,0.65,0.98,0.35])
    offset = 0
    before = 250
    after = 200
    dx = 100
    dy = 50
    if dump:
        fid = open('%s_individual_%d_traces.csv'%(os.path.basename(os.path.abspath('.')),individual),'w')
    for resp in responses[individual].values():
        # this is because of the variable time-step integration
        start = np.max((0,np.where(resp['time'] > stim_start-before)[0][0] - 1))
        stop = np.where(resp['time'] < stim_end+after)[0][-1] + 2
        idx = np.arange(start,stop)
        plt.plot(resp['time'][idx],resp['voltage'][idx]+offset,'k',lw=1)
        old_offset = offset
        offset += np.diff([np.min(resp['voltage']),np.max(resp['voltage'])])[0] + 5
        if dump:
            for v in resp['time'][idx]:
                fid.write('%f,' % (v-resp['time'][idx[0]]))
            fid.write('\n')
            for v in resp['voltage'][idx]:
                fid.write('%f,' % v)
            fid.write('\n')
    if dump:
        fid.close()
    
    plt.plot(stim_start-before+100+np.zeros(2),old_offset-dy/2+np.array([0,dy]),'k',linewidth=1)
    plt.plot(stim_start-before+100+np.array([0,dx]),old_offset-dy/2+np.zeros(2),'k',linewidth=1)
    plt.text(stim_start-before+100+dx/2,old_offset-dy/2*1.3,'%d ms'%dx,fontsize=8,
             verticalalignment='top',horizontalalignment='center')
    plt.text(stim_start-before+70,old_offset,'%d mV'%dy,rotation=90,fontsize=8,
             verticalalignment='center',horizontalalignment='center')
    plt.axis('tight')
    plt.xlim([stim_start-before,stim_end+after])
    plt.axis('off')

    fnt = 9
    offset = 0.175
    space = 0.03
    dx = (0.97 - offset - (nsteps-1)*space)/nsteps
    all_feature_names = sorted(features_std_units['Step3'].keys())
    n_features = len(all_feature_names)
    Y = list(range(n_features,0,-1))
    dy = 0.3
    green = [0,.7,.3]
    for i,(stepnum,feat) in enumerate(features_std_units.items()):
        ax = plt.axes([offset+i*(dx+space),0.4,dx,0.2])
        X = np.zeros(n_features)
        for k,v in feat.items():
            idx, = np.where([k == f for f in all_feature_names])
            X[idx] = v
        for x,y in zip(X,Y):
            ax.add_patch(Rectangle((0,y-dy),x,2*dy,\
                                   edgecolor=green,facecolor=green,linewidth=1))
        plt.plot([3,3],[np.min(Y)-1,np.max(Y)+1],'r--',lw=1)
        plt.title(stepnum,fontsize=fnt)
        plt.xlabel('Objective value (# std)',fontsize=fnt)
        
        if np.max(X) > 31:
            dtick = 10
        elif np.max(X) > 14:
            dtick = 5
        else:
            dtick = 2
        plt.xticks(np.arange(0,np.ceil(np.max(X))+1,dtick),fontsize=fnt)
    
        if i == 0:
            plt.yticks(Y,all_feature_names,fontsize=6)
        else:
            plt.yticks(Y,[])
        
        plt.axis([0,np.max([3.1,np.ceil(np.max(X))]),np.min(Y)-1,np.max(Y)+1])

    blue = [.9,.9,1]
    ax = plt.axes([0.25,0.05,0.72,0.25])
    dy = 0.3
    plot = plt.semilogx
    for y0,(par,best_par,m,b) in enumerate(zip(params,best_params,params_median,bounds)):
        y = y0 - dy/2 +  dy*np.random.rand(pop_size)
        plot(par,y,'o',color=[.6,.6,.6],linewidth=0.5,markersize=3,markerfacecolor='w')
        plot(m+np.zeros(2),y0-dy+np.array([0,2*dy]),'r',linewidth=2)
        plot(best_par,y0,'ks',linewidth=1,markersize=4,markerfacecolor=[1,1,.7])
        if b[0] < 1e-12:
            b[0] = 1e-12
        ax.add_patch(Rectangle((b[0],y0-dy),b[1]-b[0],2*dy,\
                               edgecolor=blue,facecolor=blue,linewidth=1))


    outfile = os.path.basename(os.path.abspath('.')) + '_individual_%d.pdf' % individual

    plt.xlim([1e-6,1500])
    plt.ylim([-1,n_params])
    plt.yticks(np.arange(n_params),param_names,fontsize=6)
    plt.savefig(outfile)
    #plt.show()


def main():
    parser = arg.ArgumentParser(description='Plot results of the optimization.', prog=os.path.basename(sys.argv[0]))
    parser.add_argument('individuals', type=int, action='store', nargs='*', default=[0], help='individuals to plot')
    parser.add_argument('-a', '--all', action='store_true', help='plot all individuals')
    parser.add_argument('-d', '--dump', action='store_true', help='dump traces and error values')
    args = parser.parse_args(args=sys.argv[1:])

    parameters,features,mechanisms,hall_of_fame,final_pop,evaluator,responses = load_files()

    if mechanisms is not None:
        write_parameters_v1(parameters, hall_of_fame, evaluator)
    else:
        cell_name = '_'.join(os.path.split(os.path.abspath('.'))[1].split('_')[1:])
        write_parameters_v2(parameters[cell_name], hall_of_fame, evaluator)

    if args.all:
        individuals = list(range(len(responses)))
    else:
        individuals = args.individuals

    for ind in individuals:
        plot_summary(features,hall_of_fame,final_pop,evaluator,responses,ind,args.dump)
    

if __name__ == '__main__':
    main()
    
