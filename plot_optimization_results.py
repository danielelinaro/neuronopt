import os
import sys
import argparse as arg
import json
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dlutils.utils import write_parameters


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


def plot_summary(target_features,hall_of_fame,final_pop,evaluator,responses,individual=0,dump=False,verbose=True):
    import efel
    efel.setThreshold(-20)
    pop_size = len(final_pop)
    n_params = len(evaluator.param_names)

    params = np.abs(final_pop.transpose().copy())
    bounds = np.zeros((n_params,2))
    for i in range(n_params):
        bounds[i,:] = evaluator.cell_model.params[evaluator.param_names[i]].bounds
    bounds = np.abs(bounds)
    idx = np.argsort(bounds[:,1])
    params_median = np.median(params,1)
    best_params = np.abs(np.array(hall_of_fame[individual])[idx])
    params = params[idx,:]
    params_median = params_median[idx]
    bounds = bounds[idx,:]
    param_names = [evaluator.param_names[i] for i in idx]

    features = {}
    features_std_units = {}
    all_feature_names = []
    proto_names = target_features.keys()
    site_names = []
    for proto,target_feature in target_features.items():
        site_names.extend(list(target_feature.keys()))
        features[proto] = {}
        features_std_units[proto] = {}
        for site in target_feature:
            feature_names = target_feature[site].keys()
            stim_start = evaluator.fitness_protocols[proto].stimuli[0].step_delay
            stim_end = stim_start + evaluator.fitness_protocols[proto].stimuli[0].step_duration
            trace = {'T': responses[individual]['{}.{}.v'.format(proto,site)]['time'],
                     'V': responses[individual]['{}.{}.v'.format(proto,site)]['voltage'],
                     'stim_start': [stim_start], 'stim_end': [stim_end]}
            feature_values = efel.getFeatureValues([trace],feature_names)[0]
            features[proto][site] = {k: [np.mean(v),np.std(v)] if v is not None else [None,None] \
                                     for k,v in feature_values.items()}
            features_std_units[proto][site] = {k: np.abs(target_feature[site][k][0]-np.mean(v)) / target_feature[site][k][1] \
                                               if v is not None else None for k,v in feature_values.items()}
            if verbose:
                print('{}.{}:'.format(proto,site))
            for name,values in target_feature[site].items():
                if not name in all_feature_names:
                    all_feature_names.append(name)
                if verbose:
                    if features[proto][site][name][0] is None:
                        print('\t%s: model: None. data: %g +- %g (std/mean: %g)' %
                              (name,values[0],values[1],values[1]/np.abs(values[0])))
                    else:
                        print('\t%s: model: %g (%g std from data mean). data: %g +- %g (std/mean: %g)' %
                              (name,features[proto][site][name][0],features_std_units[proto][site][name],
                               values[0],values[1],values[1]/np.abs(values[0])))

    site_names = list(set(site_names))
    site_names = sorted(site_names, key = lambda s: 'a' if s == 'soma' else s, reverse = True)
    n_proto = len(proto_names)
    n_sites = len(site_names)

    if dump:
        fid = open('{}_individual_{}_errors.csv'.format(os.path.basename(os.path.abspath('.')),individual), 'w')
        fid.write('feature name')
        for proto in proto_names:
            for site in site_names:
                fid.write(',{}.{}'.format(proto,site))
        fid.write('\n')
        for k in all_feature_names:
            fid.write(k)
            for proto in proto_names:
                for site in site_names:
                    fid.write(',{}'.format(features_std_units[proto][site][k]))
            fid.write('\n')
        fid.close()

    h = 5 + (n_sites - 1) * 1 # [in]
    top = 1.25 # [in]
    bottom = 1.75 # [in]
    plt.figure(figsize=(6,h))
    plt.axes([0.01,1-top/h,0.98,top/h])
    offset = 0
    before = 250
    after = 200
    dx = 100
    dy = 50
    colors = 'rgbcmyk'
    colors = colors[:n_sites-1]
    colors += 'k'
    cmap = {site: col for site,col in zip(site_names,colors)}
    if dump:
        fid = open('{}_individual_{}_traces.csv'.format(os.path.basename(os.path.abspath('.')),individual), 'w')
    for proto in proto_names:
        l_min = 1e6
        l_max = -1e6
        for site in site_names:
            key = '{}.{}.v'.format(proto,site)
            if not key in responses[individual]:
                continue
            resp = responses[individual][key]
            # this is because of the variable time-step integration
            start = np.max((0,np.where(resp['time'] > stim_start-before)[0][0] - 1))
            stop = np.where(resp['time'] < stim_end+after)[0][-1] + 2
            idx = np.arange(start,stop)
            plt.plot(resp['time'][idx],resp['voltage'][idx]+offset,cmap[site],lw=1)
            if dump:
                fid.write('{}.{}.time'.format(proto,site))
                for v in resp['time'][idx]:
                    fid.write(',{:.4f}'.format(v-resp['time'][idx[0]]))
                fid.write('\n')
                fid.write('{}.{}.voltage'.format(proto,site))
                for v in resp['voltage'][idx]:
                    fid.write(',{:.4f}'.format(v))
                fid.write('\n')
            if np.min(resp['voltage']) < l_min:
                l_min = np.min(resp['voltage'])
            if np.max(resp['voltage']) > l_max:
                l_max = np.max(resp['voltage'])
        old_offset = offset
        offset += np.max(np.diff([l_min,l_max])[0] + 5)

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

    fnt = 8
    offset = {'x': 0.175, 'y': 0.125 + bottom/h}
    space = {'x': 0.03, 'y': 0.06}
    dx = (0.97 - offset['x'] - (n_proto-1)*space['x'])/n_proto
    dy = (1 - top/h - 0.035 - offset['y'] - (n_sites-1)*space['y'])/n_sites
    all_feature_names = sorted(all_feature_names)
    n_features = len(all_feature_names)
    Y = list(range(n_features,0,-1))
    green = [0,.7,.3]
    for i,proto in enumerate(proto_names):
        for j,site in enumerate(site_names):
            if not site in features_std_units[proto]:
                continue
            ax = plt.axes([offset['x']+i*(dx+space['x']),offset['y']+j*(dy+space['y']),dx,dy])
            feat = features_std_units[proto][site]
            X = np.zeros(n_features)
            for k,v in feat.items():
                idx, = np.where([k == f for f in all_feature_names])
                X[idx] = v
            for x,y in zip(X,Y):
                ax.add_patch(Rectangle((0,y-0.3),x,2*0.3,\
                                       edgecolor=green,facecolor=green,linewidth=1))
            ax.plot([5,5],[np.min(Y)-1,np.max(Y)+1],'r--',lw=1)
            ax.set_title('{}.{}'.format(proto,site),fontsize=fnt)
            if np.max(X) > 31:
                dtick = 10
            elif np.max(X) > 14:
                dtick = 5
            elif np.max(X) > 5:
                dtick = 2
            else:
                dtick = 1

            plt.xticks(np.arange(0,np.max([6,np.ceil(np.nanmax(X))+1]),dtick),fontsize=fnt)

            if j == 0:
                plt.xlabel('Objective value (# std)',fontsize=fnt)
            if i == 0:
                plt.yticks(Y,all_feature_names,fontsize=fnt-3)
            else:
                plt.yticks(Y,[])
        
            ax.set_xlim([0,np.max([5.1,np.ceil(np.nanmax(X))])])
            ax.set_ylim([np.min(Y)-1,np.max(Y)+1])

    blue = [.9,.9,1]
    ax = plt.axes([0.25,0.05,0.72,bottom/h])
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

    outfile = os.path.basename(os.path.abspath('.')) + '_individual_{}.pdf'.format(individual)

    plt.xlim([1e-6,1500])
    plt.ylim([-1,n_params])
    plt.yticks(np.arange(n_params),param_names,fontsize=fnt-3)
    plt.savefig(outfile)


def main():
    parser = arg.ArgumentParser(description='Plot results of the optimization.', prog=os.path.basename(sys.argv[0]))
    parser.add_argument('individuals', type=int, action='store', nargs='*', default=[0], help='individuals to plot')
    parser.add_argument('-a', '--all', action='store_true', help='plot all individuals')
    parser.add_argument('-d', '--dump', action='store_true', help='dump traces and error values')
    parser.add_argument('-q', '--quiet', action='store_true', help='be quiet')
    args = parser.parse_args(args=sys.argv[1:])

    parameters,features,mechanisms,hall_of_fame,final_pop,evaluator,responses = load_files()

    config = None
    default_parameters = None
    if mechanisms is not None:
        default_parameters = parameters
    else:
        cell_name = '_'.join(os.path.split(os.path.abspath('.'))[1].split('_')[1:])
        config = parameters[cell_name]
    
    write_parameters(hall_of_fame, evaluator, config, default_parameters)
    
    if args.all:
        individuals = list(range(len(responses)))
    else:
        individuals = args.individuals

    for ind in individuals:
        plot_summary(features,hall_of_fame,final_pop,evaluator,responses,ind,args.dump,not args.quiet)
    

if __name__ == '__main__':
    main()
    
