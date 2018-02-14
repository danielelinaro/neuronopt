#!/usr/bin/env python

import os
import json
import efel
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import bluepyopt.ephys as ephys
from neuron import h

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
    mechanisms = json.load(open(files['config']['mechanisms'],'r'))
    
    hall_of_fame = pickle.load(open(files['results']['hof'],'r'))
    final_pop = pickle.load(open(files['results']['final_pop'],'r'))
    evaluator = pickle.load(open(files['results']['eval'],'r'))
    responses = pickle.load(open(files['results']['hof_resp'],'r'))

    return parameters,features,mechanisms,hall_of_fame,final_pop,evaluator,responses


def write_optimal_parameters(parameters,hall_of_fame,evaluator):
    optimal_individual = evaluator.param_dict(hall_of_fame[0])
    for par in parameters:
        if 'value' not in par:
            par['value'] = optimal_individual[par['param_name'] + '.' + par['sectionlist']]
            par.pop('bounds')
    json.dump(parameters,open('optimal_parameters.json','w'),indent=4)


def plot_summary(target_features,hall_of_fame,final_pop,evaluator,responses):
    pop_size = len(final_pop)
    n_params = len(evaluator.param_names)

    params = np.array([map(lambda x: x[i], final_pop) \
                       for i in range(n_params)])
    bounds = np.zeros((n_params,2))
    for i in range(n_params):
        bounds[i,:] = evaluator.cell_model.params[evaluator.param_names[i]].bounds
    idx = np.argsort(bounds[:,1])
    params_median = np.median(params,1)
    best_params = np.array(hall_of_fame[0])[idx]
    params = params[idx,:]
    params_median = params_median[idx]
    bounds = bounds[idx,:]
    param_names = [evaluator.param_names[i] for i in idx]

    features = {}
    features_std_units = {}
    for proto,feature_names in target_features.iteritems():
        stim_start = evaluator.fitness_protocols[proto].stimuli[0].step_delay
        stim_end = stim_start + evaluator.fitness_protocols[proto].stimuli[0].step_duration
        trace = {'T': responses[0][proto+'.soma.v']['time'],
                 'V': responses[0][proto+'.soma.v']['voltage'],
                 'stim_start': [stim_start], 'stim_end': [stim_end]}
        features[proto] = {k:[np.mean(v),np.std(v)] for k,v in efel.getFeatureValues([trace],feature_names['soma'])[0].iteritems()}
        features_std_units[proto] = {k:np.abs(target_features[proto]['soma'][k][0]-np.mean(v))/target_features[proto]['soma'][k][1] \
                                     for k,v in efel.getFeatureValues([trace],feature_names['soma'])[0].iteritems()}

    nsteps = len(responses[0].keys())

    plt.figure()
    plt.axes([0.01,0.65,0.98,0.35])
    offset = 0
    n = 1
    before = 250
    after = 200
    dx = 100
    dy = 50
    for k in responses[0]:
        for i in range(n):
            # this is because of the variable time-step integration
            start = np.where(responses[i][k]['time'] > stim_start-before)[0][0] - 1
            stop = np.where(responses[i][k]['time'] < stim_end+after)[0][-1] + 2
            idx = np.arange(start,stop)
            plt.plot(responses[i][k]['time'][idx],responses[i][k]['voltage'][idx]+offset,
                     color=[0 + 0.6/n*i for j in range(3)],linewidth=1)
        old_offset = offset
        offset += np.diff([np.min(responses[0][k]['voltage']),np.max(responses[0][k]['voltage'])])[0] + 5

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
    all_feature_names = features_std_units['Step3'].keys()
    idx = [i[0] for i in sorted(enumerate(all_feature_names), key=lambda x:x[1])]
    all_feature_names = [all_feature_names[i] for i in idx]
    n_features = len(all_feature_names)
    Y = range(n_features,0,-1)
    dy = 0.3
    green = [0,.7,.3]
    for i,(stepnum,feat) in enumerate(features_std_units.iteritems()):
        ax = plt.axes([offset+i*(dx+space),0.4,dx,0.2])
        X = np.zeros(n_features)
        for k,v in feat.iteritems():
            idx, = np.where([k == f for f in all_feature_names])
            X[idx] = v
        for x,y in zip(X,Y):
            ax.add_patch(Rectangle((0,y-dy),x,2*dy,\
                                   edgecolor=green,facecolor=green,linewidth=1))

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
        
        plt.axis([0,np.ceil(np.max(X)),np.min(Y)-1,np.max(Y)+1])

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


    outfile = os.path.basename(os.path.abspath('.')) + '.pdf'

    plt.xlim([1e-6,1500])
    plt.ylim([-1,n_params])
    plt.yticks(np.arange(n_params),param_names,fontsize=6)
    plt.savefig(outfile)
    plt.show()


def define_mechanisms():
    """Define mechanisms"""

    mech_definitions = json.load(open('mechanisms.json','r'))

    mechanisms = []
    for sectionlist, channels in mech_definitions.items():
        seclist_loc = ephys.locations.NrnSeclistLocation(
            sectionlist,
            seclist_name=sectionlist)
        for channel in channels:
            mechanisms.append(ephys.mechanisms.NrnMODMechanism(
                name='%s.%s' % (channel, sectionlist),
                mod_path=None,
                suffix=channel,
                locations=[seclist_loc],
                preloaded=True))

    return mechanisms


def define_parameters():
    """Define parameters"""

    param_configs = json.load(open('optimal_parameters.json','r'))
    parameters = []

    for param_config in param_configs:
        if 'value' in param_config:
            frozen = True
            value = param_config['value']
            bounds = None
        elif 'bounds' in param_config:
            frozen = False
            bounds = param_config['bounds']
            value = None
        else:
            raise Exception(
                'Parameter config has to have bounds or value: %s'
                % param_config)

        if param_config['type'] == 'global':
            parameters.append(
                ephys.parameters.NrnGlobalParameter(
                    name=param_config['param_name'],
                    param_name=param_config['param_name'],
                    frozen=frozen,
                    bounds=bounds,
                    value=value))
        elif param_config['type'] in ['section', 'range']:
            if param_config['dist_type'] == 'uniform':
                scaler = ephys.parameterscalers.NrnSegmentLinearScaler()
            elif param_config['dist_type'] == 'exp':
                scaler = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                    distribution=param_config['dist'])
            seclist_loc = ephys.locations.NrnSeclistLocation(
                param_config['sectionlist'],
                seclist_name=param_config['sectionlist'])

            name = '%s.%s' % (param_config['param_name'],
                              param_config['sectionlist'])

            if param_config['type'] == 'section':
                parameters.append(
                    ephys.parameters.NrnSectionParameter(
                        name=name,
                        param_name=param_config['param_name'],
                        value_scaler=scaler,
                        value=value,
                        frozen=frozen,
                        bounds=bounds,
                        locations=[seclist_loc]))
            elif param_config['type'] == 'range':
                parameters.append(
                    ephys.parameters.NrnRangeParameter(
                        name=name,
                        param_name=param_config['param_name'],
                        value_scaler=scaler,
                        value=value,
                        frozen=frozen,
                        bounds=bounds,
                        locations=[seclist_loc]))
        else:
            raise Exception(
                'Param config type has to be global, section or range: %s' %
                param_config)

    return parameters


def define_morphology(swc_filename):
    """Define morphology"""

    return ephys.morphologies.NrnFileMorphology(swc_filename,do_replace_axon=False,do_set_nseg=True)


def simulate_optimal_model(swc_file):
    """Simulate cell model"""

    cells = []

    amp = 0.3
    delay = 125.
    dur = 500.

    cell = ephys.models.CellModel(
        'CA3',
        morph=define_morphology(swc_file),
        mechs=define_mechanisms(),
        params=define_parameters())
    cells.append(cell)

    soma_loc = ephys.locations.NrnSeclistCompLocation(name='soma',seclist_name='somatic',sec_index=0,comp_x=0.5)

    stim = ephys.stimuli.NrnSquarePulse(step_amplitude=amp,step_delay=delay,step_duration=dur,
                                        location=soma_loc,total_duration=dur+2*delay)
    rec = ephys.recordings.CompRecording(name='step.soma.v',location=soma_loc,variable='v')
    step_protocols = ephys.protocols.SequenceProtocol('step', protocols=[ephys.protocols.SweepProtocol('step', [stim], [rec])])

    nrn = ephys.simulators.NrnSimulator()
    
    #### let's simulate the optimal protocol
    responses = step_protocols.run(cell_model=cell, param_values={}, sim=nrn)

    t = np.array(responses['step.soma.v']['time'])
    V = np.array(responses['step.soma.v']['voltage'])

    #### load the saved data
    hof_responses = pickle.load(open('hall_of_fame_responses.pkl','r'))
    hof_t = hof_responses[0]['Step3.soma.v']['time']
    hof_V = hof_responses[0]['Step3.soma.v']['voltage']

    import cell_utils
    cell = cell_utils.Cell('CA3_cell',{'morphology': swc_file,
                                       'mechanisms': 'mechanisms.json',
                                       'parameters': 'optimal_parameters.json'}, h)
    cell.instantiate()
    cells.append(cell)

    cclamp = h.IClamp(cell.morpho.soma[0](0.5))
    cclamp.amp = amp
    cclamp.delay = delay
    cclamp.dur = dur

    recorders = {'t': h.Vector(), 'v': h.Vector()}
    recorders['t'].record(h._ref_t)
    recorders['v'].record(cell.morpho.soma[0](0.5)._ref_v)

    if h.cvode_active():
        print('CVode is active. minstep = %g ms. reltol = %g, abstol = %g. celsius = %g C.' % \
              (nrn.neuron.h.cvode.minstep(),nrn.neuron.h.cvode.rtol(),nrn.neuron.h.cvode.atol(),nrn.neuron.h.celsius))
    else:
        print('CVode is not active.')

    #h.cvode_active(1)
    #h.cvode.rtol(1e-6)
    #h.cvode.atol(1e-3)
    h.celsius = 36
    h.tstop = dur + 2*delay
    h.t = 0
    h.run()

    #### let's plot the results
    plt.figure()
    plt.plot(t,V,'k',label='BPO simulation')
    plt.plot(hof_t,hof_V,'b',label='BPO optimization')
    plt.plot(recorders['t'],recorders['v'],'r',label='My simulation')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$V_m$ (mV)')
    plt.legend(loc='best')
    plt.show()


def main():
    parameters,features,mechanisms,hall_of_fame,final_pop,evaluator,responses = load_files()
    plot_summary(features,hall_of_fame,final_pop,evaluator,responses)
    write_optimal_parameters(parameters,hall_of_fame,evaluator)
    # swc_file = glob.glob('*.swc')[0]
    # simulate_optimal_model(swc_file)
    

if __name__ == '__main__':
    main()
    
