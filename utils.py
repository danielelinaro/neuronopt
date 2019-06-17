
import json
import numpy as np


def extract_mechanisms(params_file, cell_name):
    mechs = json.load(open(params_file,'r'))[cell_name]['mechanisms']
    if 'alldend' in mechs:
        mechs['apical'] = mechs['alldend']
        mechs['basal'] = mechs['alldend']
        mechs.pop('alldend')
    return mechs


def build_parameters_dict(individuals, evaluator, config=None, default_parameters=None):

    cells = []

    if config is None:
        for individual in individuals:
            param_dict = evaluator.param_dict(individual)
            parameters_copy = [p.copy() for p in default_parameters]
            for par in parameters_copy:
                if 'value' not in par:
                    par['value'] = param_dict[par['param_name'] + '.' + par['sectionlist']]
                    par.pop('bounds')
            cells.append(parameters_copy)

    else:
        for individual in individuals:
            param_dict = evaluator.param_dict(individual)
            parameters = []
            for param_type,params in config['fixed'].items():
                if param_type == 'global':
                    for par in params:
                        parameters.append({'param_name': par[0], 'value': par[1], 'type': 'global'})
                elif param_type == 'all':
                    for par in params:
                        param = {'param_name': par[0], 'value': par[1], 'type': 'section',
                                 'dist_type': 'uniform', 'sectionlist': 'all'}
                        if par[2] != 'secvar':
                            print('I do not know how to deal with a fixed parameter of dist_type "{}".'.format(par[2]))
                            import ipdb
                            ipdb.set_trace()
                        parameters.append(param)
            for section_list,params in config['optimized'].items():
                for par in params:
                    param_name = par[0]
                    value = param_dict[par[0] + '.' + section_list]
                    dist_type = par[3]
                    param = {'param_name': param_name,
                             'sectionlist': section_list,
                             'value': value}

                    if param_name in ('g_pas','e_pas','cm','Ra'):
                        param['type'] = 'section'
                    else:
                        param['mech'] = param_name.split('_')[-1]
                        param['mech_param'] = '_'.join(param_name.split('_')[:-1])
                        param['type'] = 'range'

                    if dist_type == 'secvar':
                        dist_type = 'uniform'
                    elif dist_type != 'uniform':
                        param['dist'] = config['distributions'][dist_type]
                        dist_type = dist_type.split('_')[0]

                    param['dist_type'] = dist_type

                    if param['sectionlist'] == 'allnoaxon':
                        for seclist in ('somatic','apical','basal'):
                            param['sectionlist'] = seclist
                            parameters.append(param.copy())
                    elif param['sectionlist'] == 'alldend':
                        for seclist in ('apical','basal'):
                            param['sectionlist'] = seclist
                            parameters.append(param.copy())
                    else:
                        parameters.append(param)

            cells.append(parameters)

    return cells

def write_parameters(individuals, evaluator, config, default_parameters, out_files=None):

    if len(individuals.shape) == 1:
        individuals = np.array([individuals])

    if out_files is not None and type(out_files) == str:
        out_files = [out_files]
    if out_files is not None and individuals.shape[0] != len(out_files):
        raise Exception('There must be as many individuals as output file names')

    cells = build_parameters_dict(individuals, evaluator, config, default_parameters)
    
    for i,params in enumerate(cells):    
        if out_files is None:
            json.dump(params,open('individual_%d.json'%i,'w'),indent=4)
        else:
            json.dump(params,open(out_files[i],'w'),indent=4)

