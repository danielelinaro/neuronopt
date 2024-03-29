"""Run simple cell optimisation"""

"""
Copyright (c) 2016, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
# pylint: disable=R0914

import os
import json
import collections

import bluepyopt.ephys as ephys
from . import morphology


__all__ = ['define_mechanisms_v1','define_parameters_v1',
           'define_mechanisms_v2','define_parameters_v2',
           'define_morphology','create_cell']


##########################
#### START OLD VERSION ###
##########################


def define_mechanisms_v1(config_dir='config',mechanisms_file='mechanisms.json'):
    """Define mechanisms"""

    mech_definitions = json.load(open(os.path.join(config_dir,mechanisms_file)))
    
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


def define_parameters_v1(config_dir='config',parameters_file='parameters.json'):
    """Define parameters"""

    param_configs = json.load(open(os.path.join(config_dir,parameters_file)))
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


##########################
##### END OLD VERSION ####
##########################


##########################
#### START NEW VERSION ###
##########################


def multi_locations(sectionlist):
    """Define mechanisms"""

    if sectionlist == "alldend":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation("basal", seclist_name="basal")
            ]
    elif sectionlist == "allnoaxon":
        seclist_locs = [
            ephys.locations.NrnSeclistLocation("apical", seclist_name="apical"),
            ephys.locations.NrnSeclistLocation("basal", seclist_name="basal"),
            ephys.locations.NrnSeclistLocation("somatic", seclist_name="somatic")
            ]
    else:
        seclist_locs = [ephys.locations.NrnSeclistLocation(
            sectionlist,
            seclist_name=sectionlist)]

    return seclist_locs


def define_mechanisms_v2(config_dir, parameters_file, etype):
    """Define mechanisms"""

    params_path = os.path.join(config_dir, parameters_file)
    with open(params_path, 'r') as params_file:
        mech_definitions = json.load(
            params_file,
            object_pairs_hook=collections.OrderedDict)[etype]["mechanisms"]

    mechanisms = []
    for sectionlist, channels in mech_definitions.items():
        seclist_locs = multi_locations(sectionlist)
        for channel in channels:
            mechanisms.append(ephys.mechanisms.NrnMODMechanism(
                name='%s.%s' % (channel, sectionlist),
                mod_path=None,
                prefix=channel,
                locations=seclist_locs,
                preloaded=True))

    return mechanisms


def define_parameters_v2(config_dir, parameters_file, etype):
    """Define parameters"""

    params_path = os.path.join(config_dir, parameters_file)
    with open(params_path, 'r') as params_file:
        definitions = json.load(
            params_file,
            object_pairs_hook=collections.OrderedDict)[etype]

    # set distributions
    distributions = collections.OrderedDict()
    distributions["uniform"] = ephys.parameterscalers.NrnSegmentLinearScaler()

    distributions_definitions = definitions["distributions"]
    for distribution, definition in distributions_definitions.items():
        distributions[distribution] = ephys.parameterscalers.NrnSegmentSomaDistanceScaler(
                                            distribution=definition)

    parameters = []

    # set fixed parameters
    fixed_params_definitions = definitions["fixed"]
    for sectionlist, params in fixed_params_definitions.items():
        if sectionlist == 'global':
            for param_name, value in params:
                parameters.append(
                    ephys.parameters.NrnGlobalParameter(
                        name=param_name,
                        param_name=param_name,
                        frozen=True,
                        value=value))
        else:
            seclist_locs = multi_locations(sectionlist)

            for param_name, value, dist in params:

                if dist == "secvar": # this is a section variable, no distribution possible
                    parameters.append(ephys.parameters.NrnSectionParameter(
                        name='%s.%s' % (param_name, sectionlist),
                        param_name=param_name,
                        value=value,
                        frozen=True,
                        locations=seclist_locs))

                else:
                    parameters.append(ephys.parameters.NrnRangeParameter(
                        name='%s.%s' % (param_name, sectionlist),
                        param_name=param_name,
                        value_scaler=distributions[dist],
                        value=value,
                        frozen=True,
                        locations=seclist_locs))

    # Compact parameter description
    # Format ->
    # - Root dictionary: keys = section list name,
    #                    values = parameter description array
    # - Parameter description array: prefix, parameter name, minbound, maxbound

    parameter_definitions = definitions["optimized"]

    for sectionlist, params in parameter_definitions.items():
        seclist_loc = ephys.locations.NrnSeclistLocation(
            sectionlist,
            seclist_name=sectionlist)

        seclist_locs = multi_locations(sectionlist)

        for param_name, min_bound, max_bound, dist in params:

            if dist == "secvar": # this is a section variable, no distribution possible
                parameters.append(ephys.parameters.NrnSectionParameter(
                    name='%s.%s' % (param_name, sectionlist),
                    param_name=param_name,
                    bounds=[min_bound, max_bound],
                    locations=seclist_locs))

            else:
                parameters.append(ephys.parameters.NrnRangeParameter(
                    name='%s.%s' % (param_name, sectionlist),
                    param_name=param_name,
                    value_scaler=distributions[dist],
                    bounds=[min_bound, max_bound],
                    locations=seclist_locs))

            print('Setting {}.{} in model.template.define_parameters()'.format(param_name, sectionlist))


    return parameters


##########################
##### END NEW VERSION ####
##########################


def define_morphology(morphology_file,replace_axon):
    """Define morphology"""
    return ephys.morphologies.NrnFileMorphology(morphology_file, do_replace_axon=replace_axon, do_set_nseg=True)


def create_cell(cell_name, morphology_file, replace_axon=True, config_dir='config',
                mechanisms_file='mechanisms.json', parameters_file='parameters.json'):
    """Create cell model"""
    if len(mechanisms_file) == 0 or mechanisms_file is None:
        return ephys.models.CellModel(cell_name,
                                      morph=define_morphology(morphology_file, replace_axon),
                                      mechs=define_mechanisms_v2(config_dir, parameters_file, cell_name),
                                      params=define_parameters_v2(config_dir, parameters_file, cell_name))
    return ephys.models.CellModel(cell_name,
                                  morph=define_morphology(morphology_file, replace_axon),
                                  mechs=define_mechanisms_v1(config_dir, mechanisms_file),
                                  params=define_parameters_v1(config_dir, parameters_file))

