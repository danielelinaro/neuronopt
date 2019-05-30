"""Convert params.json and fixed_params.json to parameters.json format"""

#
# Code taken from BluePyOpt distribution: https://github.com/BlueBrain/BluePyOpt
#
# Modified by Daniele Linaro (danielelinaro@gmail.com) in September 2017
#

import os
import sys
import json
import argparse as arg

def main():
    """Main"""

    parser = arg.ArgumentParser(description='Convert params.json and fixed_params.json to parameters.json format',\
                                prog=os.path.basename(sys.argv[0]))
    parser.add_argument('--params', default='params.json', type=str,
                        help='parameters file (default: params.json)')
    parser.add_argument('--fixed-params', default='fixed_params.json', type=str,
                        help='fixed parameters file (default: fixed_params.json)')
    parser.add_argument('-o', '--output', default='parameters.json', type=str,
                        help='output parameters file (default: parameters.json)')

    args = parser.parse_args(args=sys.argv[1:])
    fixed_params = json.load(open(args.fixed_params,'r'))
    params = json.load(open(args.params,'r'))

    parameters = []

    for sectionlist in fixed_params:
        if sectionlist == 'global':
            for param_name, value in fixed_params[sectionlist]:
                param = {
                    'value': value,
                    'param_name': param_name,
                    'type': 'global'}
                parameters.append(param)
        else:
            for param_name, value, dist_type in fixed_params[sectionlist]:
                param = {
                    'value': value,
                    'param_name': param_name,
                    'type': 'section',
                    'dist_type': dist_type,
                    'sectionlist': sectionlist
                }
                parameters.append(param)

    for sectionlist in params:
        for mech, param_name, min_bound, max_bound, dist_type in \
                params[sectionlist]:
            param = {
                'bounds': [min_bound, max_bound],
                'mech': mech,
                'mech_param': param_name,
                'param_name': '%s_%s' % (param_name, mech),
                'type': 'range',
                'dist_type': dist_type,
                'sectionlist': sectionlist
            }

            if mech == 'Ih':
                del param['bounds']
                param['value'] = 8e-5

            if dist_type == 'exp':
                param['dist'] = \
                    '(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}'

            parameters.append(param)

    json.dump(parameters, open(args.output, 'w'),
              indent=4,
              separators=(',', ': '))

if __name__ == '__main__':
    main()
