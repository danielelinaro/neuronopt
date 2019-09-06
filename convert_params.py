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



def write_old_config(fixed_params_file, params_file):
    fixed_params = json.load(open(fixed_params_file,'r'))
    params = json.load(open(params_file,'r'))

    parameters = []

    for sectionlist in fixed_params:
        if sectionlist == 'global':
            for param_name, value in fixed_params[sectionlist]:
                param = {
                    'param_name': param_name,
                    'type': 'global',
                    'value': value
                }
                parameters.append(param)
        else:
            for param_name, value, dist_type in fixed_params[sectionlist]:
                param = {
                    'param_name': param_name,
                    'sectionlist': sectionlist,
                    'type': 'section',
                    'dist_type': dist_type,
                    'value': value
                }
                parameters.append(param)

    for sectionlist in params:
        for mech, param_name, min_bound, max_bound, dist_type in \
                params[sectionlist]:
            param = {
                'param_name': '%s_%s' % (param_name, mech),
                'mech': mech,
                'bounds': [min_bound, max_bound],
                'dist_type': dist_type,
                'mech_param': param_name,
                'type': 'range',
                'sectionlist': sectionlist
            }

            if mech == 'Ih':
                del param['bounds']
                param['value'] = 8e-5

            if dist_type == 'exp':
                param['dist'] = \
                    '(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}'

            parameters.append(param)

    json.dump(parameters, open(args.output, 'w'), indent=4, separators=(',', ': '))



def write_new_config(fixed_params_file, optimized_params_file, mechs_file, out_file):

    config = {
        'mechanisms': json.load(open(mechs_file,'r')),
        'distributions': {'exp': '(-0.8696 + 2.087*math.exp(({distance})*0.0031))*{value}'},
        'fixed': json.load(open(fixed_params_file,'r')),
        'optimized': json.load(open(optimized_params_file,'r'))
    }

    for data in config['fixed'].values():
        for entry in data:
            try:
                entry.pop(entry.index('uniform'))
                entry.append('secvar')
            except:
                pass

    for data in config['optimized'].values():
        for entry in data:
            mech = entry.pop(0)
            entry[0] += '_' + mech

    indent = 4
    blob = json.dumps(config, indent=indent, separators=(',', ': '))
    skip = 0
    fid = open(out_file, 'w')
    prev = ''
    for c in blob:
        if skip == 0 or not c in ('\n',' '):
            fid.write(c)
        if c == ',':
            if skip > 0 and prev == ']':
                fid.write('\n' + ' '*indent*3)
            else:
                fid.write(' ')
        if c == '[':
            skip += 1
        elif c == ']':
            skip -= 1
        prev = c
    fid.close()



def main():
    """Main"""

    progname = os.path.basename(sys.argv[0])

    parser = arg.ArgumentParser(description='Convert params.json and fixed_params.json to parameters.json format',\
                                prog=progname)
    parser.add_argument('-p', '--optimized-params', default='optimized_params.json', type=str,
                        help='parameters file (default: optimized_params.json)')
    parser.add_argument('-f', '--fixed-params', default='fixed_params.json', type=str,
                        help='fixed parameters file (default: fixed_params.json)')
    parser.add_argument('-m', '--mechs', default='mechanisms.json', type=str,
                        help='mechanisms file (default: mechanisms.json)')
    parser.add_argument('--old-config-style', action='store_true',
                        help='whether to use old configuration file output')
    parser.add_argument('-o', '--output', default='parameters.json', type=str,
                        help='output parameters file (default: parameters.json)')
    args = parser.parse_args(args=sys.argv[1:])

    if not os.path.isfile(args.fixed_params):
        print('%s: %s: no such file.' % (progname, args.fixed_params))
        sys.exit(1)

    if not os.path.isfile(args.optimized_params):
        print('%s: %s: no such file.' % (progname, args.optimized_params))
        sys.exit(2)

    if args.old_config_style:
        write_old_config(args.fixed_params, args.optimized_params, args.output)
    else:
        if not os.path.isfile(args.mechs):
            print('%s: %s: no such file.' % (progname, args.mechs))
            sys.exit(3)
        write_new_config(args.fixed_params, args.optimized_params, args.mechs, args.output)



if __name__ == '__main__':
    main()
