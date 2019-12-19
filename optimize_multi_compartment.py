import os
import sys
import time
import shutil
import pickle
import argparse as arg
import numpy as np
from numpy.random import poisson
import bluepyopt
import dlopt


def get_responses(evaluator, individuals, filename=None):
    responses = []
    for individual in individuals:
        responses.append(evaluator.run_protocols(protocols=evaluator.fitness_protocols.values(),
                                                 param_values=evaluator.param_dict(individual)))
    if not filename is None:
        pickle.dump(responses,open(filename,'wb'))

    return responses


def main():
    parser = arg.ArgumentParser(description='Optimize a multi-compartment neuron model.')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-c', '--config-dir', default='config', type=str, help='folder containing the configuration files')
    parser.add_argument('-s', '--suffix', default=None, type=str, help='suffix for configuration files')
    parser.add_argument('--cell-name', default=None, type=str, help='cell name')
    parser.add_argument('--parameters', default='parameters.json', type=str, help='parameters')
    parser.add_argument('--features', default='features.json', type=str, help='features')
    parser.add_argument('--mechanisms', default='mechanisms.json', type=str, help='mechanisms')
    parser.add_argument('--protocols', default='protocols.json', type=str, help='protocols')
    parser.add_argument('--no-mech-file', action='store_true', help='indicate that there is no mechanisms file')
    parser.add_argument('-p', '--population-size', default=100, type=int, help='population size')
    parser.add_argument('-g', '--num-generation', default=100, type=int, help='number of generations')
    parser.add_argument('--replace-axon', action='store_true', help='replace the axon in the SWC file with an AIS stub')
    parser.add_argument('--no-add-axon', action='store_true', help='do not add an AIS stub if the morphology lacks an axon')
    args = parser.parse_args(args=sys.argv[1:])

    if args.replace_axon and args.no_add_axon:
        print('You cannot specify both --replace-axon and --no-add-axon.')
        sys.exit(1)

    swc_filename = args.swc_file

    if not os.path.exists(swc_filename):
        print('%s: %s: no such file.' % (os.path.basename(sys.argv[0]), swc_filename))
        sys.exit(2)

    morpho = np.loadtxt(swc_filename)
    if np.min(np.abs(morpho[:,1] - 2)) < 0.5:
        has_axon = True
    else:
        has_axon = False

    replace_axon = args.replace_axon
    if replace_axon:
        if has_axon:
            print('The cell has an axon: replacing it with an AIS stub.')
        else:
            print('The cell has no axon: adding an AIS stub.')
    else:
        if has_axon:
            print('The cell has an axon: not replacing it with an AIS stub.')
        elif not args.no_add_axon:
            print('The cell has no axon: adding an AIS stub even though the --replace-axon ' \
                  'option was not set. To prevent this, use the --no-add-axon option.')
            replace_axon = True
        else:
            print('The cell has no axon: not adding an AIS stub because the --no-add-axon option was set.')

    cell_name = args.cell_name
    if cell_name is None:
        cell_name = os.path.basename(swc_filename).split('.')[0].replace('-','_')
        if cell_name[0] in '1234567890':
            cell_name = 'c' + cell_name

    if args.suffix is None:
        suffix = ''
    else:
        suffix = '_' + args.suffix
        
    filenames = {'morphology': swc_filename,
                 'parameters': ''.join(args.parameters.split('.')[:-1]) + suffix + '.' + args.parameters.split('.')[-1],
                 'features': ''.join(args.features.split('.')[:-1]) + suffix + '.' + args.features.split('.')[-1],
                 'protocols': ''.join(args.protocols.split('.')[:-1]) + suffix + '.' + args.protocols.split('.')[-1]}

    for f in filenames.values():
        if not os.path.isfile(f) and not os.path.isfile(args.config_dir + '/' + f):
            print('%s: %s: no such file.' % (os.path.basename(sys.argv[0]),f))
            sys.exit(1)

    if args.no_mech_file:
        filenames['mechanisms'] = ''
    else:
        filenames['mechanisms'] = ''.join(args.mechanisms.split('.')[:-1]) + suffix + '.' + args.mechanisms.split('.')[-1]

    time.sleep(poisson(10))
    while True:
        now = time.gmtime(time.time())
        output_folder = '%d%02d%02d%02d%02d%02d_%s' % (now.tm_year,now.tm_mon,now.tm_mday,
                                                       now.tm_hour,now.tm_min,now.tm_sec,
                                                       cell_name)
        if os.path.isdir(output_folder):
            nsec = poisson(5)
            print('Output folder [%s] exists: sleeping for %d seconds...' % (output_folder,nsec))
            time.sleep(nsec)
        else:
            os.mkdir(output_folder, 0o755)
            break

    for k,f in filenames.items():
        if os.path.isfile(f):
            shutil.copy(f,output_folder)
        elif os.path.isfile(args.config_dir+'/'+f):
            shutil.copy(args.config_dir+'/'+f, output_folder)
        elif not (args.no_mech_file and k == 'mechanisms'):
            print('%s: %s: cannot find file.' % (os.path.basename(sys.argv[0]),f))

    evaluator = dlopt.evaluator.create(cell_name, filenames, replace_axon=replace_axon, config_dir=args.config_dir)
    print(evaluator.cell_model)

    seed = int(time.time())
    optimisation = bluepyopt.optimisations.DEAPOptimisation(evaluator=evaluator,
                                                            offspring_size=args.population_size,
                                                            use_scoop=True,
                                                            seed=seed)

    final_pop,hall_of_fame,logbook,history = optimisation.run(max_ngen=args.num_generation)
 
    best_ind = hall_of_fame[0]
    best_ind_dict = evaluator.param_dict(best_ind)

    #### let's simulate the responses of the hall of fame population
    responses = get_responses(evaluator, hall_of_fame, output_folder+'/hall_of_fame_responses.pkl')

    #### save everything
    pickle.dump(hall_of_fame, open(output_folder+'/hall_of_fame.pkl','wb'))
    pickle.dump(final_pop, open(output_folder+'/final_population.pkl','wb'))
    pickle.dump(evaluator, open(output_folder+'/evaluator.pkl','wb'))
    pickle.dump(logbook, open(output_folder+'/logbook.pkl','wb'))
    pickle.dump(history, open(output_folder+'/history.pkl','wb'))
    pickle.dump({'seed': seed, 'replace_axon': replace_axon, 'no_add_axon': args.no_add_axon},
                open(output_folder+'/simulation_parameters.pkl','wb'))

if __name__ == '__main__':
    main()
