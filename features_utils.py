import os
import sys
import csv
import glob
import json
import pickle
import numpy as np
import argparse as arg
import matplotlib.pyplot as plt

progname = os.path.basename(sys.argv[0])

EFEL_AP_threshold = -20

# This is the full set of voltage features to extract for each individual cell, irrespective
# of its type. The set of features that will be used in the optimization may be different, and
# is specified by the variable "feature_names"
feature_names_full_set = ['AP_amplitude','AP_begin_voltage','spike_half_width',
                          'time_to_first_spike','adaptation_index2',
                          'ISI_values','ISI_CV','doublet_ISI',
                          'min_AHP_values','AHP_slow_time','AHP_depth_abs_slow',
                          'Spikecount','fast_AHP','burst_mean_freq','interburst_voltage',
                          'AP_rise_rate','AP_fall_rate','AP_amplitude_change',
                          'AP_duration_change','AP_rise_rate_change','AP_fall_rate_change',
                          'AP_duration_half_width_change','amp_drop_first_second',
                          'mean_frequency','AP_height','AP_width','AHP_depth_abs',
                          'voltage_base', 'steady_state_voltage',
                          'voltage_deflection', 'voltage_deflection_begin',
                          'Spikecount', 'time_to_last_spike',
                          'inv_time_to_first_spike', 'inv_first_ISI',
                          'inv_second_ISI', 'inv_third_ISI', 'inv_fourth_ISI',
                          'inv_fifth_ISI', 'inv_last_ISI', 'AP_amplitude_from_voltagebase']

feature_names = {'CA3': ['AP_amplitude','AP_begin_voltage','spike_half_width',
                         'AP_fall_rate','AP_rise_rate','AHP_slow_time',
                         'voltage_base','steady_state_voltage',
                         'ISI_CV','Spikecount','doublet_ISI',
                         'time_to_first_spike','time_to_last_spike','adaptation_index2',
                         'ISI_values','AHP_depth_abs_slow','fast_AHP',
                         'min_AHP_values','inv_first_ISI','inv_second_ISI','inv_third_ISI'],
                 'BBP_CTX': ['AP_height', 'AHP_slow_time', 'ISI_CV',
                             'doublet_ISI','AHP_depth_abs_slow',
                             'AP_width','time_to_first_spike','AHP_depth_abs',
                             'adaptation_index2','mean_frequency','Spikecount',
                             'voltage_base','voltage_deflection'],
                 'BBP_CTX_bAP': ['AP_height', 'AP_width', 'Spikecount',
                                 'voltage_base', 'AP_amplitude_from_voltagebase'],
                 'BBP_HPC': ['voltage_base', 'steady_state_voltage',
                             'voltage_deflection', 'voltage_deflection_begin',
                             'Spikecount', 'time_to_last_spike',
                             'inv_time_to_first_spike', 'inv_first_ISI',
                             'inv_second_ISI', 'inv_third_ISI', 'inv_fourth_ISI',
                             'inv_fifth_ISI', 'inv_last_ISI'],
                 'RS-01': ['AP_amplitude','AP_begin_voltage','spike_half_width',
                           'time_to_first_spike','adaptation_index2',
                           'ISI_values','ISI_CV','doublet_ISI',
                           'min_AHP_values','AHP_slow_time','AHP_depth_abs_slow',
                           'Spikecount','fast_AHP','AP_fall_rate','AP_rise_rate'],
                 'RS-02': ['AP_height','AP_begin_voltage','spike_half_width',
                           'time_to_first_spike','adaptation_index2',
                           'ISI_values','ISI_CV','doublet_ISI',
                           'min_AHP_values','AHP_slow_time','AHP_depth_abs_slow',
                           'fast_AHP','AP_fall_rate','AP_rise_rate','mean_frequency'],
                 'B-01': ['AP_amplitude','AP_begin_voltage','spike_half_width',
                          'time_to_first_spike','doublet_ISI',
                          'min_AHP_values','AHP_slow_time','AHP_depth_abs_slow',
                          'Spikecount','AP_fall_rate','AP_rise_rate',
                          'AP_duration_change','AP_rise_rate_change','AP_fall_rate_change',
                          'amp_drop_first_second'],
                 'B-02': ['AP_amplitude','AP_begin_voltage','spike_half_width',
                          'time_to_first_spike','doublet_ISI',
                          'ISI_values','ISI_CV','burst_mean_freq','interburst_voltage',
                          'Spikecount','AP_fall_rate','AP_rise_rate',
                          'AP_duration_change','AP_rise_rate_change','AP_fall_rate_change',
                          'amp_drop_first_second'],
                 'test': feature_names_full_set}


############################################################
###                        WRITE                         ###
############################################################


def write_features():
    parser = arg.ArgumentParser(description='Write configuration file using features from multiple cells.',\
                                prog=progname+' write')
    parser.add_argument('files', type=str, nargs='+',
                        help='pkl files containing the data relative to each cell')
    parser.add_argument('-N', '--nsteps', default=3, type=int,
                        help='number of current steps to include in the protocols (default: 3)')
    parser.add_argument('--all-amps', action='store_true', help='use all amplitudes in the pickle file')
    parser.add_argument('--step-amps', type=str,
                        help='current amplitudes to include in the protocols, in alternative to the --nsteps option')
    parser.add_argument('--round-amp', default=0.025, type=float,
                        help='the current amplitudes will be rounded to the closest integer multiple of this quantity (default: 0.025 nA)')
    parser.add_argument('--features-file', default=None,
                        help='output features file name (deault: features.json)')
    parser.add_argument('--protocols-file', default=None,
                        help='output protocols file name (deault: protocols.json)')
    parser.add_argument('-o', '--suffix', default='',
                        help='suffix for the output file names (default: no suffix)')
    parser.add_argument('--cell-type', default='CA3',
                        help='feature set to use (default: "CA3")')
    parser.add_argument('--recording-site', default='soma', help='recording site (default: "soma")')
    parser.add_argument('--stim-start', default=None, type=float, help='delay before application of the stimulus')
    parser.add_argument('--stim-dur', default=None, type=float, help='duration of the stimulus')
    parser.add_argument('--after', default=500, type=float, help='time after the application of the stimulus')
    parser.add_argument('--prompt-user', action='store_true', help='ask the user whether to remove protocols that do not have all features')

    args = parser.parse_args(args=sys.argv[2:])

    for f in args.files:
        if not os.path.isfile(f):
            print('%s: %s: no such file.' % (progname,f))
            sys.exit(1)

    if len(args.files) > 1 and args.all_amps:
        print('--all-amps can only be used when there is only one pickle file')
        sys.exit(1)

    if args.all_amps and not args.step_amps is None:
        print('--all-amps and --step-amps cannot be used simultaneously')
        sys.exit(1)

    if args.all_amps:
        data = pickle.load(open(args.files[0],'rb'))
        desired_amps = np.unique(data['current_amplitudes'])
        nsteps = len(desired_amps)
    else:
        nsteps = args.nsteps
        desired_amps = None

    if not args.step_amps is None:
        if os.path.isfile(args.step_amps):
            desired_amps = json.load(open(args.step_amps, 'r'))
            nsteps = [len(amp) for amp in desired_amps.values()]
            if len(set(nsteps)) != 1:
                print('The number of step amplitudes in file {} must be the same for all cells.'.format(args.step_amps))
                sys.exit(1)
            nsteps = nsteps[0]
        else:
            desired_amps = list(map(float,args.step_amps.split(',')))
            nsteps = len(desired_amps)

    if nsteps <= 0:
        print('%s: the number of features must be greater than 0.' % progname)
        sys.exit(1)

    if desired_amps is None and args.round_amp <= 0:
        print('%s: the rounding amplitude  must be greater than 0.' % progname)
        sys.exit(1)

    recording_site = args.recording_site

    if args.features_file is None:
        features_file = 'features'
    else:
        features_file = args.features_file
    if args.suffix != '':
        features_file += '_' + args.suffix
    features_file += '.json'

    if args.protocols_file is None:
        protocols_file = 'protocols'
    else:
        protocols_file = args.protocols_file
    if args.suffix != '':
        protocols_file += '_' + args.suffix
    protocols_file += '.json'

    if not args.cell_type in feature_names.keys():
        print('Unknown cell type "%s". Available values are "%s".' % \
              (args.cell_type,'", "'.join(feature_names.keys())))
        sys.exit(1)

    if args.stim_start is not None and args.stim_start < 0:
        print('Time before stimulus must be greater than 0.')
        sys.exit(1)

    if args.stim_dur is not None and args.stim_dur < 0:
        print('Stimulus duration must be greater than 0.')
        sys.exit(1)

    if args.after < 0:
        print('Time after stimulus must be greater than 0.')
        sys.exit(1)

    amplitudes = []
    features = []
    rheobases = []
    for f in args.files:
        data = pickle.load(open(f,'rb'))
        stim_dur = data['stim_dur']
        stim_start = data['stim_start']
        features.append(data['features'])
        rheobases.append(np.min(data['current_amplitudes'][np.where(data['has_spikes'])[0]]))
        amplitudes.append(data['current_amplitudes'] - rheobases[-1])
    print('Mean rheobase: %g nA.' % np.mean(rheobases))

    if args.stim_start is not None:
        stim_start = args.stim_start
    if args.stim_dur is not None:
        stim_dur = args.stim_dur

    if desired_amps is None:
        desired_amps = np.zeros((len(amplitudes),nsteps))
        for i,(amplitude,feature) in enumerate(zip(amplitudes,features)):
            amps = np.unique(amplitude)
            if len(amps) < nsteps:
                print('Not enough distinct amplitudes.')
                sys.exit(0)
            j = 0
            while feature[j]['Spikecount'][0] == 0:
                j = j+1
            min_amp = amps[j]
            max_amp = amps[-1]
            amp_step = (max_amp-min_amp)/(nsteps-1)
            desired_amps[i,:] = np.arange(min_amp,max_amp+amp_step/2,amp_step)
            for j in range(nsteps):
                if not desired_amps[i,j] in amps:
                    desired_amps[i,j] = amps[np.argmin(np.abs(amps - desired_amps[i,j]))]
    else:
        if isinstance(desired_amps, dict):
            desired_amps = np.array(list(desired_amps.values()))
        else:
            desired_amps = np.tile(np.array(desired_amps),(len(amplitudes),1))
        for i in range(len(rheobases)):
            desired_amps[i] -= rheobases[i]

    RHEOBASE = np.mean(rheobases)

    protocols_dict = {}
    for i in range(nsteps):
        stepnum = 'Step%d'%(i+1)
        protocols_dict[stepnum] = {'stimuli': [{
            'delay': stim_start, 'amp': RHEOBASE + np.round(np.mean(desired_amps[:,i])/args.round_amp)*args.round_amp,
            'duration': stim_dur, 'totduration': stim_dur+stim_start+args.after}]}

    #for sublist in l:
    #    if sublist is not None:
    #        for item in sublist:
    #            item
    flatten = lambda l: [item for sublist in l if sublist is not None for item in sublist]

    all_features = [{} for i in range(nsteps)]
    features_dict = {'Step%d'%i: {recording_site: {}} for i in range(1,nsteps+1)}
    for name in feature_names[args.cell_type]:
        for i in range(len(args.files)):
            for j in range(nsteps):
                idx, = np.where(np.abs(amplitudes[i] - desired_amps[i][j]) < 1e-6)
                for k in idx:
                    if name in features[i][k]:
                        if not name in all_features[j]:
                            all_features[j][name] = []
                        all_features[j][name].append(features[i][k][name])
        for i in range(nsteps):
            if name in all_features[i]:
                stepnum = 'Step%d' % (i+1)
                all_features[i][name] = flatten(all_features[i][name])
                if len(all_features[i][name]) > 0:
                    features_dict[stepnum][recording_site][name] = [np.mean(all_features[i][name]),
                                                                    np.std(all_features[i][name])]
                    if features_dict[stepnum][recording_site][name][1] == 0:
                        std = np.abs(features_dict[stepnum][recording_site][name][0]/5)
                        if std == 0:
                            features_dict[stepnum][recording_site].pop(name)
                        else:
                            features_dict[stepnum][recording_site][name][1] = std
                            print('Standard deviation of feature {} for {} @ {} is 0: setting it to {:.4f}.'\
                                  .format(name,stepnum,recording_site,features_dict[stepnum][recording_site][name][1]))

    num_features = len(feature_names[args.cell_type])
    to_remove = []
    for stepnum,step in features_dict.items():
        if len(step[recording_site]) == 0:
            to_remove.append(stepnum)
        if args.prompt_user and len(step[recording_site]) < num_features:
            print('Not all features were extracted for protocol "%s".' % stepnum)
            print('The extracted features are the following:\n')
            for i,feat in enumerate(step[recording_site]):
                print('[%02d] %s' % (i+1,feat))
            print('')
            while True:
                resp = input('Remove this protocol? [yes/no] ')
                if resp.lower() == 'yes':
                    to_remove.append(stepnum)
                    break
                elif resp.lower() == 'no':
                    break
                else:
                    print('Please enter yes or no.')

    for stepnum in to_remove:
        features_dict.pop(stepnum)
        protocols_dict.pop(stepnum)
    json.dump(features_dict, open(features_file,'w'),indent=4)
    json.dump(protocols_dict, open(protocols_file,'w'),indent=4)


############################################################
###                      WRITE-XLS                       ###
############################################################


def read_sheet(book, stimulus_name, recording_site):
    if stimulus_name in book:
        sheet_name = stimulus_name
    elif (stimulus_name + '_' + recording_site) in book:
        sheet_name = stimulus_name + '_' + recording_site
    else:
        return None
    sheet = book[sheet_name]

    j = 1
    while True:
        interval = 'A{}:A{}'.format(j,j+2)
        rows = sheet[interval]
        if rows[0][0].value == 'Feature' and rows[1][0].value == 'Mean' and rows[2][0].value == 'Std':
            start = j
            break
        j += 1

    print('Features start at line {} in sheet {}.'.format(j, sheet_name))
    features = {}
    for letter in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
        interval = '{}{}:{}{}'.format(letter,start,letter,start+2)
        rows = sheet[interval]
        if rows[0][0].value is None:
            break
        feature_name = rows[0][0].value
        feature_mean = float(rows[1][0].value)
        feature_std = float(rows[2][0].value)
        if not np.isnan(feature_mean):
            if np.isnan(feature_std) or feature_std == 0:
                feature_std = feature_mean / 5
                print('Invalid value of standard deviation for feature {} in sheet {}: setting it to {.3f}.' \
                      .format('\033[93m'+feature_name+'\033[0m', '\033[94m'+stimulus_name+'\033[0m', feature_std))
            features[feature_name] = [feature_mean, feature_std]
        else:
            print('Value of feature {} in sheet {} is \033[93mNaN\033[0m: not adding feature to the JSON file.' \
                  .format('\033[93m'+feature_name+'\033[0m', '\033[94m'+stimulus_name+'\033[0m'))
    return features


def write_features_xls():
    bap_amplitude = 4
    bap_dur = 2
    bap_delay = 1000

    parser = arg.ArgumentParser(description='Write configuration file using features from multiple cells stored in an Excel file.',
                                prog=progname+' write-xls')
    parser.add_argument('file', type=str, help='the Excel file contaning the features')
    parser.add_argument('--step-amps', required=True, type=str,
                        help='current amplitudes to include in the protocols, comma separated')
    parser.add_argument('--features-file', default='features.json',
                        help='output features file name (deault: features.json)')
    parser.add_argument('--protocols-file', default='protocols.json',
                        help='output protocols file name (deault: protocols.json)')
    parser.add_argument('-o', '--suffix', default='',
                        help='suffix for the output file names (default: no suffix)')
    parser.add_argument('--recording-sites', default='soma', help='recording sites (default: "soma")')
    parser.add_argument('--stim-start', default=1000, type=float, help='delay before application of the stimulus')
    parser.add_argument('--stim-dur', required=True, type=float, help='duration of the stimulus')
    parser.add_argument('--after', default=500, type=float, help='time after the application of the stimulus')
    parser.add_argument('--with-bap', action='store_true', help='add b-AP stimulus')
    parser.add_argument('--bap-amplitude', default=None, type=float, \
                        help='b-AP stimulus amplitude (default: {} pA)'.format(bap_amplitude))
    parser.add_argument('--bap-dur', default=None, type=float, \
                        help='b-AP stimulus duration (default: {} ms)'.format(bap_dur))
    parser.add_argument('--bap-delay', default=None, type=float, \
                        help='b-AP stimulus delay (default: {} ms)'.format(bap_delay))
    args = parser.parse_args(args=sys.argv[2:])

    xls_file = args.file
    if not os.path.isfile(xls_file):
        print('{}: {}: no such file.'.format(progname, xls_file))
        sys.exit(1)

    amplitudes = np.array([float(amp) for amp in args.step_amps.split(',')])
    n_steps = len(amplitudes)

    delay = args.stim_start
    if delay < 0:
        print('{}: the beginning of the stimulus must be >= 0.'.format(progname))
        sys.exit(2)

    dur = args.stim_dur
    if dur <= 0:
        print('{}: the duraton of the stimulus must be > 0.'.format(progname))
        sys.exit(3)

    after = args.after
    if after < 0:
        print('{}: the time after the application of the stimulus must be >= 0.'.format(progname))
        sys.exit(4)

    recording_sites = args.recording_sites.split(',')
    n_recording_sites = len(recording_sites)

    if n_recording_sites > 1:
        try:
            extra_recordings = json.load(open('extra_recordings.json','r'))
        except:
            print('{}: file `extra_recordings.json` is required when more than one recording site are present.'.format(progname))
            sys.exit(5)

    if args.with_bap:
        with_bap = True
    elif args.bap_amplitude is not None or args.bap_dur is not None or args.bap_delay is not None:
        with_bap = True
    else:
        with_bap = False

    if with_bap and n_recording_sites == 1:
        print('{}: cannot have b-AP stimulus with just one recording site.'.format(progname))
        sys.exit(6)

    if args.bap_amplitude is not None:
        bap_amplitude = args.bap_amplitude
    if args.bap_dur is not None:
        bap_dur = args.bap_dur
    if args.bap_delay is not None:
        bap_delay = args.bap_delay

    if bap_amplitude > 50:
        print('{}: pulse amplitude for b-AP protocol is > 50 nA: are you sure that is correct?')
        sys.exit(7)

    if bap_dur < 0.2:
        print('{}: pulse duration for b-AP protocol is < 0.2 ms: are you sure that is correct?')
        sys.exit(8)

    import openpyxl
    book = openpyxl.load_workbook(xls_file)

    features = {}
    protocols = {}

    for i in range(n_steps):
        step = 'Step{}'.format(i+1)
        protocols[step] = {'stimuli': [
            {'delay': delay, 'amp': amplitudes[i], 'duration': dur, 'totduration': delay+dur+after}
        ]}
        if step in extra_recordings:
            protocols[step]['extra_recordings'] = extra_recordings[step]['extra_recordings']
        features[step] = {}
        for recording_site in recording_sites:
            feat = read_sheet(book, step, recording_site)
            if feat is None:
                print('{} does not contain a sheet named {} or {}.'.format(xls_file, step, step+'_'+recording_site))
                continue
            features[step][recording_site] = feat
    if with_bap:
        protocols['bAP'] = {'stimuli': [
            {'delay': bap_delay, 'amp': bap_amplitude, 'duration': bap_dur, \
             'totduration': bap_delay+bap_dur+100}
        ]}
        protocols['bAP']['extra_recordings'] = extra_recordings['bAP']['extra_recordings']
        features['bAP'] = {site: read_sheet(book, 'bAP', site) for site in recording_sites}

    if args.suffix != '':
        if args.suffix[0] in ('-','_'):
            suffix = args.suffix
        else:
            suffix = '_' + args.suffix
        fname,ext = os.path.splitext(args.features_file)
        features_file = fname + suffix + ext
        fname,ext = os.path.splitext(args.protocols_file)
        protocols_file = fname + suffix + ext
    json.dump(features, open(features_file,'w'), indent=4)
    json.dump(protocols, open(protocols_file,'w'), indent=4)


############################################################
###                       EXTRACT                        ###
############################################################


def extract_features_from_LCG_files(files_in, kernel_file, file_out):
    import lcg
    import aec
    import efel
    efel.setThreshold(EFEL_AP_threshold)
    Ke = np.loadtxt(kernel_file)
    traces = []
    amplitudes = []
    for f in files_in:
        entities,info = lcg.loadH5Trace(f)
        stim_start = entities[1]['metadata'][0,0]*1e3
        stim_dur = entities[1]['metadata'][1,0]*1e3
        I = entities[1]['data']
        Vr = entities[0]['data']
        Vc = aec.compensate(Vr,I*1e-9,Ke)
        time = np.arange(len(Vc)) * info['dt'] * 1e3
        if np.max(Vc) > -20:
            traces.append({'T': time, 'V': Vc, 'stim_start': [stim_start], 'stim_end': [stim_start+stim_dur]})
            amplitudes.append(entities[1]['metadata'][1,2]*1e-3)
            print(amplitudes[-1])
            plt.plot(time,Vc,'k',lw=1)
            plt.show()
    features = efel.getFeatureValues(traces,feature_names_full_set)
    data = {'features': features, 'current_amplitudes': np.array(amplitudes), \
            'stim_dur': stim_dur, 'stim_start': stim_start}
    pickle.dump(data,open(file_out,'wb'))


def extract_features_from_file(file_in,stim_dur,stim_start,sampling_rate,offset=0):
    import igor.binarywave as ibw
    import efel
    efel.setThreshold(EFEL_AP_threshold)
    data = ibw.load(file_in)
    voltage = data['wave']['wData']
    if len(voltage.shape) == 1:
        voltage = np.array([voltage])
    elif voltage.shape[0] > voltage.shape[1]:
        voltage = voltage.T
    time = np.arange(voltage.shape[1]) / sampling_rate

    stim_end = stim_start + stim_dur
    if stim_dur > 100:
        idx, = np.where((time>stim_start-10) & (time<=stim_end+10))
    else:
        idx, = np.where((time>stim_start-5) & (time<=stim_end+5))
    traces = [{'T': time[idx], 'V': sweep[idx], 'stim_start': [stim_start], 'stim_end': [stim_end]} \
              for sweep in voltage]
    voltage_range = [np.min(voltage),np.max(voltage)]
    recording_dur = time[-1]

    if voltage_range[0] > -100 and voltage_range[1] < 100:
        plt.plot(time[idx],voltage[:,idx].T+offset,'k',lw=1)

    return efel.getFeatureValues(traces,feature_names_full_set),voltage_range,recording_dur


def extract_features_from_files(files_in,current_amplitudes,stim_dur,stim_start,sampling_rate=20,files_out=[],quiet=False):
    if type(files_out) != list:
        files_out = [files_out]
    if len(files_out) == 1:
        features = []
        has_spikes = []
        to_keep = []
        offset = 0
        for i,f in enumerate(files_in):
            print('I = %g nA.' % current_amplitudes[i])
            feat,voltage_range,_ = extract_features_from_file(f,stim_dur,stim_start,sampling_rate)
            if 'Spikecount' in feat[0]:
                # Spikecount feature is present
                with_spikes = [fe['Spikecount'][0] > 0 for fe in feat]
            elif 'mean_frequency' in feat[0]:
                # mean_frequency feature is present
                with_spikes = [True if fe['mean_frequency'] is not None else False for fe in feat]
            good, = np.where([not all(v is None or len(v) == 0 for v in fe.values()) for fe in feat])
            if voltage_range[0] > -100 and voltage_range[1] < 100:
                for j in good:
                    features.append({k: v for k,v in feat[j].items() if v is not None and len(v) > 0})
                    has_spikes.append(with_spikes[j])
                    to_keep.append(offset + j)
            offset += len(feat)
        amplitudes = [current_amplitudes[i] for i in to_keep]
        idx = np.argsort(amplitudes)
        amplitudes = [amplitudes[jdx] for jdx in idx]
        features = [features[jdx] for jdx in idx]
        has_spikes = [has_spikes[jdx] for jdx in idx]
        data = {'features': features, 'current_amplitudes': np.array(amplitudes), \
                'stim_dur': stim_dur, 'stim_start': stim_start,
                'has_spikes': np.array(has_spikes)}
        pickle.dump(data,open(files_out[0],'wb'))
    else:
        if len(files_out) == 0:
            files_out = [x+'.pkl' for x in files_in]
        elif len(files_out) != len(files_in):
            raise Exception('There must be as many input as output files')
        for f_in,f_out in zip(files_in,files_out):
            feat,_ = extract_features_from_file(f_in)
            pickle.dump(feat,open(f_out,'wb'))

    if not quiet:
        print('-------------------------------------------------------')
        for feat,amp in zip(features,amplitudes):
            print('>>> Amplitude: %g nA' % amp)
            for name,values in feat.items():
                try:
                    print('%s has the following values: %s' % \
                          (name, ', '.join([str(x) for x in values])))
                except:
                    print('{} has the following values: \033[91m'.format(name) + str(values) + '\033[0m')
            print('-------------------------------------------------------')


def read_ibw_history_file(filename):
    import igor.binarywave as ibw
    data = ibw.load(filename)
    rows,cols = data['wave']['wData'].shape
    tmp = [[] for _ in range(cols)]
    m,n = 0,0
    for i in range(cols):
        for j in range(rows):
            tmp[i].append(data['wave']['wData'][m][n])
            n += 1
            if n == cols:
                n = 0
                m += 1
    info = {}
    for lst in tmp:
        k = lst[0].decode('UTF-8').replace(' ','_').lower()
        info[k] = []
        for elem in lst[1:]:
            v = elem.decode('UTF-8')
            if v  == '':
                info[k].append(None)
            elif '.' in v:
                try:
                    info[k].append(float(v))
                except:
                    info[k].append(v)
            else:
                try:
                    info[k].append(int(v))
                except:
                    info[k].append(v)
    return info


def read_tab_delim_file(filename):
    with open(filename,'r') as fid:
        header = fid.readline()
        keys = [k.strip().lower().replace(' ','_') for k in header.split('\t')]
        data = {k: [] for k in keys}
        reader = csv.reader(fid,delimiter='\t')
        for line in reader:
            for k,v in zip(keys,line):
                if v == '':
                    data[k].append(None)
                elif '.' in v:
                    try:
                        data[k].append(float(v))
                    except:
                        data[k].append(v)
                else:
                    try:
                        data[k].append(int(v))
                    except:
                        data[k].append(v)
    return data


def parse_sweeps(filename):
    with open(filename, 'r') as fid:
        sweeps = []
        for line in fid.read().splitlines():
            if '-' in line:
                ss = [int(x) for x in line.split('-')]
                for sweep in range(ss[0], ss[1]+1):
                    sweeps.append(sweep)
            else:
                sweeps.append(int(line))
    return sweeps


def extract_features():
    parser = arg.ArgumentParser(description='Extract ephys features from recordings.',\
                                prog=progname+' extract')
    parser.add_argument('-d', '--folder', default=None,
                        help='the folder where data is stored')
    parser.add_argument('-f', '--file', default=None,
                        help='the file where data is stored')
    parser.add_argument('-F', '--sampling-rate', default=20., type=float,
                        help='the sampling rate at which data was recorded (default 20 kHz)')
    parser.add_argument('--history-file', default='DP_Sweeper/history.ibw', type=str,
                        help='history file (default: DP_Sweeper/history.ibw)')
    parser.add_argument('--stim-dur', default=500., type=float,
                        help='Stimulus duration (default 500 ms)')
    parser.add_argument('--stim-start', default=125., type=float,
                        help='beginning of the stimulus (default 125 ms)')
    parser.add_argument('--Imin', default=None, type=float,
                        help='minimum injected current, in nA')
    parser.add_argument('--Istep', default=0.1, type=float,
                        help='current step (default 0.1 nA)')
    parser.add_argument('--Ipulse', default=None, type=float,
                        help='current pulse (default 2 nA)')
    parser.add_argument('--spike-threshold', default=-20., type=float,
                        help='spike threshold (default -20 mV)')
    parser.add_argument('--quiet', action='store_true', help='be quiet')

    args = parser.parse_args(args=sys.argv[2:])

    if args.folder is not None and args.file is not None:
        print('Cannot specify both folder and file simultaneously.')
        sys.exit(1)

    if args.folder is not None:
        if not os.path.isdir(args.folder):
            print('%s: %s: no such directory.' % (progname,args.folder))
            sys.exit(1)
        folder = os.path.abspath(args.folder)
        if len(glob.glob(folder + '/*.ibw')) > 0:
            mode = 'CA3'
        else:
            mode = 'LCG'
    else:
        if not os.path.isfile(args.file):
            print('%s: %s: no such file.' % (progname,args.file))
            sys.exit(1)
        filename = os.path.abspath(args.file)
        folder = os.path.dirname(filename)
        mode = 'cortex'

    if args.sampling_rate <= 0:
        print('%s: the sampling rate must be positive.' % progname)
        sys.exit(2)

    if args.stim_dur <= 0:
        print('The duration of the stimulus must be positive.')
        sys.exit(3)

    if mode == 'CA3':
        history_file = folder + '/' + args.history_file
        if not os.path.isfile(history_file):
            print('%s: %s: no such file.' % (progname,args.history_file))
            sys.exit(4)
        if history_file[-3:] == 'ibw':
            info = read_ibw_history_file(history_file)
        elif history_file[-3:] == 'txt':
            info = read_tab_delim_file(history_file)
        else:
            print('%s: %s: unknown history file type. File suffix should be either ibw or txt.' % \
                  (progname+'-extract',history_file))
            sys.exit(5)

        try:
            sweeps_to_ignore = parse_sweeps(folder + '/IGNORE_SWEEPS')
        except:
            sweeps_to_ignore = []
        try:
            good_sweeps = parse_sweeps(folder + '/GOOD_SWEEPS')
        except:
            good_sweeps = [i for i in range(1,1001) if not i in sweeps_to_ignore]
        if len(np.intersect1d(sweeps_to_ignore, good_sweeps)) > 0:
            print('IGNORE_SWEEPS and GOOD_SWEEPS are not mutually exclusive.')
            sys.exit(6)
        files_in = []
        file_out = folder.split('/')[-1] + '.pkl'
        current_amplitudes = []
        n = len(info['sweep_index'])
        for i in range(n):
            if (info['builder_name'][i] == 'StepPulse' or info['builder_name'][i] == 'BuiltinPulse') \
               and info['sweep_index'].count(info['sweep_index'][i]) == 2:
                params = info['builder_parameters'][i].split(';')
                for p in params:
                    if 'duration' in p:
                        dur = float(p.split('=')[1])
                    elif 'amplitude' in p:
                        amp = float(p.split('=')[1])*info['multiplier'][i]*1e-3
                flag = True
            elif info['builder_name'][i] == 'Train' and info['sweep_index'].count(info['sweep_index'][i]) == 3:
                params = info['builder_parameters'][i].split(';')
                for p in params:
                    if 'pulseDuration' in p:
                        dur = float(p.split('=')[1])
                    elif 'amplitude' in p:
                        amp = float(p.split('=')[1])*info['multiplier'][i]*1e-3
                flag = False
            else:
                flag = False
            if flag and not info['sweep_index'][i] in sweeps_to_ignore and \
               info['sweep_index'][i] in good_sweeps and \
               dur == args.stim_dur:
                print('[%02d] dur=%g ms, amp=%g nA' % (info['sweep_index'][i],dur,amp))
                files_in.append('%s/ad0_%d.ibw' % (folder,info['sweep_index'][i]))
                current_amplitudes.append(amp)
    elif mode == 'cortex':
        import igor.binarywave as ibw
        files_in = [filename]
        file_out = os.path.basename(filename).split('.')[0] + '.pkl'
        data = ibw.load(filename)
        nsteps = data['wave']['wData'].shape[1]
        if args.Ipulse is None:
            Istep = args.Istep
            if args.Imin is None:
                s = np.std(data['wave']['wData'],0)
                Imin = -args.Istep*np.argmin(s)
                print('Guessing the minimum value of injected current: %g nA.' % Imin)
            else:
                Imin = args.Imin
            current_amplitudes = np.arange(nsteps)*Istep + Imin
            current_amplitudes = current_amplitudes.tolist()
        else:
            Ipulse = args.Ipulse
            current_amplitudes = [Ipulse for _ in range(nsteps)]
    elif mode == 'LCG':
        h5_files = glob.glob(folder + '/*.h5')
        files_in = []
        file_out = folder.split('/')[-2] + '.pkl'
        for file in h5_files:
            if not os.path.isfile(file.split('.h5')[0] + '_kernel.dat'):
                files_in.append(file)
            else:
                kernel_file = file.split('.h5')[0] + '_kernel.dat'

    EFEL_AP_threshold = args.spike_threshold

    if mode == 'LCG':
        extract_features_from_LCG_files(files_in, kernel_file, folder+'/'+file_out)
    else:
        extract_features_from_files(files_in,current_amplitudes,args.stim_dur,args.stim_start,
                                    args.sampling_rate,files_out=[folder+'/'+file_out],quiet=args.quiet)

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.savefig(folder+'/'+file_out.split('.pkl')[0]+'.pdf',dpi=300)
    if not args.quiet:
        plt.show()


############################################################
###                         DIFF                         ###
############################################################


def diff_features():
    parser = arg.ArgumentParser(description='Show differences between two feature files.',
                                prog=progname+' diff')
    parser.add_argument('--recording-site', default='soma', help='recording site (default: "soma")')
    parser.add_argument('files', type=str, nargs=2, help='feature files')

    args = parser.parse_args(args=sys.argv[2:])

    for f in args.files:
        if not os.path.isfile(f):
            print('%s: %s: no such file.' % (progname,f))
            sys.exit(1)

    recording_site = args.recording_site

    features = [json.load(open(f,'r')) for f in args.files]
    arrow = ['<','>']
    for step_num in features[0].keys():
        printed_header = False
        if step_num in features[1].keys():
            for feature_name in features[0][step_num][recording_site].keys():
                if feature_name in features[1][step_num][recording_site].keys():
                    feature_values = np.array([feat[step_num][recording_site][feature_name] for feat in features])
                    if np.any(np.abs(np.diff(feature_values,axis=0)) > 1e-6):
                        if not printed_header:
                            print('%s:' % step_num)
                            printed_header = True
                        print('\t%s: %s [%g,%g (%.0f%%)] %s [%g,%g (%.0f%%)]' % (feature_name,arrow[0],
                                                                                 feature_values[0,0],feature_values[0,1],
                                                                                 np.abs(feature_values[0,1]/feature_values[0,0]*100),
                                                                                 arrow[1],
                                                                                 feature_values[1,0],feature_values[1,1],
                                                                                 np.abs(feature_values[1,1]/feature_values[1,0]*100)))

    for this in range(2):
        other = (this+1)%2
        for step_num in features[this].keys():
            if not step_num in features[other].keys():
                print('%s %s' % (arrow[this],step_num))
            else:
                for feature_name in features[this][step_num][recording_site].keys():
                    if not feature_name in features[other][step_num][recording_site].keys():
                        print('%s %s:%s' % (arrow[this],step_num,feature_name))


############################################################
###                         DUMP                         ###
############################################################


def dump_features():
    parser = arg.ArgumentParser(description='Dump feature files into several CSV files.',
                                prog=progname+' dump')
    parser.add_argument('files', type=str, nargs='+', help='feature files')
    parser.add_argument('--recording-site', default='soma', help='recording site (default: "soma")')
    parser.add_argument('--no-std', action='store_true', help='do not dump the standard deviation')

    args = parser.parse_args(args=sys.argv[2:])

    recording_site = args.recording_site

    for f in args.files:
        if not os.path.isfile(f):
            print('%s: %s: no such file.' % (progname,f))
            sys.exit(1)

    from itertools import chain

    features = []
    for f in args.files:
        features.append(json.load(open(f,'r')))
    labels = args.files
    nsteps = 9
    for i in range(1,nsteps+1):
        stepnum = 'Step%d' % i
        fid = open('{}_{}.csv'.format(stepnum, recording_site), 'w')
        fid.write('Feature,')
        for lbl in labels:
            if args.no_std:
                fid.write('%s (mean),' % lbl.replace('_',' '))
            else:
                fid.write('%s (mean),%s (std),' % (lbl.replace('_',' '),lbl.replace('_',' ')))
        fid.write('\n')
        j = 0
        feature_names = []
        for feat in features:
            if stepnum in feat:
                feature_names.append(list(feat[stepnum][recording_site].keys()))
        if len(feature_names) == 0:
            continue
        feature_names = list(set( chain(*feature_names) ))
        feature_names.sort()
        for name in feature_names:
            fid.write('%s,' % name)
            for feat in features:
                try:
                    values = feat[stepnum][recording_site][name]
                except:
                    values = [np.nan,np.nan]
                if args.no_std:
                    fid.write('%g,' % values[0])
                else:
                    fid.write('%g,%g,' % (values[0],values[1]))
            fid.write('\n')
        fid.close()


############################################################
###                      PICK_FILES                      ###
############################################################


def pick_files():
    import igor.binarywave as ibw

    def dump_file(infile):
        try:
            with open(infile, 'r') as fid:
                print('--- ' + infile + ' ' + '-'*(30 - 5 - len(infile)))
                for line in fid:
                    if line != '\n':
                        print(line.rstrip('\n'))
                print('-' * 30)
        except:
            pass

    parser = arg.ArgumentParser(description='Pick the files containing good sweeps.',\
                                prog=progname+' pick-files')
    parser.add_argument('folder', type=str, nargs='?', default='.',
                        help='the folder where the files are located (default: .)')
    parser.add_argument('-o', '--output', default='GOOD_SWEEPS',
                        help='output file name (default: GOOD_SWEEPS)')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of output file if it exists')
    parser.add_argument('-d', '--max-sweep-dur', default=np.inf, type=float,
                        help='maximum sweep duration, in seconds (default: inf)')
    parser.add_argument('-F', '--sampling-rate', default=20e3, type=float,
                        help='sampling rate, in Hz (default: 20000)')

    args = parser.parse_args(args=sys.argv[2:])

    folder = args.folder
    if not os.path.isdir(folder):
        print('pick-files: {}: no such folder.'.format(folder))
        sys.exit(1)

    max_sweep_dur = args.max_sweep_dur
    if max_sweep_dur <= 0:
        print('pick-files: maximum sweep duration must be > 0.')
        sys.exit(2)

    sampling_rate = args.sampling_rate
    if sampling_rate < 1000:
        print('pick-files: you gave a sampling rate < 1000 Hz: are you sure that is correct?')
        sys.exit(3)


    history = read_ibw_history_file(folder + '/DP_Sweeper/history.ibw')
    pulses = [True if name in ('StepPulse', 'BuiltinPulse') else False for name in history['builder_name']]

    dump_file(folder + '/' + args.output)

    files = glob.glob(folder + '/ad0*ibw')
    num = [int(os.path.splitext(f)[0].split('_')[1]) for f in files]
    files = [files[idx] for idx in np.argsort(num)]

    to_keep = []

    plt.ion()
    plt.figure()

    for f in files:
        sweep_index = int(os.path.splitext(f)[0].split('_')[1])
        try:
            idx = np.where((np.array(history['sweep_index']) == sweep_index) & pulses)[0][0]
        except:
            import ipdb
            ipdb.set_trace()
        mult = history['multiplier'][idx]
        pars = history['builder_parameters'][idx]
        amp = float(pars.split(';')[1].split('=')[1])
        data = ibw.load(f)
        voltage = data['wave']['wData']
        if len(voltage.shape) == 1:
            voltage = np.array([voltage])
        elif voltage.shape[0] > voltage.shape[1]:
            voltage = voltage.T
        time = np.arange(voltage.shape[1]) / sampling_rate
        if time[-1] > max_sweep_dur:
            continue
        plt.plot(time, voltage.T, 'k')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (mV)')
        plt.title('Sweep #{} I = {} pA'.format(sweep_index, amp*mult))
        plt.show()
        res = input('Keep {}? [y/N] '.format(f))
        if res.lower() in ('y','yes'):
            to_keep.append(sweep_index)
        plt.clf()

    to_keep.sort()

    outfile = folder + '/' + args.output
    if os.path.isfile(outfile) and not args.force:
        import datetime
        now = datetime.datetime.now()
        outfile += '-' + now.strftime('%Y%m%d%H%M%S')

    if np.max(np.diff(to_keep)) == 1:
        fid = open(outfile, 'w')
        fid.write('{}-{}\n'.format(to_keep[0], to_keep[-1]))
        fid.close()
    else:
        np.savetxt(outfile, to_keep, fmt='%d')

    dump_file(outfile)


############################################################
###                         HELP                         ###
############################################################


def help():
    if len(sys.argv) > 2 and sys.argv[2] in commands:
        cmd = sys.argv[2]
        sys.argv = [sys.argv[0], cmd, '-h']
        commands[cmd]()
    else:
        print('Usage: %s <command> [<args>]' % progname)
        print('')
        print('Available commands are:')
        print('   pick-files     Pick the files containing good sweeps.')
        print('   extract        Extract the features from a given cell.')
        print('   write          Write a configuration file using data from multiple cells stored in pkl files.')
        print('   write-xls      Write a configuration file using data from multiple cells stored in an Excel file.')
        print('   diff           Show differences between two feature files in a human way.')
        print('   dump           Dump feature files into several CSV files.')
        print('')
        print('Type \'%s help <command>\' for help about a specific command.' % progname)


############################################################
###                         MAIN                         ###
############################################################


# all the commands currently implemented
commands = {'help': help, 'extract': extract_features, 'write': write_features, 'write-xls': write_features_xls,
            'diff': diff_features, 'dump': dump_features, 'pick-files': pick_files}

def main():
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        commands['help']()
        sys.exit(0)
    if not sys.argv[1] in commands:
        print('%s: %s is not a recognized command. See \'%s --help\'.' % (progname,sys.argv[1],progname))
        sys.exit(1)
    commands[sys.argv[1]]()


if __name__ == '__main__':
    main()

