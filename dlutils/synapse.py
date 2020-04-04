
import numpy as np
from neuron import h
from . import cell as cu

__all__ = ['Synapse', 'AMPANMDASynapse', 'GABAASynapse', 'build_cell_with_synapses']

class Synapse (object):
    def __init__(self, sec, x, weight, delay=1.):
        self.seg = sec(x)
        self.syn = self.make_synapse(sec, x)
        self.stim = h.VecStim()
        self.nc = h.NetCon(self.stim, self.syn)
        self.nc.weight[0] = weight
        self.nc.delay = delay

    def make_synapse(self, sec, x):
        raise NotImplementedError()

    def set_presynaptic_spike_times(self, spike_times):
        self.spike_times = h.Vector(spike_times)
        self.stim.play(self.spike_times)

    def get_presynaptic_spike_times(self):
        return np.array(self.spike_times)


class AMPANMDASynapse (Synapse):
    def __init__(self, sec, x, E, weight, delay=1., **kwargs):
        Synapse.__init__(self, sec, x, weight[0], delay)

        self.ampa_syn.Erev = E
        self.ampa_syn.kon = kwargs['AMPA']['kon'] if 'AMPA' in kwargs and 'kon' in kwargs['AMPA'] else 12.88
        self.ampa_syn.koff = kwargs['AMPA']['koff'] if 'AMPA' in kwargs and 'koff' in kwargs['AMPA'] else 6.47
        self.ampa_syn.CC = kwargs['AMPA']['CC'] if 'AMPA' in kwargs and 'CC' in kwargs['AMPA'] else 69.97
        self.ampa_syn.CO = kwargs['AMPA']['CO'] if 'AMPA' in kwargs and 'CO' in kwargs['AMPA'] else 6.16
        self.ampa_syn.Beta = kwargs['AMPA']['Beta'] if 'AMPA' in kwargs and 'Beta' in kwargs['AMPA'] else 100.63
        self.ampa_syn.Alpha = kwargs['AMPA']['Alpha'] if 'AMPA' in kwargs and 'Alpha' in kwargs['AMPA'] else 173.04

        self.nmda_syn.Erev = E
        self.nmda_syn.kon = kwargs['NMDA']['kon'] if 'NMDA' in kwargs and 'kon' in kwargs['NMDA'] else 86.69
        self.nmda_syn.koff = kwargs['NMDA']['koff'] if 'NMDA' in kwargs and 'koff' in kwargs['NMDA'] else 0.69
        self.nmda_syn.CC = kwargs['NMDA']['CC'] if 'NMDA' in kwargs and 'CC' in kwargs['NMDA'] else 9.64
        self.nmda_syn.CO = kwargs['NMDA']['CO'] if 'NMDA' in kwargs and 'CO' in kwargs['NMDA'] else 2.6
        self.nmda_syn.Beta = kwargs['NMDA']['Beta'] if 'NMDA' in kwargs and 'Beta' in kwargs['NMDA'] else 0.68
        self.nmda_syn.Alpha = kwargs['NMDA']['Alpha'] if 'NMDA' in kwargs and 'Alpha' in kwargs['NMDA'] else 0.079

        self.syn = [self.ampa_syn, self.nmda_syn]
        self.ampa_nc = self.nc
        self.nmda_nc = h.NetCon(self.stim, self.nmda_syn)
        self.nmda_nc.weight[0] = weight[1]
        self.nc = [self.ampa_nc, self.nmda_nc]

    def make_synapse(self, sec, x):
        self.ampa_syn = h.AMPA_KIN(sec(x))
        self.nmda_syn = h.NMDA_KIN2(sec(x))
        return self.ampa_syn


class GABAASynapse (Synapse):
    def __init__(self, sec, x, E, weight, delay=1., **kwargs):
        Synapse.__init__(self, sec, x, weight, delay)

        self.gaba_a_syn.Erev = E
        self.gaba_a_syn.kon = kwargs['kon'] if 'kon' in kwargs else 5.397
        self.gaba_a_syn.koff = kwargs['koff'] if 'koff' in kwargs else 4.433
        self.gaba_a_syn.CC = kwargs['CC'] if 'CC' in kwargs else 20.945
        self.gaba_a_syn.CO = kwargs['CO'] if 'CO' in kwargs else 1.233
        self.gaba_a_syn.Beta = kwargs['Beta'] if 'Beta' in kwargs else 283.09
        self.gaba_a_syn.Alpha = kwargs['Alpha'] if 'Alpha' in kwargs else 254.52
        self.gaba_a_syn.gmax = 0.000603

    def make_synapse(self, sec, x):
        self.gaba_a_syn = h.GABA_A_KIN(sec(x))
        return self.gaba_a_syn


def build_cell_with_synapses(swc_file, cell_parameters, mechanisms, replace_axon, add_axon_if_missing, passive_cell, \
                             synapse_parameters, distr_name, mu, sigma, scaling=1., slm_border=100.):
    """
    Builds a cell and inserts one synapse per segment. Each synapse is activated sequentially.
    """

    cell = cu.Cell('CA3_cell_%d' % int(np.random.uniform()*1e5), swc_file, cell_parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing, force_passive=passive_cell)

    if distr_name == 'normal':
        rand_func = np.random.normal
    elif distr_name == 'lognormal':
        rand_func = np.random.lognormal
        mu = np.log(mu)
    else:
        raise Exception('Unknown distribution [%s]' % distr_name)

    # one synapse in each basal segment
    Nsyn = {'basal': len(cell.basal_segments)}
    weights = {'basal': [x if x > 0 else 0 for x in rand_func(mu,sigma,size=Nsyn['basal'])]}
    synapses = {}
    synapses['basal'] = [AMPANMDASynapse(basal_segment['sec'], basal_segment['seg'].x, 0, [w,scaling*w], **synapse_parameters) \
                         for basal_segment,w in zip(cell.basal_segments,weights['basal'])]
    # one synapse in each apical segment that is within slm_border um from the tip of the apical dendrites
    y_coord = np.array([h.y3d(round(h.n3d(sec=segment['sec'])*segment['seg'].x),sec=segment['sec']) \
                        for segment in cell.apical_segments])
    max_y_coord = max(y_coord) - slm_border
    idx, = np.where(y_coord<max_y_coord)
    Nsyn['apical'] = len(idx)
    weights['apical'] = [x if x > 0 else 0 for x in rand_func(mu,sigma,size=Nsyn['apical'])]
    synapses['apical']  = [AMPANMDASynapse(cell.apical_segments[i]['sec'], cell.apical_segments[i]['seg'].x, 0, [w,scaling*w], **synapse_parameters) \
                           for i,w in zip(idx,weights['apical'])]

    return cell,synapses

