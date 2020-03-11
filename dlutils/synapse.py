
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
    def __init__(self, sec, x, E, weight, delay=1.):
        Synapse.__init__(self, sec, x, weight[0], delay)

        self.ampa_syn.Erev = E
        self.ampa_syn.kon = 139.87
        self.ampa_syn.koff = 4.05
        self.ampa_syn.CC = 54.54
        self.ampa_syn.CO = 10.85
        self.ampa_syn.Beta = 102.37
        self.ampa_syn.Alpha = 11.66

        self.nmda_syn.Erev = E
        self.nmda_syn.kon = 85.47
        self.nmda_syn.koff = 0.68
        self.nmda_syn.CC = 9.48
        self.nmda_syn.CO = 2.56
        self.nmda_syn.Beta = 0.72
        self.nmda_syn.Alpha = 0.078

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
    def __init__(self, sec, x, E, weight, delay=1.):
        Synapse.__init__(self, sec, x, weight, delay)

        self.gaba_a_syn.Erev = E
        self.gaba_a_syn.kon = 5.397
        self.gaba_a_syn.koff = 4.433
        self.gaba_a_syn.CC = 20.945
        self.gaba_a_syn.CO = 1.233
        self.gaba_a_syn.Beta = 283.09
        self.gaba_a_syn.Alpha = 254.52
        self.gaba_a_syn.gmax = 0.000603

    def make_synapse(self, sec, x):
        self.gaba_a_syn = h.GABA_A_KIN(sec(x))
        return self.gaba_a_syn


def build_cell_with_synapses(swc_file, parameters, mechanisms, replace_axon, add_axon_if_missing, distr_name, mu, sigma, scaling=1., slm_border=100.):
    """
    Builds a cell and inserts one synapse per segment. Each synapse is activated sequentially.
    """

    cell = cu.Cell('CA3_cell_%d' % int(np.random.uniform()*1e5), swc_file, parameters, mechanisms)
    cell.instantiate(replace_axon, add_axon_if_missing)

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
    synapses['basal'] = [AMPANMDASynapse(basal_segment['sec'], basal_segment['seg'].x, 0, [w,scaling*w]) \
                         for basal_segment,w in zip(cell.basal_segments,weights['basal'])]
    # one synapse in each apical segment that is within slm_border um from the tip of the apical dendrites
    y_coord = np.array([h.y3d(round(h.n3d(sec=segment['sec'])*segment['seg'].x),sec=segment['sec']) \
                        for segment in cell.apical_segments])
    max_y_coord = max(y_coord) - slm_border
    idx, = np.where(y_coord<max_y_coord)
    Nsyn['apical'] = len(idx)
    weights['apical'] = [x if x > 0 else 0 for x in rand_func(mu,sigma,size=Nsyn['apical'])]
    synapses['apical']  = [AMPANMDASynapse(cell.apical_segments[i]['sec'], cell.apical_segments[i]['seg'].x, 0, [w,scaling*w]) \
                           for i,w in zip(idx,weights['apical'])]

    return cell,synapses

