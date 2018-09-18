
from neuron import h

class Synapse (object):
    def __init__(self, sec, x, weight, delay=1.):
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


