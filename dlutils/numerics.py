
import numpy as np
from scipy.optimize import least_squares

__all__ = ['double_exp', 'fit_epsp']


def double_exp(t1, t2, d, t):
    """
    out = double_exp(t1, t2, d, t)
    t1 [s] : rising time constant
    t2 [s] : decaying time constant
    d  [s] : delay
    t  [s] : time axis

    out --> double exponential, with max amplitude == 1,
    regardless of the combination {t1, t2}..
    """
    # Let's consider the function f(t) = (exp(-t/t1)-exp(-t/t2))/(t1-t2)
    # I compute the time 'tt' at which the first derivative of f(t) is zero
    tt = (t1 * t2 / (t2 - t1)) * np.log(t2 / t1);          

    # I then compute f(tt)
    aa = (np.exp(- tt / t1) - np.exp(- tt / t2)) / (t1 - t2);

    # I appropriately prepare a normalization prefactor.. (so it has unitary
    # peak amplitude).
    A  = 1. / (aa * (t1 - t2));

    # Then, the actual function as an output
    return A * (np.exp(- (t - d) / t1) - np.exp(- (t - d) / t2) ) * (t >= d)


def fit_epsp(t, V, t0, duration, slope_window=2.5, ax=None):

    #idx, = np.where((t > t0 - 50e-3) & (t < t0 - 10e-3))
    idx, = np.where((t > t0 + duration - 50e-3) & (t < t0 + duration))
    baseline = np.mean(V[:, idx], axis=1)
    idx, = np.where((t > t0) & (t < t0 + duration))

    # start EPSP time at zero
    t_EPSP = t[idx] - t[idx[0]]

    # the EPSPs
    EPSPs = V[:,idx]
    n_EPSPs,n_samples = EPSPs.shape

    # set the baseline to (approximately) zero
    EPSPs = EPSPs - np.tile(baseline, (1, n_samples))

    # peak amplitudes
    amplitudes = np.max(EPSPs, 1)
    idx_max = np.argmax(EPSPs, 1)
    # normalise the amplitude
    EPSPs /= np.tile(amplitudes, (1, n_samples))

    tau_rise = np.zeros(n_EPSPs)
    tau_decay = np.zeros(n_EPSPs)
    slope = np.zeros(n_EPSPs)
    delay = np.zeros(n_EPSPs)

    fun = lambda x, voltage, time: voltage - double_exp(x[0], x[1], x[2], time)

    for k in range(n_EPSPs):
        # initial conditions for the rise time, the
        # decay time, and the axonal propagation delay
        EPSP_pars  = np.array([t_EPSP[idx_max[k]] / 5, \
                               (duration - t_EPSP[idx_max[k]]) / 10, \
                               t_EPSP[idx_max[k]] / 10])

        res = least_squares(fun, \
                            EPSP_pars, \
                            bounds=((0,0,0), (np.inf,np.inf,np.inf)), \
                            args=(EPSPs[k,:], t_EPSP))

        x0 = np.mean(EPSPs[k,t_EPSP < res['x'][2]])
        if x0 < 0.05:
            EPSPs[k,:] = (EPSPs[k,:] - x0) / (1 + np.abs(x0))
            EPSP_pars = res['x']
            res = least_squares(fun, \
                                EPSP_pars, \
                                bounds=((0,0,0), (np.inf,np.inf,np.inf)), \
                                args=(EPSPs[k,:], t_EPSP))
            amplitudes[k] *= (1 + np.abs(x0))

        tau_rise[k] = np.min(res['x'][:2])
        tau_decay[k] = np.max(res['x'][:2])
        delay[k] = res['x'][2]
        idx, = np.where((t > t0 + delay[k]) & (t < t0 + delay[k] + slope_window * 1e-3))
        p = np.polyfit(t[idx], V[k,idx], 1) # [mV/s]
        slope[k] = p[0]

        if ax is not None:
            ax.plot(t_EPSP, EPSPs[k,:], 'k')
            ax.plot(t_EPSP, double_exp(res['x'][0], res['x'][1], res['x'][2], t_EPSP), 'r')
        
    return tau_rise, tau_decay, amplitudes, slope, delay


