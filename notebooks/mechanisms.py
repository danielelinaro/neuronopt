
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['TEMPERATURE', 'R', 'FARADAY', 'trap0', 'plot_var']

TEMPERATURE = 34
R = 8.313424
FARADAY = 96.48533212

def trap0(v, th, a, q):
    y = a * q * np.ones(v.shape)
    idx, = np.where(np.abs(v - th) > 1e-6)
    y[idx] = a * (v[idx] - th) / (1 - np.exp(-(v[idx] - th) / q))
    return y


def plot_var(xinf, taux, ax, x_range=[-150,101], col='k', label=''):
    n = 1013
    if x_range[0] < 0 or np.diff( np.log10(x_range) ) < 3:
        x = np.linspace(x_range[0], x_range[1], n)
        var_name = 'Voltage (mV)'
        mode = 'lin'
    else:
        x = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), n)
        var_name = 'Calcium (mM)'
        mode = 'log'
    if mode == 'lin':
        ax[0].plot(x, xinf(x), color=col, lw=2)
    else:
        ax[0].semilogx(x, xinf(x), color=col, lw=2)
    ax[0].set_ylabel(r'$x_\infty$')
    if mode == 'lin':
        ax[1].plot(x, taux(x), color=col, lw=2, label=label)
    else:
        ax[1].semilogx(x, taux(x), color=col, lw=2, label=label)
    if label != '':
        ax[1].legend(loc='best')
    ax[1].set_xlabel(var_name)
    ax[1].set_ylabel(r'$\tau$ (ms)')

