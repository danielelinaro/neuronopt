
import os
import sys
import glob
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import btmorph

matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': 'Arial', 'size': 8})
matplotlib.rc('axes', **{'linewidth': 0.7})

def plot_tree(tree, ax=None):
    if ax is None:
        _,ax = plt.subplots(1, 1)
    for node in tree:
        if not node.parent is None and node.content['p3d'].type in (1,3,4):
            if node.content['p3d'].type == 1 and node.parent.content['p3d'].type == 1:
                continue
            parent_xy = node.parent.content['p3d'].xyz[:2]
            xy = node.content['p3d'].xyz[:2]
            r = node.content['p3d'].radius
            try:
                if node.content['on_oblique_branch']:
                    col = 'g'
                elif node.content['on_terminal_branch']:
                    col = 'm'
                else:
                    col = 'k'
            except:
                col = 'k'
            ax.plot([parent_xy[0], xy[0]], [parent_xy[1], xy[1]], color=col, linewidth=r)
    ax.axis('equal')


if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == 'thorny':
        thorny = True
    else:
        thorny = False

    passive = False
    with_TTX = True

    pkl_file_pattern = 'synaptic_cooperativity'
    if thorny:
        pkl_file_pattern += '_thorny_*'
        example_ind = 1
    else:
        pkl_file_pattern += '_a-thorny_*'
        example_ind = 0
    if passive:
        pkl_file_pattern += '_passive'
    else:
        pkl_file_pattern += '_active'
        if with_TTX:
            pkl_file_pattern += '_with_TTX'
    pkl_file_pattern += '.pkl'

    pkl_files = glob.glob(pkl_file_pattern)
    idx = np.argsort([int(pkl_file.split('_')[3]) for pkl_file in pkl_files])
    pkl_files = [pkl_files[i] for i in idx]
    print('Using the following data files:')
    for pkl_file in pkl_files:
        print('   ' + pkl_file)
    all_data = [pickle.load(open(pkl_file, 'rb')) for pkl_file in pkl_files]
    data = all_data[example_ind]

    fig = plt.figure(constrained_layout=True, figsize=(9,6))
    gs = fig.add_gridspec(3, 3)
    ax_morpho = fig.add_subplot(gs[:2,0])
    ax_AR = fig.add_subplot(gs[2,0])
    ax = [[fig.add_subplot(gs[i,j]) for j in (1,2)] for i in range(3)]

    swc_file = '/Users/daniele/Postdoc/Research/Janelia/01_model_optimization/' + \
               data['config']['cell_type'].capitalize() + '/' + \
               data['config']['cell_name'] + '/' + data['config']['optimization_run'] + '/' + \
               data['config']['swc_file']
    tree = btmorph.STree2()
    tree.read_SWC_tree_from_file(swc_file)
    plot_tree(tree, ax_morpho)
    ax_morpho.plot(data['spine_centers'][:,0], data['spine_centers'][:,1], 'ro', markersize=2, \
                   markerfacecolor='r', markeredgewidth=0.2)
    xlim = ax_morpho.get_xlim()
    ylim = ax_morpho.get_ylim()
    dx = np.diff(xlim)
    dy = np.diff(ylim)
    ax_morpho.plot(xlim[1] - dx/100 + np.zeros(2), np.array([0,100]), 'k', linewidth=1)
    ax_morpho.text(xlim[1] - dx/20, 50, r'$100\,\mu\mathrm{m}$', rotation=90, \
                   horizontalalignment='center', verticalalignment='center')

    t = data['t']
    Vspine = data['Vspine']
    Vdend = data['Vdend']
    idx = (t > 990) & (t < 1200)
    ax_AR.plot(t[idx] - 1000, Vspine[0,idx], 'k', label='Spine', lw=1)
    ax_AR.plot(t[idx] - 1000, Vdend[0,idx], 'r', label='Dendrite', lw=1)
    ax_AR.legend(loc='best')
    ax_AR.set_xlabel('Time from input (ms)')
    ax_AR.set_ylabel('Vm (mV)')
    ylim = ax_AR.get_ylim()
    dy = np.diff(ylim)
    x = 200
    y = ylim[0] + dy/5
    ax_AR.text(x, y+dy/5, r'$\mathrm{R}_{\mathrm{dend}} = %.0f \mathrm{M}\Omega$' % data['R']['dend'], \
               horizontalalignment='right')
    ax_AR.text(x, y+dy/10, r'$\mathrm{R}_{\mathrm{neck}} = %.0f \mathrm{M}\Omega$' % data['R']['neck'], \
               horizontalalignment='right')
    ax_AR.text(x, y, 'AR = {:.2f}'.format(data['AR']), horizontalalignment='right')
    spike_times = data['spike_times']
    n_spines = len(spike_times)

    Vsoma = data['Vsoma']
    gnmda = data['gnmda']
    MgBlock = data['MgBlock']
    V_soma_pks = data['V_soma_pks']
    V_dend_pks = data['V_dend_pks']
    G_NMDA_pks = data['G_NMDA_pks']

    ms = 4
    cmap = plt.get_cmap('viridis', n_spines)
    window = [10, 100]
    for i,spk in enumerate(spike_times[0][:n_spines]):
        idx = (t > spk - window[0]) & (t < spk + window[1])
        ax[0][0].plot(t[idx] - spk, Vdend[0,idx], color=cmap(i), linewidth=1)
        ax[1][0].plot(t[idx] - spk, Vsoma[idx], color=cmap(i), linewidth=1)
        ax[2][0].plot(t[idx] - spk, gnmda[0,idx] * MgBlock[0,idx], color=cmap(i), linewidth=1)
    for i,spk in enumerate(spike_times[0][:n_spines]):
        ax[0][0].plot(t[V_dend_pks[i]] - spk, Vdend[0,V_dend_pks[i]], 'ro', \
                      markersize=ms, markerfacecolor='w', markeredgewidth=1)
        ax[1][0].plot(t[V_soma_pks[i]] - spk, Vsoma[V_soma_pks[i]], 'ro', \
                      markersize=ms, markerfacecolor='w', markeredgewidth=1)
        ax[2][0].plot(t[G_NMDA_pks[i]] - spk, gnmda[0,G_NMDA_pks[i]] * MgBlock[0,G_NMDA_pks[i]], 'ro', \
                      markersize=ms, markerfacecolor='w', markeredgewidth=1)
    ax[2][0].set_xlabel('Time from input (ms)')
    ax[0][0].set_ylabel('Dendritic voltage (mV)')
    ax[1][0].set_ylabel('Somatic voltage (mV)')
    ax[2][0].set_ylabel('NMDA conductance (nS)');

    dV_dend = np.array([data['dV_dend'] for data in all_data])
    dV_soma = np.array([data['dV_soma'] for data in all_data])
    dG_AMPA = np.array([data['dG_AMPA'] for data in all_data])
    dG_NMDA = np.array([data['dG_NMDA'] for data in all_data])
    n = 1 + np.arange(n_spines)

    def plot_mean_sem(x, y, ax, color, marker, lbl, **kwargs):
        y_mean = y.mean(axis=0)
        y_sem = y.std(axis=0) / np.sqrt(y.shape[0])
        for a,m,s in zip(x, y_mean, y_sem):
            ax.plot(a + np.zeros(2), m + s * np.array([-1,1]), color, **kwargs)
        ax.plot(x, y_mean, color + marker + '-', label=lbl, **kwargs)

    ax[0][1].plot(n, n * dV_dend.mean(axis=0)[0], 'rs-', lw=1, markerfacecolor='w', markersize=ms, label='Linear prediction')
    plot_mean_sem(n, dV_dend, ax[0][1], 'k', 'o', 'Measured', lw=1, markerfacecolor='w', markersize=ms)

    ax[1][1].plot(n, n * dV_soma.mean(axis=0)[0], 'rs-', lw=1, markerfacecolor='w', markersize=ms)
    plot_mean_sem(n, dV_soma, ax[1][1], 'k', 'o', None, lw=1, markerfacecolor='w', markersize=ms)
    
    ax[2][1].plot(n, dG_AMPA[0], 'bs-', lw=1, markerfacecolor='w', markersize=ms, label='AMPA')
    plot_mean_sem(n, dG_NMDA, ax[2][1], 'k', 'o', 'NMDA', lw=1, markerfacecolor='w', markersize=ms)
    
    ax[2][1].set_xlabel('Input number')
    ax[0][1].set_ylabel('Dendritic EPSP (mV)')
    ax[1][1].set_ylabel('Somatic EPSP (mV)')
    ax[2][1].set_ylabel('Conductance (nS)')
    ax[0][1].legend(loc='lower right')
    ax[2][1].legend(loc='best')
    ax[0][1].set_ylim([0, np.max([dV_dend.mean(axis=0)[-1], dV_dend.mean(axis=0)[0]*n_spines]) * 1.1])
    ax[1][1].set_ylim([0, np.max([dV_soma.mean(axis=0)[-1], dV_soma.mean(axis=0)[0]*n_spines]) * 1.1])
    ax[2][1].set_ylim([dG_NMDA.mean(axis=0)[0] - 0.1, dG_NMDA.mean(axis=0)[-1] + 0.1]);
    for i in range(3):
        ax[i][1].set_xticks(n)

    for side in ('right','top'):
        for i in range(3):
            for j in range(2):
                ax[i][j].spines[side].set_visible(False)
        ax_AR.spines[side].set_visible(False)
    ax_morpho.axis('off')

    plt.savefig(os.path.splitext(pkl_file_pattern.replace('_*',''))[0] + '.pdf')
