
import numpy as np
from scipy.interpolate import NearestNDInterpolator

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from matplotlib import cm

from .morpho import Tree

__all__ = ['set_rc_defaults', 'remove_border', 'make_axes', 'plot_means_with_errorbars', 'plot_tree']

def set_rc_defaults():
    plt.rc('font', family='Arial', size=10)
    plt.rc('lines', linewidth=1, color='k')
    plt.rc('axes', linewidth=1, titlesize='medium', labelsize='medium')
    plt.rc('xtick', direction='out')
    plt.rc('ytick', direction='out')
    #plt.rc('figure', dpi=300)


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecessary plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()


def make_axes(r, c, i, offset=[0.1,0.1], spacing=[0.1,0.1], border=[0.05,0.05]):
    """
    A better subplot.
    """
    if np.isscalar(offset):
        offset = [offset,offset]
    if np.isscalar(spacing):
        spacing = [spacing,spacing]
    if np.isscalar(border):
        border = [border,border]
    w = (1 - offset[0] - spacing[0]*(c-1) - border[0])/c
    h = (1 - offset[1] - spacing[1]*(r-1) - border[1])/r
    i = i-1
    x = i%c
    y = r - 1 - int(i/c)
    ax = plt.axes([offset[0] + (w+spacing[0])*x, 
                   offset[1] + (h+spacing[1])*y,
                   w, h])
    return ax


def plot_means_with_errorbars(x, y, mode='sem', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    Ym = np.nanmean(y,axis=0)
    if mode == 'sem':
        Ys = np.nanstd(y,axis=0) / np.sqrt(y.shape[0])
    else:
        Ys = np.nanstd(y,axis=0)
    try:
        lbl = kwargs.pop('label')
    except:
        lbl = None
    for i,ym,ys in zip(x,Ym,Ys):
        ax.plot([i,i], [ym-ys,ym+ys], **kwargs)
    if lbl is not None:
        ax.plot(x, Ym, 'o-', label=lbl, **kwargs)
    else:
        ax.plot(x, Ym, 'o-', **kwargs)


def _plot_tree_fast(tree, type_ids=(1,2,3,4), scalebar_length=None, cmap=None, points=None, values=None,
                    cbar_levels=None, cbar_ticks=10, cbar_orientation='vertical', cbar_label='', ax=None,
                    bounds=None):
    
    if ax is None:
        ax = plt.gca()

    if points is None or values is None:
        uniform_color_branches = True
        if cmap is None:
            color_fun = lambda i: [
                [0,0,0],    # soma
                [.2,.2,.2], # axon
                [.7,0,.7],  # basal
                [0,.7,0]    # apical
            ][i-1]
        elif isinstance(cmap, dict):
            color_fun = lambda key: cmap[key]
        else:
            color_fun = cmap
    else:
        uniform_color_branches = False
        interp = NearestNDInterpolator(points, values)
        norm = colors.Normalize(vmin = values.min(), vmax = values.max())

    for branch in tree.branches:
        if branch[0].type not in type_ids:
            continue
        if branch[0].parent is not None:
            node = branch[0].parent
            xyzd = np.concatenate((np.array([node.x, node.y, node.z, node.diam], ndmin=2),
                                [[node.x, node.y, node.z, node.diam] for node in branch]))
        else:
            xyzd = np.array([[node.x, node.y, node.z, node.diam] for node in branch])

        if branch[0].parent is not None and branch[0].parent.type == 1:
            xyzd[0,-1] = xyzd[1,-1]

        xy = xyzd[:,:2].reshape(-1, 1, 2)
        segments = np.concatenate([xy[:-1], xy[1:]], axis=1)
        if uniform_color_branches:
            lc = LineCollection(segments, linewidths=xyzd[:,-1]/2, colors=color_fun(branch[0].type))
        else:
            lc = LineCollection(segments, linewidths=xyzd[:,-1]/2, cmap=cmap, norm=norm)
            lc.set_array(interp(xyzd[:,:3]))
        line = ax.add_collection(lc)

    if bounds is not None:
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
    else:
        ax.set_xlim(tree.bounds[0])
        ax.set_ylim(tree.bounds[1])
        ax.axis('equal')

    if scalebar_length is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x = xlim[0] / 1.5
        y = (ylim[1] - scalebar_length) / 2
        ax.plot(x + np.zeros(2), y + np.array([0, scalebar_length]), 'k', lw=2)
        ax.text(x - np.diff(xlim)/15, y + scalebar_length/2, r'{} $\mu$m'.format(scalebar_length), fontsize=12, \
                horizontalalignment='center', verticalalignment='center', rotation=90)

    if not uniform_color_branches and cbar_levels is not None:
        if np.isscalar(cbar_ticks):
            ticks = np.round(np.linspace(values.min(), values.max(), cbar_ticks))
            levels = np.linspace(ticks[0], ticks[-1], cbar_levels)
        else:
            ticks = cbar_ticks
            levels = np.linspace(values.min(), values.max(), cbar_levels)
        cbar = plt.colorbar(line, ax=ax, fraction=0.1, shrink=0.5, aspect=30, ticks=ticks, orientation=cbar_orientation)
        if cbar_orientation == 'vertical':
            cbar.ax.set_ylabel(cbar_label)
        else:
            cbar.ax.set_xlabel(cbar_label)



def _plot_tree_btmorph(tree, type_ids=(1,2,3,4), ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _,ax = plt.subplots(1, 1)
    min_x, max_x = 0, 0
    for node in tree:
        if not node.parent is None and node.content['p3d'].type in type_ids:
            if node.content['p3d'].type == 1 and node.parent.content['p3d'].type == 1:
                continue
            parent_xy = node.parent.content['p3d'].xyz[:2]
            xy = node.content['p3d'].xyz[:2]
            if xy[0] >  max_x:
                max_x = xy[0]
            if xy[0] < min_x:
                min_x = xy[0]
            r = node.content['p3d'].radius
            if 'on_oblique_branch' in node.content and node.content['on_oblique_branch']:
                col = 'g'
            elif 'on_terminal_branch' in node.content and node.content['on_terminal_branch']:
                col = 'm'
            else:
                col = 'k'
            ax.plot([parent_xy[0], xy[0]], [parent_xy[1], xy[1]], color=col, linewidth=r)
    width = max_x - min_x
    dx = 100
    ax.plot(max_x - width / 10 + np.zeros(2), 50 + np.array([0,dx]), 'k', lw=1)
    ax.text(max_x - width / 6.5, 50 + dx/2, r'{} $\mu$m'.format(dx), horizontalalignment='center', \
            verticalalignment='center', rotation=90)
    ax.axis('equal')



def plot_tree(tree, type_ids=(1,2,3,4), scalebar_length=None, cmap=None, points=None, values=None,
              cbar_levels=None, cbar_ticks=10, cbar_orientation='vertical', cbar_label='', ax=None, bounds=None):
    if isinstance(tree, Tree):
        _plot_tree_fast(tree, type_ids, scalebar_length, cmap, points, values,
                        cbar_levels, cbar_ticks, cbar_orientation, cbar_label, ax, bounds)
    else:
        _plot_tree_btmorph(tree, type_ids, ax)

