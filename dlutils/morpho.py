

import numpy as np


__all__ = ['find_node_in_tree', 'compute_branch_id', 'find_terminal_branches',
           'find_oblique_branches', 'find_terminal_and_oblique_branches', 'plot_tree']


SWC_types = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4, 'soma_contour': 16}


def find_node_in_tree(ID, tree):
    ID = int(ID)
    for node in tree:
        if node.index == ID:
            return node
    return None


def compute_branch_id(tree):
    branch_id = 0
    for node in tree:
        if node.parent is None:
            node.content['branch_id'] = branch_id
        elif len(node.parent.children) > 1:
            branch_id += 1
            node.content['branch_id'] = branch_id
        else:
            node.content['branch_id'] = node.parent.content['branch_id']


def find_terminal_branches(tree, point_types = (SWC_types['basal'], SWC_types['apical'])):
    for node in tree:
        if not 'on_terminal_branch' in node.content:
            node.content['on_terminal_branch'] = False
        if node.parent is not None and node.content['p3d'].type in point_types and len(node.parent.children) > 1:
            child = node
            while len(child.children) == 1:
                child = child.children[0]
            if len(child.children) == 0 and child.content['p3d'].xyz[1] > 0:
                parent = child
                while parent != node:
                    parent.content['on_terminal_branch'] = True
                    parent = parent.parent
                node.content['on_terminal_branch'] = True


def find_oblique_branches(tree, end_point_limits=[0,240], max_angle=70):
    if not 'on_terminal_branch' in tree.root.content:
        find_terminal_branches(tree)
    for node in tree:
        if not node.content['on_terminal_branch']:
            node.content['on_oblique_branch'] = False
        elif len(node.parent.children) > 1:
            child = node
            while len(child.children) == 1:
                child = child.children[0]
            start_point = node.content['p3d'].xyz[:2]
            end_point = child.content['p3d'].xyz[:2]
            x,y = end_point - start_point
            angle = np.abs(np.rad2deg(np.arctan(y / x)))
            if node.content['p3d'].type == SWC_types['apical'] and \
               end_point[1] > end_point_limits[0] and \
               end_point[1] < end_point_limits[1] and \
               angle < max_angle:
                on_oblique = True
            else:
                on_oblique = False
            parent = child
            while parent is not None and parent != node:
                parent.content['on_oblique_branch'] = on_oblique
                parent = parent.parent
            node.content['on_oblique_branch'] = on_oblique


def find_terminal_and_oblique_branches(swc_file, terminal_point_types = (SWC_types['basal'], SWC_types['apical']),
                                       branch_types = ('terminal', 'oblique'),
                                       end_point_limits=[0,240], max_angle=70):
    import btmorph
    tree = btmorph.STree2()
    tree.read_SWC_tree_from_file(swc_file)
    compute_branch_id(tree)
    if 'oblique' in branch_types:
        if not SWC_types['apical'] in terminal_point_types:
            point_types = (SWC_types['apical'],) + terminal_point_types
        else:
            point_types = terminal_point_types
        find_terminal_branches(tree, point_types)
        find_oblique_branches(tree, end_point_limits, max_angle)
        if not 'terminal' in branch_types:
            # remove all terminal nodes
            for node in tree:
                node.content['on_terminal_branch'] = False
        elif not SWC_types['apical'] in terminal_point_types:
            for node in tree:
                if node.content['p3d'].type == SWC_types['apical']:
                    node.content['on_terminal_branch'] = False
    else:
        find_terminal_branches(tree, terminal_point_types)
    return tree


def plot_tree(tree, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _,ax = plt.subplots(1, 1)
    min_x, max_x = 0, 0
    for node in tree:
        if not node.parent is None and node.content['p3d'].type in (1,3,4):
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

