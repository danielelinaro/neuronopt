

import numpy as np


__all__ = ['find_node_in_tree', 'compute_branch_id', 'find_terminal_branches',
           'find_oblique_branches', 'find_terminal_and_oblique_branches',
           'Node', 'Tree']


SWC_types = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4, # classical types
             'AIS': 6, 'AIS_K': 7, 'axonmyelin': 8, 'axonnodes': 9,
             'basal_dend': 10, 'pf_targets': 11, 'aa_targets': 12, 'sodium_dend': 13,
             'soma_contour': 16}


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


class Node (object):
    def __init__(self, x, y, z, diam, node_type, node_id):
        self._x = x
        self._y = y
        self._z = z
        self._xyz = np.array([x,y,z])
        self._diam = diam
        self._node_type = node_type
        self._node_id = node_id
        self._parent = None
        self._children = []

    @property
    def id(self):
        return self._id
    
    @property
    def type(self):
        return self._node_type
    
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, value):
        self._x = value
        self._xyz[0] = value

    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, value):
        self._y = value
        self._xyz[1] = value

    @property
    def z(self):
        return self._z
    @z.setter
    def z(self, value):
        self._z = value
        self._xyz[2] = value
        
    @property
    def diam(self):
        return self._diam
    @diam.setter
    def diam(self, value):
        if diam <= 0:
            raise Exception('Diameter must be > 0')
        self._diam = value

    @property
    def xyz(self):
        return self._xyz
    @xyz.setter
    def xyz(self, value):
        self._xyz = value
        self._x, self._y, self._z = value
        
    @property
    def children(self):
        return self._children

    def add_to_children(self, node):
        if not node in self._children:
            self._children.append(node)
            node.parent = self
            
    def remove_from_children(self, node):
        idx = self._children.index(node)
        self._children.pop(idx)

    @property
    def parent(self):
        return self._parent
    @parent.setter
    def parent(self, value):
        if self._parent == value:
            return
        old_parent = self._parent
        if old_parent is not None:
            old_parent.remove_from_children(self)
        self._parent = value
        value.add_to_children(self)


class Tree (object):
    def __init__(self, swc_file, swc_types='all'):
        from collections import OrderedDict
        data = np.loadtxt(swc_file)
        jdx = np.array([2,3,4])
        if swc_types != 'all'  and isinstance(swc_types, (list,set)):
            idx = np.sum([X==y for y in Y], axis=0).astype(bool)
        else:
            idx = np.arange(data.shape[0])
        IDX,JDX = np.meshgrid(idx,jdx)
        x,y,z = data[IDX,JDX]
        x_max,x_min = x.max(),x.min()
        y_max,y_min = y.max(),y.min()
        z_max,z_min = z.max(),z.min()
        self.ratio = {
            'xy': (x_max - x_min) / (y_max - y_min),
            'xz': (x_max - x_min) / (z_max - z_min),
            'yz': (y_max - y_min) / (z_max - z_min)
        }
        self.xy_ratio = self.ratio['xy']
        self.xz_ratio = self.ratio['xz']
        self.yz_ratio = self.ratio['yz']
        self.bounds = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
        nodes = OrderedDict()
        for row in data:
            node_id   = int(row[0])
            node_type = int(row[1])
            x, y, z,  = row[2:5]
            diam      = row[5]
            parent_id = int(row[6])
            nodes[node_id] = Node(x, y, z, diam, node_type, node_id)
            if parent_id > 0:
                nodes[node_id].parent = nodes[parent_id]
        _,self._root = nodes.popitem(last=False)
        self.branches = []
        self._make_branches(self.root, self.branches)
        
    @property
    def root(self):
        return self._root

    def _gather_nodes(self, node, node_list):
        if not node is None:
            node_list.append(node)
            for child in node.children :
                self._gather_nodes(child, node_list)

    def __iter__(self):
        nodes = []
        self._gather_nodes(self.root, nodes)
        for n in nodes:
            yield n

    def _make_branches(self, node, branches):
        branch = []
        while len(node.children) == 1:
            branch.append(node)
            node = node.children[0]
        branch.append(node)
        branches.append(branch)
        for child in node.children:
            self._make_branches(child, branches)



