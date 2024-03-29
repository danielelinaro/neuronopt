import os
import sys
import argparse as arg
import numpy as np
from numpy.linalg import svd
import btmorph

# the name of this script
progname = os.path.basename(sys.argv[0])
# the standard SWC types
SWC_types = {'soma': 1, 'axon': 2, 'basal': 3, 'apical': 4, 'soma_contour': 16}

############################################################
###                       CONVERT                        ###
############################################################

def make_node(entry,parent):
    node = btmorph.SNode2(entry[0])
    p3d = btmorph.P3D2(np.array(entry[2:5]),entry[5],entry[1])
    node.set_content({'p3d': p3d})
    node.set_parent(parent)
    return node


def convert():
    parser = arg.ArgumentParser(description='Convert a morphology to the 3-point soma convention usable by NEURON',
                                prog=progname+' convert')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of existing output file')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')
    parser.add_argument('-c', '--coeff', default=1, type=float, help='coefficient to multiply coordinates and radius')
    parser.add_argument('--flip-y', action='store_true', help='flip Y coordinate')
    args = parser.parse_args(args=sys.argv[2:])
    
    f_in = args.swc_file
    if args.output is None:
        f_out = ''.join(f_in.split('.swc')[:-1]) + '.converted.swc'
    else:
        f_out = args.output

    if not os.path.isfile(f_in):
        print('%s: %s: no such file.' % (progname,f_in))
        sys.exit(0)

    if os.path.isfile(f_out) and not args.force:
        print('%s exists: use the -f option to force overwrite.' % f_out)
        sys.exit(0)

    entry_to_string = lambda e: '%d @ (%.3f,%.3f,%.3f), R=%.3f, type=%d, parent=%d' % \
                      (e[0],e[2],e[3],e[4],e[5],e[1],e[6])
    
    morpho_in = np.loadtxt(f_in)

    print('Fixing the morphology.')
    # we first change the type of those points that are of type soma but have parent
    # and children of a different type
    while True:
        done = True
        soma_idx, = np.where(morpho_in[:,1] == SWC_types['soma'])
        soma_ids = morpho_in[soma_idx,0]
        for entry in morpho_in[soma_idx,:]:
            if not entry[-1] in soma_ids and entry[-1] != -1:
                done = False
                parent_idx, = np.where(morpho_in[:,0] == entry[-1])
                print(entry_to_string(entry) + ' >> parent not in soma (type = %d).' % \
                      morpho_in[parent_idx,1])
                children_idx, = np.where(morpho_in[:,-1] == entry[0])
                print('%d children present' % len(children_idx))
                children_types = morpho_in[children_idx,1]
                if not np.any(children_types == SWC_types['soma']):
                    print('None of the children are of type soma: convert entry %d to type %d.' % \
                          (entry[0],morpho_in[parent_idx,1]))
                    morpho_in[morpho_in[:,0] == entry[0],1] = morpho_in[parent_idx,1]
                else:
                    print('At least one child of type soma: do not know what to do.')
                    import pdb; pdb.set_trace()
        if done:
            break

    # center and rotate the morphology
    print('Centering and rotating the morphology.')
    idx, = np.where(morpho_in[:,1] != SWC_types['axon'])
    xyz = morpho_in[:,2:5]
    xyz_centered = xyz - np.mean(xyz[idx,:],axis=0)
    U,S,V = svd(xyz_centered[idx,:])
    xyz_rot = np.dot(xyz_centered,np.transpose(V))
    xyz_rot = xyz_rot[:,(1,0,2)]
    morpho_in[:,2:5] = xyz_rot

    # convert to three-point soma representation
    print('Converting to three-point soma.')
    soma_idx, = np.where(morpho_in[:,1] == SWC_types['soma'])
    if soma_idx.size == 0:
        soma_contour = True
        soma_idx, = np.where(morpho_in[:,1] == SWC_types['soma_contour'])
        soma_type = SWC_types['soma_contour']
        print('The contour of the soma contains %d points.' % soma_idx.size)
    else:
        soma_contour = False
        soma_type = SWC_types['soma']
    soma_ids = morpho_in[soma_idx,0]
    center = np.mean(morpho_in[soma_idx,2:5],axis=0)
    print('The center of the soma is at (%.3f,%.3f,%.3f).' % (center[0],center[1],center[2]))
    morpho_in[:,2:5] -= center
    radius = np.mean(np.sqrt(np.sum(morpho_in[soma_idx,2:5]**2,axis=1)))

    # three-point soma representation
    morpho_out = [[1, SWC_types['soma'], 0, 0, 0, radius, -1],
                  [2, SWC_types['soma'], 0, -radius, 0, radius, 1],
                  [3, SWC_types['soma'], 0, radius, 0, radius, 1]]

    for entry in morpho_in:
        if entry[1] != soma_type:
            # do not add entries that are part of the soma in the original morphology
            if entry[-1] in soma_ids or (soma_contour and entry[-1] == -1):
                if soma_contour and entry[-1] == -1:
                    print(entry_to_string(entry))
                # if an entry's parent belonged to the soma, attach it to the root
                entry[-1] = 1
            morpho_out.append(entry)

    # convert to a numpy array for easier handling
    morpho_out = np.array(morpho_out)

    # create the empty tree
    tree = btmorph.STree2()

    # a dictionary of nodes indexed by their id
    nodes = {-1: None}

    # build the tree
    print('Building the tree to sort the entries.')
    for entry in morpho_out:
        parent = nodes[entry[-1]]
        node = make_node(entry,parent)
        nodes[node.index] = node
        if parent is None:
            tree.set_root(node)
        else:
            parent.add_child(node)
    
    # save the tree to file
    print('Saving the tree to file %s.' % f_out)
    fid = open(f_out,'w')
    for index,node in enumerate(tree):
        node.index = index+1
        xyz = node.content['p3d'].xyz
        if node.parent is None:
            index = -1
        else:
            index = node.parent.index
        y_coeff = -1 if args.flip_y else 1
        fid.write('%d %d %g %g %g %g %d\n' % (node.index,node.content['p3d'].type,
                                              xyz[0]*args.coeff, xyz[1]*args.coeff*y_coeff, xyz[2]*args.coeff,
                                              node.content['p3d'].radius*args.coeff, index))
    fid.close()

############################################################
###                        BUILD                         ###
############################################################

def build():
    from dlutils import cell as cu
    from dlutils import utils
    from neuron import h
    import json
    
    parser = arg.ArgumentParser(description='Use a converted morphology to build a cell in NEURON',
                                prog=progname+' build')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-p', '--params-file', default=None, type=str, help='parameters file')
    parser.add_argument('-m', '--mech-file', default=None, type=str, help='mechanisms file')
    parser.add_argument('-c', '--config-file', default=None, type=str, help='configuration file')
    parser.add_argument('-n', '--cell-name', default=None, type=str, help='cell name (required only with --config-file)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    args = parser.parse_args(args=sys.argv[2:])

    if not os.path.isfile(args.swc_file):
        print('%s: %s: no such file.' % (progname,args.swc_file))
        sys.exit(1)

    if args.params_file is not None:
        if not os.path.isfile(args.params_file):
            print('%s: %s: no such file.' % (progname,args.params_file))
            sys.exit(1)
        parameters = json.load(open(args.params_file,'r'))
    else:
        parameters = []

    if args.mech_file is not None:
        if not os.path.isfile(args.mech_file):
            print('%s: %s: no such file.' % (progname,args.mech_file))
            sys.exit(1)
        mechanisms = json.load(open(args.mech_file,'r'))
    elif args.config_file is not None:
        if not os.path.isfile(args.config_file):
            print('%s: %s: no such file.' % (progname,args.config_file))
            sys.exit(1)
        if args.cell_name is None:
            print('--cell-name must be present with --config-file option.')
            sys.exit(1)
        mechanisms = utils.extract_mechanisms(args.config_file, args.cell_name)
    else:
        mechanisms = []

    cell = cu.Cell('MyCell', args.swc_file, parameters, mechanisms)
    cell.instantiate()

    regions = ['soma','basal','apical']
    if len(cell.morpho.axon) > 0:
        # does the morphology have an axon?
        regions.append('axon')
    nsections = {r: 0 for r in regions}
    npoints = {r: 0 for r in regions}
    lengths = {r: 0 for r in regions}
    areas = {r: 0 for r in regions}
    for reg in regions:
        region = getattr(cell.morpho,reg)
        for sec in region:
            nsections[reg] += 1
            npoints[reg] += h.n3d(sec=sec)
            area = 0
            for seg in sec:
                area += h.area(seg.x,sec=sec)
            if args.verbose:
                print(sec.name())
                for i in range(int(h.n3d(sec=sec))):
                    pt = np.array([h.x3d(i,sec=sec),h.y3d(i,sec=sec),h.z3d(i,sec=sec)])
                    print('   [{:4d}] ({:7.1f},{:7.1f},{:7.1f})'.format(i+1,pt[0],pt[1],pt[2]))
                print('      L = {:6.1f} um.'.format(sec.L))
                print('   diam = {:6.1f} um.'.format(sec.diam))
                print('   area = {:6.1f} um2.'.format(area))
            lengths[reg] += sec.L
            areas[reg] += area

    print('')
    for reg in regions:
        print('-----------------------------------------')
        print('[%s]' % reg.upper())
        print('        number of sections: %d' % nsections[reg])
        print('          number of points: %d' % npoints[reg])
        print('    total dendritic length: %.0f um' % lengths[reg])
        print('        total surface area: %.0f um2' % areas[reg])

    print('=========================================\n')
    print(' Total number of sections: %d' % np.sum([v for v in nsections.values()]))
    print('   Total number of points: %d' % np.sum([v for v in npoints.values()]))
    print('   Total dendritic length: %.0f um' % np.sum([v for v in lengths.values()]))
    print('       Total surface area: %.0f um2' % np.sum([v for v in areas.values()]))


############################################################
###                       SIMPLIFY                       ###
############################################################


def copy_node(node_in, index=None):
    if index is None:
        index = node_in.index
    node_out = btmorph.SNode2(index)
    p3d_in = node_in.content['p3d']
    p3d_out = btmorph.P3D2(p3d_in.xyz, p3d_in.radius, p3d_in.type)
    node_out.set_content({'p3d': p3d_out})
    return node_out


def parse_branch(node, parent, max_branch_length=np.inf):
    branch = [node]
    if node.content['p3d'].type != SWC_types['soma']:
        branch_length = 0
        while len(node.children) == 1 and branch_length < max_branch_length:
            node = node.children[0]
            branch.append(node)
            branch_length += np.sqrt(np.sum((branch[-2].content['p3d'].xyz-branch[-1].content['p3d'].xyz)**2))
    else:
        xyz = branch[-1].content['p3d'].xyz
        print('Node (%f,%f,%f) belongs to soma.' % (xyz[0],xyz[1],xyz[2]))
    end_point = copy_node(branch[-1])
    end_point.set_parent(parent)
    parent.add_child(end_point)
    for child in node.children:
        parse_branch(child, end_point, max_branch_length)


def plot_tree(tree,marker='-',color=None):
    cmap = {idx: col for idx,col in zip(SWC_types.values(),'krbg')}
    for node in tree:
        if not node.parent is None:
            x = [node.content['p3d'].xyz[0],node.parent.content['p3d'].xyz[0]]
            y = [node.content['p3d'].xyz[1],node.parent.content['p3d'].xyz[1]]
            lw = node.content['p3d'].radius
            if color is None:
                c = cmap[node.content['p3d'].type]
            else:
                c = color
            plt.plot(x,y,c+marker,linewidth=lw)


def simplify():
    parser = arg.ArgumentParser(description='Simplify a converted morphology stored in an SWC file.',
                                prog=progname+' simplify')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-l', '--max-length', default=40, type=int, help='maximum path length for section')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')
    args = parser.parse_args(args=sys.argv[2:])

    if not os.path.isfile(args.swc_file):
        print('%s: no such file.' % args.swc_file)
        sys.exit(1)

    if args.output is None:
        output_file = args.swc_file.split('.swc')[0] + '.simplified.%d_um.swc' % args.max_length
        
    # the original tree
    full_tree = btmorph.STree2()
    full_tree.read_SWC_tree_from_file(args.swc_file,types=list(range(10)))
    print('There are %d nodes in the full representation of the morphology.' % len(full_tree.get_nodes()))

    # the simplified tree
    tree = btmorph.STree2()

    # the two trees have the same root
    tree.set_root(copy_node(full_tree.root))

    # start the recursion with the children of the root
    for node in full_tree.root.children:
        parse_branch(node,tree.root,args.max_length)

    print('There are %d nodes in the simplified representation of the morphology.' % len(tree.get_nodes()))

    fid = open(output_file,'w')
    for index,node in enumerate(tree):
        node.index = index+1
        xyz = node.content['p3d'].xyz
        if node.parent is None:
            index = -1
        else:
            index = node.parent.index
        fid.write('%d %d %f %f %f %f %d\n' % (node.index,node.content['p3d'].type,
                                              xyz[0], xyz[1], xyz[2],
                                              node.content['p3d'].radius, index))
    fid.close()
    #tree.write_SWC_tree_to_file(output_file)



############################################################
###                         PLOT                         ###
############################################################


def plot():
    parser = arg.ArgumentParser(description='Plot a converted morphology',
                                prog=progname+' plot')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')
    args = parser.parse_args(args=sys.argv[2:])
    
    f_in = args.swc_file
    if args.output is None:
        f_out = f_in.split('.swc')[0] + '.pdf'
    else:
        f_out = args.output

    if not os.path.isfile(f_in):
        print('%s: %s: no such file.' % (progname,f_in))
        sys.exit(0)

    import neurom.viewer
    import matplotlib.pyplot as plt
    neurom.viewer.draw(neurom.load_neuron(f_in))
    plt.show()



############################################################
###                        CLEAN                         ###
############################################################


def clean():
    parser = arg.ArgumentParser(description='Clean a morphology',
                                prog=progname+' clean')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-o', '--output', default=None, type=str, help='output file name')
    parser.add_argument('-n', default=None, type=int, help='minimum number of points to keep a branch')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of existing output file')
    args = parser.parse_args(args=sys.argv[2:])

    f_in = args.swc_file
    if args.output is None:
        f_out = f_in.split('.swc')[0] + '_min_branch_length_%d.swc' % args.n
    else:
        f_out = args.output

    if not os.path.isfile(f_in):
        print('%s: %s: no such file.' % (progname,f_in))
        sys.exit(0)

    if args.n is None:
        print('You must specify the minimum number of points necessary for a branch to be kept (-n option).')
        sys.exit(0)

    if os.path.isfile(f_out) and not args.force:
        print('%s exists: use the -f option to force overwrite.' % f_out)
        sys.exit(0)

    tree = btmorph.STree2()
    tree.read_SWC_tree_from_file(f_in)

    def remove_short_branches(node,branch_start,length,min_length):
        if not node.parent is None and len(node.parent.children) > 1:
            branch_start = node
            length = 0
        length += 1
        removed = 0
        if len(node.children) == 0:
            if length < min_length:
                if branch_start.content['p3d'].type != 1:
                    tree.remove_node(branch_start)
                    removed = 1
        else:
            for child in node.children:
                removed += remove_short_branches(child,branch_start,length,min_length)
        return removed

    total_num_removed = 0
    cnt = 0
    while True:
        num_removed = remove_short_branches(tree.root, tree.root, 0, args.n)
        total_num_removed += num_removed
        cnt += 1
        print('[%02d] number of branches removed: %d.' % (cnt,num_removed))
        if num_removed == 0:
            break
    print('>>> total number of branches with length <= %d points removed: %d.' % (args.n,total_num_removed))

    fid = open(f_out,'w')
    for index,node in enumerate(tree):
        node.index = index+1
        xyz = node.content['p3d'].xyz
        if node.parent is None:
            index = -1
        else:
            index = node.parent.index
        fid.write('%d %d %g %g %g %g %d\n' % (node.index,node.content['p3d'].type,
                                              xyz[0], xyz[1], xyz[2],
                                              node.content['p3d'].radius, index))
    fid.close()

    sys.exit(0)
    
    import matplotlib.pyplot as plt
    plt.figure()
    for node in tree:
        if node.parent is None:
            continue
        x = [node.parent.content['p3d'].xyz[0],node.content['p3d'].xyz[0]]
        y = [node.parent.content['p3d'].xyz[1],node.content['p3d'].xyz[1]]
        plt.plot(x,y,'k.-')
        if len(node.children) == 0:
            plt.plot(x[1],y[1],'ro')
        elif len(node.children) > 1:
            plt.plot(x[1],y[1],'yo')
        #if node.index in to_remove:
        #    if len(node.children) == 0:
        #        plt.plot(x[1],y[1],'ro')
        #    else:
        #        plt.plot(x[1],y[1],'gx')
    plt.show()



############################################################
###                      ANALYSE                         ###
############################################################


def analyse():
    import btmorph

    def find_bifurcation(node):
        if len(node.children) == 1:
            return find_bifurcation(node.children[0])
        return node

    def find_leaves(node):
        tree = btmorph.STree2()
        tree.set_root(node)
        leaves = []
        for node in tree:
            if len(node.children) == 0:
                leaves.append(node)
        return leaves

    def find_major_bifurcation(node, min_y_coord):
        while True:
            bif = find_bifurcation(node)
            if len(bif.children) == 0:
                # there are no major bifurcations on the tree originating from this  node
                return None
            max_y_coords = []
            for child in bif.children:
                terminals = find_leaves(child)
                y_coords = [node.content['p3d'].xyz[1] for node in terminals]
                max_y_coords.append(np.max(y_coords))
                #print(max_y_coords[-1])
            #print('-----')
            max_y_coords = np.array(max_y_coords)
            if np.all(max_y_coords > min_y_coord):
                break
            idx, = np.where(max_y_coords > min_y_coord)
            if len(idx) > 1:
                print('There is more than one branch that reaches past 900 um.')
                import ipdb
                ipdb.set_trace()
            node = bif.children[idx[0]]
        return bif

    dst = lambda x,y: np.sqrt(np.sum((x-y)**2))

    parser = arg.ArgumentParser(description='Analyse a morphology',
                                prog=progname+' analyse')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-c', '--min-y-coord', default=900, type=float, \
                        help='minimum y coordinate of a branch to be considered a main one (default 900 um)')
    args = parser.parse_args(args=sys.argv[2:])

    swc_file = args.swc_file
    if not os.path.isfile(swc_file):
        print('%s: %s: no such file.' % (progname,swc_file))
        sys.exit(1)

    morpho = np.loadtxt(swc_file)
    xyz = morpho[:,2:5]
    x_lim = np.array([np.min(xyz[:,0]), np.max(xyz[:,0])])
    y_lim = np.array([np.min(xyz[:,1]), np.max(xyz[:,1])])

    min_y_coord = args.min_y_coord
    if min_y_coord > y_lim[1]:
        print('Maximum y coordinate of cell {} is smaller than {} um.'.format(swc_file, min_y_coord))
        sys.exit(2)

    tree = btmorph.STree2()
    tree.read_SWC_tree_from_file(swc_file)

    root_coord = tree.root.content['p3d'].xyz

    APICAL_TYPE = 4
    apical_root = None
    for child in tree.root.children:
        if child.content['p3d'].type == APICAL_TYPE:
            apical_root = child
            xyz = child.content['p3d'].xyz
            print('Found first apical point @ ({:.2f},{:.2f},{:.2f}) um.'.format(xyz[0], xyz[1], xyz[2]))
            break
    if apical_root is None:
        print('No apical dendrite connected to the soma: searching the whole tree...')
        for node in tree:
            if node.content['p3d'].type == APICAL_TYPE:
                apical_root = node
                xyz = node.content['p3d'].xyz
                print('Found first apical point @ ({:.2f},{:.2f},{:.2f}) um.'.format(xyz[0], xyz[1], xyz[2]))
                break
    if apical_root is None:
        print('The morphology {} does not contain an apical dendrite (SWC type {}).'.format(swc_file, APICAL_TYPE))
        sys.exit(0)

    first_apic_bif = find_major_bifurcation(apical_root, min_y_coord)
    if first_apic_bif is None:
        print('There is no bifurcation on the main apical dendrite of morphology {}.'.format(swc_file))
        sys.exit(0)
    first_apic_bif_coord = first_apic_bif.content['p3d'].xyz
    first_apic_bif_dst = dst(root_coord, first_apic_bif_coord)
    print('The first bifurcation on the main apical dendrite is located @ ({:.2f},{:.2f},{:.2f}) um.'\
          .format(first_apic_bif_coord[0], first_apic_bif_coord[1], first_apic_bif_coord[2]))

    bifurcations = [find_major_bifurcation(child, min_y_coord) for child in first_apic_bif.children]
    second_apic_bif = []
    for bif,s in zip(bifurcations, ['first','second']):
        if bif is None:
            print('There is no main bifurcation on the {} secondary apical dendrite of morphology {}'.format(s,swc_file))
        else:
            second_apic_bif.append(bif)
            print('The first bifurcation on the {} secondary apical dendrite is located @ ({:.2f},{:.2f},{:.2f}) um.'\
                  .format(s, bif.content['p3d'].xyz[0], bif.content['p3d'].xyz[1], bif.content['p3d'].xyz[2]))
    second_apic_bif_coord = [node.content['p3d'].xyz for node in second_apic_bif]
    second_apic_bif_dst = [dst(root_coord, coord) for coord in second_apic_bif_coord]

    idx = np.argmin(second_apic_bif_dst)

    print('The main bifurcation is located at a distance of {:.1f} um.'.format(first_apic_bif_dst))
    print('The primary tuft ends at a distance of {:.1f} um.'.format(second_apic_bif_dst[idx]))

    import neurom.viewer
    import matplotlib.pyplot as plt
    neurom.viewer.draw(neurom.load_neuron(swc_file))
    x = np.linspace( -np.min([np.abs(x_lim[0]), first_apic_bif_dst]),
                     np.min([x_lim[1], first_apic_bif_dst]), 100)
    y = np.sqrt(first_apic_bif_dst**2 - x**2)
    plt.plot(x, y, 'r--')
    x = np.linspace( -np.min([np.abs(x_lim[0]), second_apic_bif_dst[idx]]),
                     np.min([x_lim[1], second_apic_bif_dst[idx]]), 100)
    y = np.sqrt(second_apic_bif_dst[idx]**2 - x**2)
    plt.plot(x, y, 'r--')
    plt.show()



############################################################
###                    TAG-BRANCHES                      ###
############################################################


def tag_branches():
    import matplotlib.pyplot as plt
    from dlutils.morpho import find_terminal_and_oblique_branches, plot_tree
    from dlutils.cell import Cell

    parser = arg.ArgumentParser(description='Tag oblique and terminal branches of a morphology',
                                prog=progname+' tag-branches')
    parser.add_argument('swc_file', type=str, action='store', help='SWC file')
    parser.add_argument('-e', '--min-end-point', default=0, type=float, \
                        help='minimum y coordinate of the end point of a branch to be considered oblique')
    parser.add_argument('-E', '--max-end-point', default=240, type=float, \
                        help='maximum y coordinate of the end point of a branch to be considered oblique')
    parser.add_argument('-a', '--max-angle', default=70, type=float, \
                        help='maximum angle for a branch to be considered oblique')
    parser.add_argument('-A', '--apical-only', action='store_true', help='Tag only points on terminal branches on the apical dendrite')
    parser.add_argument('-B', '--basal-only', action='store_true', help='Tag only points on terminal branches on the basal dendrite')
    parser.add_argument('-T', '--terminal-only', action='store_true', help='Tag only points on terminal branches')
    parser.add_argument('-O', '--oblique-only', action='store_true', help='Tag only points on oblique branches')
    parser.add_argument('-P', '--plot', action='store_true', help='Plot the tagged morphology')
    
    args = parser.parse_args(args=sys.argv[2:])

    swc_file = args.swc_file
    if not os.path.isfile(swc_file):
        print('{}: {}: no such file.'.format(progname, swc_file))
        return

    if args.apical_only and args.basal_only:
        print('Options --apical-only and --basal-only are mutually exclusive.')
        return

    if args.terminal_only and args.oblique_only:
        print('Options --terminal-only and --oblique-only are mutually exclusive.')
        return

    if args.basal_only:
        point_types = SWC_types['basal'],
    elif args.apical_only:
        point_types = SWC_types['apical'],
    else:
        point_types = SWC_types['basal'], SWC_types['apical']

    if args.terminal_only:
        branch_types = 'terminal',
    elif args.oblique_only:
        branch_types = 'oblique',
        point_types = SWC_types['apical'],
    else:
        branch_types = ('terminal', 'oblique')

    if args.oblique_only and args.basal_only:
        print('There are no oblique branches on the basal dendrite.')
        return
    
    if args.oblique_only and args.apical_only:
        print('Option --apical-only is redundant when --oblique-only is specified.')

    tree = find_terminal_and_oblique_branches(swc_file, point_types, branch_types,
                                              end_point_limits = [args.min_end_point, args.max_end_point],
                                              max_angle = args.max_angle)

    tree_list = [node for node in tree]
    tree_coords = np.array([node.content['p3d'].xyz for node in tree])
    
    cell = Cell('cell', swc_file, parameters=[], mechanisms=[])
    cell.instantiate(replace_axon = False,
                     add_axon_if_missing = False,
                     use_dlambda_rule = False,
                     force_passive = True,
                     TTX = False)

    print('Number of  basal compartments: {:3d}.'.format(len(cell.basal_segments)))
    print('Number of apical compartments: {:3d}.'.format(len(cell.apical_segments)))

    folder,filename = os.path.split(swc_file)
    if folder == '':
        folder = '.'
    suffix = os.path.splitext(filename)[0]

    if SWC_types['basal'] in point_types:
        on_terminal_branches = []
        for i,seg in enumerate(cell.basal_segments):
            idx = np.sum((tree_coords - seg['center']) ** 2, axis=1).argmin()
            node = tree_list[idx]
            if node.content['on_terminal_branch']:
                on_terminal_branches.append(i)
        with open(folder + '/' + suffix + '_basal_terminal.txt', 'w') as fid:
            for i in on_terminal_branches:
                fid.write('{:3d}\n'.format(i))
        print('{} out of {} basal compartments are on terminal branches.'.format(len(on_terminal_branches), len(cell.basal_segments)))

    if SWC_types['apical'] in point_types:
        on_terminal_branches = []
        on_oblique_branches = []
        for i,seg in enumerate(cell.apical_segments):
            idx = np.sum((tree_coords - seg['center']) ** 2, axis=1).argmin()
            node = tree_list[idx]
            if 'on_oblique_branch' in node.content and node.content['on_oblique_branch']:
                on_oblique_branches.append(i)
            elif 'on_terminal_branch' in node.content and node.content['on_terminal_branch']:
                on_terminal_branches.append(i)
        if 'terminal' in branch_types:
            with open(folder + '/' + suffix + '_apical_terminal.txt', 'w') as fid:
                for i in on_terminal_branches:
                    fid.write('{:3d}\n'.format(i))
            print('{} out of {} apical compartments are on terminal branches.'.format(len(on_terminal_branches), len(cell.apical_segments)))
        if 'oblique' in branch_types:
            with open(folder + '/' + suffix + '_apical_oblique.txt', 'w') as fid:
                for i in on_oblique_branches:
                    fid.write('{:3d}\n'.format(i))
            print('{} out of {} apical compartments are on oblique branches.'.format(len(on_oblique_branches), len(cell.apical_segments)))

    if args.plot:
        fig,ax = plt.subplots(1, 1, figsize=(3,5))
        plot_tree(tree, ax)
        plt.savefig(folder + '/' + suffix + '.pdf')
        plt.show()



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
        print('   convert        Convert a morphology to the 3-point soma convention usable by NEURON')
        print('   build          Use a converted morphology to build a cell in NEURON and obtain some statistics')
        print('   plot           Plot a converted morphology file')
        print('   simplify       Simplify a converted morphology')
        print('   tag-branches   Tag oblique and terminal branches in a morphology')
        print('   clean          Remove branches below a certain length')
        print('')
        print('Type \'%s help <command>\' for help about a specific command.' % progname)



############################################################
###                         MAIN                         ###
############################################################


# all the commands currently implemented
commands = {'help': help, 'convert': convert, 'build': build, 'plot': plot,
            'simplify': simplify, 'clean': clean, 'analyse': analyse,
            'tag-branches': tag_branches}

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


