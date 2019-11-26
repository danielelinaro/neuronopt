
import json
import math
import numpy as np
from neuron import h

h.load_file('stdlib.hoc')
h.load_file('stdrun.hoc')

DEBUG = False

__all__ = ['compute_section_area', 'distance', 'Cell']

def compute_section_area(section):
    a = 0.
    for segment in section:
        a += h.area(segment.x, sec=section)
    return a

def distance(origin, end):
    # origin and end are two Segment objects
    h.distance(0, origin.x, sec=origin.sec)
    return h.distance(1, end.x, sec=end.sec)

def dst(x,y):
    return np.sqrt(np.sum((x-y)**2))

class Cell (object):
    @staticmethod
    def create_empty_template(template_name, seclist_names=None, secarray_names=None):
        '''create an hoc template named template_name for an empty cell'''
    
        objref_str = 'objref this, CellRef'
        newseclist_str = ''
    
        if seclist_names:
            for seclist_name in seclist_names:
                objref_str += ', %s' % seclist_name
                newseclist_str += \
                                  '             %s = new SectionList()\n' % seclist_name

        create_str = ''
        if secarray_names:
            create_str = 'create '
            create_str += ', '.join(
                '%s[1]' % secarray_name
                for secarray_name in secarray_names)
            create_str += '\n'

        template = '''\
        begintemplate %(template_name)s
        %(objref_str)s
        proc init() {\n%(newseclist_str)s
        forall delete_section()
        CellRef = this
        }

        gid = 0
        
        proc destroy() {localobj nil
        CellRef = nil
        }

        %(create_str)s
        endtemplate %(template_name)s
        ''' % dict(template_name=template_name, objref_str=objref_str,
                   newseclist_str=newseclist_str,
                   create_str=create_str)

        return template


    @staticmethod
    def set_nseg(morpho, use_dlambda_rule):
        if use_dlambda_rule:
            print('Setting the number of segments using the d_lambda rule.')
            for sec in morpho.all:
                sec.nseg = int((sec.L/(0.1*h.lambda_f(100,sec=sec))+0.9)/2)*2 + 1
                if DEBUG:
                    print('%s: length = %g um, nseg = %d.' % (sec.name(),sec.L,sec.nseg))
        else:
            print('Setting the number of segments using only section length.')
            for sec in morpho.all:
                sec.nseg = 1 + 2 * int(sec.L/40)
                if DEBUG:
                    print('%s: length = %g um, nseg = %d.' % (sec.name(),sec.L,sec.nseg))


    @staticmethod
    def replace_axon(morpho):
        """Replace axon"""

        nsec = len([sec for sec in morpho.axonal])

        if nsec == 0:
            ais_diams = [1, 1]
        elif nsec == 1:
            ais_diams = [morpho.axon[0].diam, morpho.axon[0].diam]
        else:
            ais_diams = [morpho.axon[0].diam, morpho.axon[0].diam]
            # Define origin of distance function
            h.distance(0, 0.5, sec=morpho.soma[0])

            for section in morpho.axonal:
                # If distance to soma is larger than 60, store diameter
                if h.distance(1, 0.5, sec=section) > 60:
                    ais_diams[1] = section.diam
                    break

        for section in morpho.axonal:
            h.delete_section(sec=section)

        # Create new axon array
        h.execute('create axon[2]', morpho)

        L = [30,30]
        for index, section in enumerate(morpho.axon):
            section.L = L[index]
            section.diam = ais_diams[index]
            section.nseg = 1 #+ 2 * int(L[index]/40)
            morpho.axonal.append(sec=section)
            morpho.all.append(sec=section)

        morpho.axon[0].connect(morpho.soma[0], 1.0, 0.0)
        morpho.axon[1].connect(morpho.axon[0], 1.0, 0.0)


    def __init__(self, cell_name, morpho_file, parameters, mechanisms):
        self.cell_name = cell_name
        self.morpho_file = morpho_file
        self.parameters = parameters
        self.mechanisms = mechanisms
        self.seclist_names = ['all', 'somatic', 'basal', 'apical', 'axonal', 'myelinated']
        self.secarray_names = ['soma', 'dend', 'apic', 'axon', 'myelin']


    def instantiate(self, replace_axon=False, add_axon_if_missing=True, use_dlambda_rule=False):
        self.template = Cell.create_empty_template(self.cell_name,self.seclist_names,self.secarray_names)
        h(self.template)
        self.template_function = getattr(h, self.cell_name)
        self.morpho = self.template_function()

        h.load_file('import3d.hoc')

        self.import3d = h.Import3d_SWC_read()
        self.import3d.input(self.morpho_file)
        self.gui = h.Import3d_GUI(self.import3d, 0)
        self.gui.instantiate(self.morpho)

        n_axonal_sec = len([sec for sec in self.morpho.axonal])
        if replace_axon:
            if n_axonal_sec == 0:
                print('The cell has no axon: adding an AIS stub.')
            else:
                print('Replacing existing axon with AIS stub.')
            Cell.replace_axon(self.morpho)
        elif add_axon_if_missing:
            if n_axonal_sec == 0:
                print('The cell has no axon: adding an AIS stub.')
                Cell.replace_axon(self.morpho)
            else:
                print('The cell has an axon: not replacing it with an AIS stub.')
        elif n_axonal_sec == 0:
            print('The cell has no axon: not replacing it with an AIS stub.')
        else:
            print('The cell has an axon: not replacing it with an AIS stub.')

        # this sets the number of segments in each section based only on length
        Cell.set_nseg(self.morpho, use_dlambda_rule=False)

        self.n_somatic_sections = len([sec for sec in self.morpho.somatic])
        self.n_axonal_sections = len([sec for sec in self.morpho.axonal])
        self.n_apical_sections = len([sec for sec in self.morpho.apical])
        self.n_basal_sections = len([sec for sec in self.morpho.basal])
        self.n_myelinated_sections = len([sec for sec in self.morpho.myelinated])
        self.n_sections = len([sec for sec in self.morpho.all])

        if self.n_axonal_sections > 0:
            self.has_axon = True
        else:
            self.has_axon = False

        self.biophysics(use_dlambda_rule)

        h.distance(0, 0.5, sec=self.morpho.soma[0])
        self.compute_total_area()
        self.compute_measures()
        self.compute_path_lengths()


    def biophysics(self, use_dlambda_rule):

        for reg,mechs in self.mechanisms.items():
            region = getattr(self.morpho,reg)
            for sec in region:
                for mech in mechs:
                    sec.insert(mech)

        if use_dlambda_rule:
            for param in self.parameters:
                if param['param_name'] in ['cm','Ra','e_pas','g_pas']:
                    region = getattr(self.morpho,param['sectionlist'])
                    for sec in region:
                        setattr(sec,param['param_name'],param['value'])
            Cell.set_nseg(self.morpho, use_dlambda_rule)
        
        for param in self.parameters:
            if param['type'] == 'global':
                setattr(h,param['param_name'],param['value'])
            elif param['type'] in ['section','range']:
                region = getattr(self.morpho,param['sectionlist'])
                if param['dist_type'] == 'uniform':
                    for sec in region:
                        setattr(sec,param['param_name'],param['value'])
                else:
                    h.distance(0, 0.5, sec=self.morpho.soma[0])
                    for sec in region:
                        for seg in sec:
                            dst = h.distance(1, seg.x, sec=sec)
                            g = eval(param['dist'].format(distance=dst,value=param['value']))
                            setattr(seg,param['param_name'],g)
            else:
                print('Unknown parameter type: %s.' % param['type'])


    def compute_total_area(self):
        self.total_area = 0
        for sec in self.morpho.all:
            for seg in sec:
                self.total_area += h.area(seg.x, sec=sec)
        if DEBUG:
            print('Total area: %.0f um^2.' % self.total_area)

    def find_section_with_point(self,pt):
        dst = []
        points = []
        names = []
        index = []
        for sec in self.morpho.all:
            for i in range(int(h.n3d(sec=sec))):
                points.append(np.array([h.x3d(i,sec=sec),h.y3d(i,sec=sec),h.z3d(i,sec=sec)]))
                dst.append(np.sqrt(np.sum((pt-points[-1])**2)))
                names.append(sec.name())
                index.append(i)
                if h.x3d(i,sec=sec) == pt[0] and \
                   h.y3d(i,sec=sec) == pt[1] and \
                   h.z3d(i,sec=sec) == pt[2]:
                    return sec
        idx = np.argsort(dst)
        dst = np.sort(dst)
        points = [points[i] for i in idx]
        names = [names[i] for i in idx]
        index = [index[i] for i in idx]
        with open('out.txt','w') as fid:
            for n,i,d,p in zip(names,index,dst,points):
                fid.write('[%s,%03d] %10.6f --- (%11.6f,%11.6f,%11.6f) <-> (%11.6f,%11.6f,%11.6f)\n' % (n,i,d,p[0],p[1],p[2],pt[0],pt[1],pt[2]))
        return None

    def distance_from_soma(self, seg):
        return distance(self.morpho.soma[0](0.5), seg)

    def compute_measures(self):
        self.total_nseg = 0.
        self.total_area = 0.
        self.all_segments = []
        self.somatic_segments = []
        self.apical_segments = []
        self.basal_segments = []
        if self.has_axon:
            self.axonal_segments = []
        for sec in self.morpho.all:
            self.total_nseg += sec.nseg
            n_points = sec.n3d()
            xyz = np.r_[np.array([sec.x3d(i) for i in range(n_points)], ndmin=2), \
                        np.array([sec.y3d(i) for i in range(n_points)], ndmin=2), \
                        np.array([sec.z3d(i) for i in range(n_points)], ndmin=2)]
            arc = np.array([sec.arc3d(i) for i in range(n_points)])
            for seg in sec:
                segment = {'seg': seg, 'sec': sec, \
                           'area': seg.area(), \
                           'dst': self.distance_from_soma(seg)}
                if len(arc) > 0:
                    idx = np.argmin(np.abs(arc - sec.L*seg.x))
                    segment['center'] =  xyz[:,idx]
                self.total_area += segment['area']
                self.all_segments.append(segment)
                if sec in self.morpho.soma:
                    self.somatic_segments.append(segment)
                elif sec in self.morpho.apic:
                    self.apical_segments.append(segment)
                elif sec in self.morpho.dend:
                    self.basal_segments.append(segment)
                elif sec in self.morpho.axon:
                    self.axonal_segments.append(segment)
        if DEBUG:
            print('Total area: %.0f um2.' % self.total_area)
            n_axonal_segments = len(self.axonal_segments) if self.has_axon else 0
            print('Total number of segments: %d (%d somatic, %d apical, %d basal and %d axonal.)' % \
                      (self.total_nseg,len(self.somatic_segments),len(self.apical_segments),\
                           len(self.basal_segments),n_axonal_segments))


    def compute_path_lengths(self):
        self.path_lengths = {self.morpho.soma[0].name(): np.array([0.0])}
        self.basal_path_lengths = []
        self.apical_path_lengths = []
        for sec in self.morpho.basal:
            self.basal_path_lengths.append(self.compute_path_lengths_from_parent(sec))
        for sec in self.morpho.apical:
            self.apical_path_lengths.append(self.compute_path_lengths_from_parent(sec))


    def compute_path_lengths_from_parent(self,sec):
        make_point = lambda sec,i: np.array([h.x3d(i,sec=sec), \
                                             h.y3d(i,sec=sec), \
                                             h.z3d(i,sec=sec)])
        key = sec.name()
        parent = h.SectionRef(sec=sec).parent
        if parent == self.morpho.soma[0]:
            ### assuming 3-point soma
            closest_parent_point = make_point(parent,1)
        else:
            n3d_parent = int(h.n3d(sec=parent))
            closest_parent_point = make_point(parent,n3d_parent-1)
        n3d = int(h.n3d(sec=sec))
        point_A = make_point(sec,0)
        self.path_lengths[key] = np.array([self.path_lengths[parent.name()][-1] + \
                                           dst(closest_parent_point,point_A)])
        for i in range(1,n3d):
            point_B = make_point(sec,i)
            self.path_lengths[key] = np.append(self.path_lengths[key],self.path_lengths[key][-1] + dst(point_A,point_B))
            point_A = point_B

        return self.path_lengths[key]

