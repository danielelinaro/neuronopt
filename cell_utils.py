
import json
import math
import numpy as np
from neuron import h

h.load_file('stdlib.hoc')
h.load_file('stdrun.hoc')

DEBUG = False

def compute_section_area(section):
    a = 0.
    for segment in section:
        a += h.area(segment.x, sec=section)
    return a


def distance(origin, end, x=0.5):
    h.distance(sec=origin)
    return h.distance(x, sec=end)

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
    def set_nseg(morpho):
        for sec in morpho.all:
            sec.nseg = 1 + 2 * int(sec.L/40)
            #sec.nseg = int((sec.L/(0.1*h.lambda_f(100,sec=sec))+0.9)/2)*2 + 1
            #print('%s: length = %g um, nseg = %d.' % (sec.name(),sec.L,sec.nseg))


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
            h.distance(sec=morpho.soma[0])

            for section in morpho.axonal:
                # If distance to soma is larger than 60, store diameter
                if h.distance(0.5, sec=section) > 60:
                    ais_diams[1] = section.diam
                    break

        for section in morpho.axonal:
            h.delete_section(sec=section)

        # Create new axon array
        h.execute('create axon[2]', morpho)

        for index, section in enumerate(morpho.axon):
            section.nseg = 1
            section.L = 30
            section.diam = ais_diams[index]
            morpho.axonal.append(sec=section)
            morpho.all.append(sec=section)

        morpho.axon[0].connect(morpho.soma[0], 1.0, 0.0)
        morpho.axon[1].connect(morpho.axon[0], 1.0, 0.0)


    def __init__(self, cell_name, config_files, set_nseg=True, replace_axon=False):
        defaults = {'mechanisms': 'mechanisms.json','parameters': 'parameters.json'}
        self.cell_name = cell_name
        self.config_files = config_files
        for k,v in defaults.iteritems():
            if k not in self.config_files:
                self.config_files[k] = v
        self.seclist_names = ['all', 'somatic', 'basal', 'apical', 'axonal', 'myelinated']
        self.secarray_names = ['soma', 'dend', 'apic', 'axon', 'myelin']
        self.has_axon = True
        self.do_set_nseg = set_nseg
        self.do_replace_axon = replace_axon

    def instantiate(self):
        self.template = Cell.create_empty_template(self.cell_name,self.seclist_names,self.secarray_names)
        h(self.template)
        self.template_function = getattr(h, self.cell_name)
        self.morpho = self.template_function()

        h.load_file('import3d.hoc')

        self.import3d = h.Import3d_SWC_read()
        self.import3d.input(self.config_files['morphology'])
        self.gui = h.Import3d_GUI(self.import3d, 0)
        self.gui.instantiate(self.morpho)

        if self.do_replace_axon:
            Cell.replace_axon(self.morpho)

        if self.do_set_nseg:
            Cell.set_nseg(self.morpho)

        self.biophysics()

        h.distance(sec=self.morpho.soma[0])
        self.compute_total_area()
        self.compute_measures()
        self.compute_path_lengths()

    def biophysics(self):
        mechanisms = json.load(open(self.config_files['mechanisms'],'r'))
        for reg,mechs in mechanisms.iteritems():
            region = getattr(self.morpho,reg)
            for sec in region:
                for mech in mechs:
                    sec.insert(mech)

        parameters = json.load(open(self.config_files['parameters'],'r'))

        ### uncomment the following if we're not setting the number of
        ### segments based only on their length
        #if self.do_set_nseg:
        #    for param in parameters:
        #        if param['param_name'] in ['cm','Ra','e_pas','g_pas']:
        #            region = getattr(self.morpho,param['sectionlist'])
        #            for sec in region:
        #                setattr(sec,param['param_name'],param['value'])
        #Cell.set_nseg(self.morpho)
        
        for param in parameters:
            if param['type'] == 'global':
                setattr(h,param['param_name'],param['value'])
            elif param['type'] in ['section','range']:
                region = getattr(self.morpho,param['sectionlist'])
                if param['dist_type'] == 'uniform':
                    for sec in region:
                        setattr(sec,param['param_name'],param['value'])
                else:
                    h.distance(sec=self.morpho.soma[0])
                    for sec in region:
                        for seg in sec:
                            dst = h.distance(seg.x,sec=sec)
                            g = eval(param['dist'].format(distance=dst,value=param['value']))
                            setattr(sec,param['param_name'],g)
            else:
                print('Unknown parameter type: %s.' % param['type'])


    def compute_total_area(self):
        self.total_area = 0
        for sec in self.morpho.all:
            for seg in sec:
                self.total_area += h.area(seg.x, sec)
        if DEBUG:
            print('Total area: %.0f um^2.' % self.total_area)


    def distance_from_soma(self, sec, x=0.5):
        return distance(self.morpho.soma[0], sec, x)


    def compute_measures(self):
        # the areas of all sections in the soma
        self.soma_areas = []
        for sec in self.morpho.somatic:
            self.soma_areas.append(compute_section_area(sec))

        # the areas of all sections in the apical dendrite
        self.apical_areas = []
        # the distances of all apical sections from the soma: here the
        # distance is just the Euclidean one, not the path length
        self.apical_distances = []
        for sec in self.morpho.apical:
            self.apical_distances.append(self.distance_from_soma(sec))
            self.apical_areas.append(compute_section_area(sec))

        # the areas of all sections in the basal dendrites
        self.basal_areas = []
        # the distances of all basal sections from the soma
        self.basal_distances = []
        for sec in self.morpho.basal:
            self.basal_distances.append(self.distance_from_soma(sec))
            self.basal_areas.append(compute_section_area(sec))

        self.total_area = np.sum(self.soma_areas) + np.sum(self.apical_areas) + np.sum(self.basal_areas)

        if self.has_axon:
            # the areas of the sections in the axon
            self.axon_areas = []
            # the path length of each section in the axon from the root
            self.axon_lengths = []
            for sec in self.morpho.axonal:
                self.axon_areas.append(compute_section_area(sec))
                # here we are assuming that the axon extends vertically from the soma,
                # in which case distance and path length are the same. if this is not the
                # case, the subclass should override this method and implement its own code
                self.axon_lengths.append(self.distance_from_soma(sec))
            self.total_area += np.sum(self.axon_areas)


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
