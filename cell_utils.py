
from neuron import h
import json
import math

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
        self.do_set_nseg = set_nseg
        self.do_replace_axon = replace_axon


    def instantiate(self):
        self.template = Cell.create_empty_template(self.cell_name,self.seclist_names,self.secarray_names)
        h(self.template)
        self.template_function = getattr(h, self.cell_name)
        self.morpho = self.template_function()

        h.load_file('stdlib.hoc')
        h.load_file('stdrun.hoc')
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
