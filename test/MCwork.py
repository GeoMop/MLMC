import os, sys
import os.path
from gmsh_io import GmshIO
from operator import add
import numpy as np
from  correlated_field import SpatialCorrelatedField
import re
from unipath import Path
import numpy as np
import scipy as sp
import scipy.stats
import texttable as tt


def confidence_interval(data, confidence = 0.95):
    s = 1.0*np.array(data)
    m, se = np.mean(s), scipy.stats.sem(s) #standard error 
    h = se*sp.stats.t._ppf((1+confidence)/2,len(s)-1)
    return (m-h,m,m+h)
    
def mc_stats(data):
    # so far just mean and variance estimates
    c_int = confidence_interval(data,0.95)
    tab = tt.Texttable()
    headings = ['Estimates of:','mean','var', 'conf. bounds 0.95']
    tab.header(headings)
    tab.add_row(['---',data.mean(),data.var(), np.round_(c_int,3)])
    s = tab.draw()
    print(s)        
    

class FlowMC(object):
    '''
    Based on the two input files; for grid (xy.msh) and Flow (xy.yaml) it enables to
    create a field of given property for the cell centers for the given mesh. It 
    can execute Flow within .Flow_run and (so far) can extract a singel value from 
    mass_balance.txt created in the output folder by Flow.
    '''

    def  __init__(self, yaml_file_dir, mesh_file_dir):
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        filename = os.path.join(fileDir, mesh_file_dir)
        gio      = GmshIO()
        with open(filename) as f:
           gio.read(f)
        coord = np.zeros((len(gio.elements),3))
        for i, one_el in enumerate(gio.elements.values()):
            i_nodes  = one_el[2]
            coord[i] = np.average(np.array([ gio.nodes[i_node] for i_node in i_nodes]), axis=0)
        self.points   = coord
        self.yaml_dir = os.path.join(fileDir, yaml_file_dir)
        self.mesh_dir = os.path.join(fileDir, mesh_file_dir)
        self.gio      = gio
        
    def add_field(self, name, mu, sig2, corr):
        '''
        Creates a random spatially variable field based on given mu, sig and corr length
        Stored in name_values.msh.
        '''       
        field = SpatialCorrelatedField(corr_exp = 'gauss', dim = self.points.shape[1], corr_length = corr,aniso_correlation = None,  )
        field.set_points(self.points, mu = mu, sigma = sig2)
        hodnoty  = field.sample() 
        p        = Path(self.mesh_dir)
        filepath = p.parent +'\\'+ name + '_values.msh'
        self.gio.write_fields(filepath, hodnoty, name) 
        print ("Field created in",filepath)
    
        
    def extract_value(self):
        # Extracts a single a value from a text file in output, so far ...
        p = Path(self.mesh_dir)
        filename = os.path.join(p.parent,'output\\mass_balance.txt')
        soubor   = open(filename,'r')
        for line in soubor:
            line = line.rstrip()
            if re.search('1', line):
                x = line
        
        y         = x.split('"conc"',100)  
        z         = y[1].split('\t')
        var_name  = -float(z[3])       
                     
        return var_name
        
    def Flow_run(self,yaml_file):    
        p = Path(self.mesh_dir)
        os.chdir(p.parent)
        os.system('call fterm.bat //opt/flow123d/bin/flow123d -s ' + str(yaml_file)) 
        
    

        
