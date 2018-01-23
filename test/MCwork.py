import os, sys
import os.path
from gmsh_io import GmshIO
from operator import add
import numpy as np
from  correlated_field import SpatialCorrelatedField
import re
from unipath import Path

class FlowMC(object):

    def  __init__(self, yaml_file_dir, mesh_file_dir):
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        filename = os.path.join(fileDir, mesh_file_dir)
        gio = GmshIO()
        with open(filename) as f:
           gio.read(f)
        coord = np.zeros((len(gio.elements),3))
        for i, one_el in enumerate(gio.elements.values()):
            i_nodes = one_el[2]
            coord[i] = np.average(np.array([ gio.nodes[i_node] for i_node in i_nodes]), axis=0)
        self.points = coord
        self.yaml_dir = os.path.join(fileDir, yaml_file_dir)
        self.mesh_dir = os.path.join(fileDir, mesh_file_dir)
        self.gio      = gio
        
    def add_field(self, name, mu, sig2, corr):
        
        field = SpatialCorrelatedField(corr_exp = 'gauss', dim = self.points.shape[1], corr_length = corr,aniso_correlation = None,  )
        field.set_points(self.points, mu = mu, sigma = sig2)
        hodnoty = field.sample() 
        p = Path(self.mesh_dir)
        filepath = p.parent +'\\'+ name + '_values.msh'
        self.gio.write_fields(filepath, hodnoty, name) 
        print ("Field created in",filepath)
    
        
    def extract_value(self):
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
        p = Path(self.yaml_dir)
        os.chdir(p.parent)
        #os.chdir("C:\\Users\\Klara\\Documents\\Intec\\PythonScripts\\MonteCarlo\\Flow_02_test")
        os.system('call fterm.bat //opt/flow123d/bin/flow123d -s ' + str(yaml_file)) 
        #os.system("call fterm.bat //opt/flow123d/bin/flow123d -s 02_mysquare.yaml")
        #os.system('call fterm.bat //opt/flow123d/bin/flow123d -s 02_mysquare.yaml')
        #os.system("call fterm.bat //opt/flow123d/bin/flow123d -s" + str(self.yaml_dir) + '"')
        
