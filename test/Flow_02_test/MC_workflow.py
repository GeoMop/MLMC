import os, sys
libdir = os.path.join(os.path.split(
         os.path.dirname(os.path.realpath(__file__)))[0],"C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\src\\mlmc")
sys.path.insert(1,libdir)

from gmsh_io import GmshIO
from operator import add
import numpy as np
from  correlated_field import SpatialCorrelatedField
import matplotlib.pyplot as plt
from test_correlated_field import Cumul
import re

# Read the mesh network of the model:
gio = GmshIO()
with open('C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\test\\Flow_02_test\\square_mesh.msh') as f:
    gio.read(f)
   
# Getting the centers for each element:
coord = np.zeros((len(gio.elements),3))
for i, one_el in enumerate(gio.elements.values()):
    i_nodes = one_el[2]
    coord[i] = np.average(np.array([ gio.nodes[i_node] for i_node in i_nodes]), axis=0)       

# Setting the "field" (conductivity)
pole = SpatialCorrelatedField(corr_exp = 'gauss', dim = 3, corr_length = 0.3,aniso_correlation = None,  )
pole.set_points(coord, mu = 0.8, sigma = 0.25)
n    = 12  # Number of realizations
f    = np.zeros(n)
cum_mean  = Cumul(len(coord))
cum_sigma = Cumul(len(coord))

for j in range(n): 
    # Generating the "field" (conductivity) 
    conductivity = pole.sample() 
    cum_mean += conductivity
    centered = conductivity - 0.8
    cum_sigma += centered * centered
    # Plotting the conductivity field (optional)
    # plt.scatter(coord[:,0],coord[:,1],c = conductivity)
    # plt.colorbar()          
    CFL = conductivity.max()*0.025/0.035
    print("Max CFl:",CFL)
    gio.write_fields('C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\test\\Flow_02_test\\vodivost_square.msh',conductivity,"vodivost") 
    
    # Running Flow123d:
    os.chdir("C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\test\\Flow_02_test")
    os.system("call fterm.bat //opt/flow123d/bin/flow123d -s 02_mysquare.yaml'")
    
    # Extracting out the result
    soubor = open('C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\test\\Flow_02_test\\output\\mass_balance.txt','r')
    output = []
    for line in soubor:
        line = line.rstrip()
        if re.search('1', line):
            x = line
        
    y    = x.split('"conc"',100)  
    z    = y[1].split('\t')
    f[j] = -float(z[3])  # The solute flux [kg?] out of the east BC at the end of simulation
                
# Postprocessing f:
                