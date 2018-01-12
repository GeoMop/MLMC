import os
import sys
libdir = os.path.join(os.path.split(
         os.path.dirname(os.path.realpath(__file__)))[0],"C:\\Users\\Klara\\Documents\\Intec\\MLMC\\src\\mlmc")
sys.path.insert(1,libdir)

from gmsh_io import GmshIO, ElementType
from operator import add
import numpy as np
from  correlated_field import SpatialCorrelatedField
import matplotlib.pyplot as plt
import re
from scipy.stats.mstats import mquantiles
import seaborn as sns

# Read the mesh network of the model:
gio = GmshIO()
with open('C:\\Users\\Klara\\Documents\\Intec\\MLMC\\test\\Flow_02_test\\square_mesh.msh') as f:
    gio.read(f)


# Getting the centers for each element:
coord = np.zeros((len(gio.elements),3))
for i, one_el in enumerate(gio.elements.values()):
    i_nodes = one_el[2]
    coord[i] = np.average(np.array([ gio.nodes[i_node] for i_node in i_nodes]), axis=0)
    # size = element_sizes[index]
    # if len(index) == ElementType.simplex_3d:
    #     coord[i] = map(add,gio.nodes[index[0]],map(add,gio.nodes[index[1]],gio.nodes[index[2]]))
    #     coord[i] = coord[i]/3
    # if len(index) == ElementType.simplex_2d:
    #     coord[i] = map(add,gio.nodes[index[0]],gio.nodes[index[1]])
    #     coord[i] = coord[i]/2
    # if len(index) == ElementType.simplex_1d:
    #     coord[i] = map(add,gio.nodes[index[0]],gio.nodes[index[1]])
    #     coord[i] = coord[i]

# Seeting the "field" (conductivity)
pole = SpatialCorrelatedField(corr_exp = 'gauss', dim = 3, corr_length = 0.3  )
pole.set_points(coord, mu = 0.8, sigma = 0.25)
n    = 100  # Number of realizations
f    = np.zeros(n,)

# Running the MC simulations:
for j in range(n): 
    # Generating the "field" (conductivity) 
    conductivity = pole.sample() 
    
    # Plotting the conductivity field (optional)
    # plt.scatter(coord[:,0],coord[:,1],c = conductivity)
    # plt.colorbar()          
    CFL = conductivity.max()*0.025/0.035  # based on the grid for 02_mysquare.yaml
    print("Max CFl:",CFL,"run:",j)
    gio.write_fields('C:\\Users\\Klara\\Documents\\Intec\\MLMC\\test\\Flow_02_test\\vodivost_square.msh',conductivity,"vodivost") 
    
    # Running Flow123d:
    os.chdir("C:\\Users\\Klara\\Documents\\Intec\\MLMC\\test\\Flow_02_test")
    os.system("call fterm.bat //opt/flow123d/bin/flow123d -s 02_mysquare.yaml'")
    
    # Extracting out the result
    soubor = open('C:\\Users\\Klara\\Documents\\Intec\\MLMC\\test\\Flow_02_test\\output\\mass_balance.txt','r')
    output = []
    for line in soubor:
        line = line.rstrip()
        if re.search('1', line):
            x = line
        
    y    = x.split('"conc"',100)  
    z    = y[1].split('\t')
    f[j] = -float(z[3])  # The solute flux [kg?] out of the east BC at the end of simulation
                
# Postprocessing f:
sns.distplot(f)
mquantiles(f,[0.01,0.05,0.5,0.95,0.99])
plt.show()
                