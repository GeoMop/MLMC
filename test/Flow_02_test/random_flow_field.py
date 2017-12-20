import os
import sys
libdir = os.path.join(os.path.split(
         os.path.dirname(os.path.realpath(__file__)))[0],"C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\src\\mlmc")
sys.path.insert(1,libdir)

from gmsh_io import GmshIO
#GmshIO.path.append('C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\src\\mlmc\\gmsh_io')
from operator import add
import numpy as np
from  correlated_field import SpatialCorrelatedField
#import matplotlib.pyplot as plt

# Read the mesh network of the model:
gio = GmshIO()
with open('C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\test\\Flow_02_test\\rectangle.msh') as f:
    gio.read(f)
   
# Getting the centers for each element:
coord = np.zeros((len(gio.elements),3))
for i in range(len(gio.elements)):
    one_el = gio.elements[i+1]
    index = one_el[2]
    coord[i] = map(add,gio.nodes[index[0]],map(add,gio.nodes[index[1]],gio.nodes[index[2]]))
coord = coord/3

# Generating the "field" (conductivity)
pole = SpatialCorrelatedField(corr_exp = 'gauss', dim = 3, corr_length = 2.6,aniso_correlation = None,  )
pole.set_points(coord, mu = 3, sigma = 1.0)
conductivity = pole.sample()  

# Plotting the conductivity field (optional)
# plt.scatter(coord[:,0],coord[:,1],c = conductivity)
# plt.colorbar()          

gio.write_fields('C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\test\\Flow_02_test\\vodivost.msh',conductivity,"vodivost") 
