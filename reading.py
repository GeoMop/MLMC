from gmsh_io import GmshIO
from operator import add
import numpy as np
from correlated_field import SpatialCorrelatedField
import matplotlib.pyplot as plt

# Read the mesh network of the model:
gio = GmshIO()
with open('C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\rectangle.msh') as f:
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
log_K        = pole.sample()
conductivity = log_K   

# Plotting the conductivity field (optional)
# plt.scatter(coord[:,0],coord[:,1],c = conductivity)
# plt.colorbar()    

with open('C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\vodivost.msh',"wb") as fout:
     fout.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
     fout.write('$ElementData\n')
     fout.write('1\n"vodivost"\n0\n3\n0\n1\n')
     fout.write('%d\n'%len(gio.elements))
     for i in range(len(gio.elements)):
         fout.write(str(i+1) +'\t'+ str(conductivity[i]) + '\n') 
     fout.write('$EndElementData\n')
fout.close()     
