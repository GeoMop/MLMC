import numpy as np
import random


def Points_select(typ, size, c):
  """
    typ : 'grid'
    size: array of size 3 with spatial max dimensions of the grid, initial point at [0,0,0] by default
    c   : array of size 3 with number of cells in each dimensions
     
    or
    typ  : 'rand' 
    size : max coordinate in each direction
    c    : number of random points
  """
    
  if typ =='grid':
    hx = np.linspace(0,size[0],num = c[0])  # x-direction coordinates
    hy = np.linspace(0,size[1],num = c[1])  # y-direction coordinates
    hz = np.linspace(0,size[2],num = c[2])  # z-direction coordinates

    X, Y, Z = np.meshgrid(hx, hy, hz)
    points  = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
     
  elif typ =='rand':
    points  = np.zeros((c,3))  
    for i in range(c):
            points[i,:] = [random.random()*size[0],random.random()*size[1], random.random()*size[2]]      
     
  return points

 

