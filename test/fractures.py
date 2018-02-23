import os
from gmsh_io import GmshIO
import numpy as np
from  correlated_field import SpatialCorrelatedField
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import vonmises_line
from scipy.stats import powerlaw
import math
import random

def Angle(mu):
        # mu is the mean angle, expressed in radians between 0 and 2*pi, and kappa is the concentration parameter, 
        # which must be greater than or equal to zero. If kappa is equal to zero, this distribution reduces to a uniform random angle over the range 0 to 2*pi.
        kappa = 50
        angle = random.vonmisesvariate(mu,kappa) 
        return angle
        

def PoissonPP(rt,Dx, Dy):
        # rate (rt)  = rho*V, density within the area V, here V is equal to 1  = Dx*Dy
        N = sp.stats.poisson( rt*Dx*Dy ).rvs()
        x = sp.stats.uniform.rvs(0,Dx,((N,1)))
        y = sp.stats.uniform.rvs(0,Dy,((N,1)))
        P = np.hstack((x,y))
        return P 
        
def Lengths(mean_length,amount):
        # Length of the fracture 
        r          = powerlaw.rvs(mean_length,size = amount)                              
        return r
        
class Fractures(object):
    '''
    When initialized it creates empty set of fractures, each fractures set with its own aorientation, lengths and 
    density can be add via add_fracset method
    All fractures (start and end of 2d lines [x1,y1,x2,y2], stored in self.coords, all centers in self.centers
    '''
           
    def  __init__(self,dx, dy, type_cc):
        # dx = np.array of grid extent in x direction
        # dy = np.array of grid extent in y directon
        # type_cc, at the moment any type (string)
        self.dx = dx
        self.dy = dy
        self.typecc = type_cc    # for placing the fractures, only one type - uniform, at the moment    
        self.coords = np.empty((0,4),float) # start and end of the fracture line
        self.centers = np.empty((0,2),float)
          
    def get_centers(self, rate):
        Dx = self.dx.max() - self.dx.min()
        Dy = self.dy.max() - self.dy.min()     
        centers = PoissonPP(rate, Dx,Dy)
        self.centers = np.concatenate((self.centers,centers),axis = 0)
        self.centers = np.asarray(self.centers) # Adding different fracture sets into the self.centers
        return centers
    
    def add_fracset(self, mean_angle, mean_length, rate):
        centers    = self.get_centers(rate)
        L          = Lengths(mean_length,len(centers)) # Length of the fracture
        coords_set = np.zeros((len(centers),4))
        min_x      = self.dx.min() + 0.025 # Intersecton of fracture and boundary creates errors in gmsh file
        min_y      = self.dx.min() + 0.025
        max_x      = self.dy.max() - 0.025
        max_y      = self.dy.max() - 0.025
        
        for i in range(len(centers)):
            l_i    = max(0.05,L[i]) # Minimal length of the fracture
            theta  = Angle(mean_angle)
            a      = math.sin(theta)*l_i
            b      = math.cos(theta)*l_i 
            x1,y1  = centers[i,0] - b, centers[i,1] - a
            x2,y2  = centers[i,0] + b, centers[i,1] + a
        
        # Clip the first point:
            if x1 < min_x:
                x1,y1 = min_x, min_x*math.tan(theta) + centers[i,1] -centers[i,0]*math.tan(theta)
            if x1> max_x:
                x1,y1 = max_x, max_x*math.tan(theta) + centers[i,1] -centers[i,0]*math.tan(theta) 
            if y1< min_y:
                x1,y1 = (min_y - centers[i,1] + math.tan(theta)*centers[i,0])/math.tan(theta), min_y
            if y1> max_y:
                x1,y1 = (max_y - centers[i,1] + math.tan(theta)*centers[i,0])/math.tan(theta), max_y
            
        # Clip the second point:
            if x2 < min_x:
                x2,y2 = min_x, min_x*math.tan(theta) + centers[i,1] -centers[i,0]*math.tan(theta)
            if x2> max_x:
                x2,y2 = max_x, max_x*math.tan(theta) + centers[i,1] -centers[i,0]*math.tan(theta) 
            if y2< min_y:
                x2,y2 = (min_y - centers[i,1] + math.tan(theta)*centers[i,0])/math.tan(theta), min_y
            if y2> max_y:
                x2,y2 = (max_y - centers[i,1] + math.tan(theta)*centers[i,0])/math.tan(theta), max_y    
                    
            coords_set[i,:] = [x1,y1,x2,y2] 
        
        self.coords = np.concatenate((self.coords,coords_set),axis = 0)
        self.coords = np.asarray(self.coords) # Adding the set into the self.coords
        return coords_set            
        
               
    def fracs_plot(self,coords):
        nf = len(coords)
#       plt.scatter(self.centers[:,0],self.centers[:,1])
        for i in range(nf):
            plt.plot([coords[i,0],coords[i,2] ], [coords[i,1], coords[i,3]], color='k', linestyle='-', linewidth=2)
#        plt.axes.set_xlim(left = min(self.grid[:,0]),right = max(self.grid[:,0])) 
#        plt.axes.set_ylim(left = min(self.grid[:,1]),right = max(self.grid[:,1]))   
        plt.show()
        
          
            
            
        
    

               
 