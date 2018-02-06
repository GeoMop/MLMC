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


def Angles(kappa):
        # Lowering kappa more flat dstribution, peak at 0 Pi,  random values on interval (-Pi,Pi)
        # 1/kappa ~ variance
        angle = vonmises_line.rvs(kappa,loc = 0, scale = 1)
        return angle
        

def PoissonPP(rt,Dx, Dy):
        # rate (rt)  = rho*V, density within the area V, here V is equal to 1  = Dx*Dy
        N = sp.stats.poisson( rt*Dx*Dy ).rvs()
        x = sp.stats.uniform.rvs(0,Dx,((N,1)))
        y = sp.stats.uniform.rvs(0,Dy,((N,1)))
        P = np.hstack((x,y))
        return P              

class Fractures(object):
    '''
    '''
           
    def  __init__(self,grid, type_cc):
        self.grid = grid
        self.dim = grid.shape[1]
        self.typecc = type_cc    # for placing the fractures, only one type - uniform, at the moment              
          
    def get_centers(self, rate):
        # Assuming by default to have an origin at [0,0]
        Dx = self.grid[:,0].max()
        Dy = self.grid[:,1].max()      
        self.centers = PoissonPP(rate, Dx,Dy)
        return self.centers
        
    def get_coords(self,CC):
        # Determins the beginning and end points of fractures in 2d whose centers are definned by CC
        # Those extending the grid area are clipped

        coords = np.zeros((len(CC),4))
        r      = powerlaw.rvs(0.5,size = len(CC)) # Length of the fracture
        min_x  = self.grid[:,0].min()
        min_y  = self.grid[:,1].min()
        max_x  = self.grid[:,0].max()
        max_y  = self.grid[:,1].max()
                
        for i in range(len(CC)):
            length = max(0.02,r[i]) # Minimal length of the fracture
            theta  = Angles(3.6)
            a      = math.sin(theta)*length
            b      = math.cos(theta)*length 
            x1,y1  = CC[i,0] - b, CC[i,1] - a
            x2,y2  = CC[i,0] + b, CC[i,1] + a
            
            # Clip the first point:
            if x1 < min_x:
                x1,y1 = min_x, min_x*math.tan(theta) + CC[i,1] -CC[i,0]*math.tan(theta)
            if x1> max_x:
                x1,y1 = max_x, max_x*math.tan(theta) + CC[i,1] -CC[i,0]*math.tan(theta) 
            if y1< min_y:
                x1,y1 = (min_y - CC[i,1] + math.tan(theta)*CC[i,0])/math.tan(theta), min_y
            if y1> max_y:
                x1,y1 = (max_y - CC[i,1] + math.tan(theta)*CC[i,0])/math.tan(theta), max_y
                
            # Clip the second point:
            if x2 < min_x:
                x2,y2 = min_x, min_x*math.tan(theta) + CC[i,1] -CC[i,0]*math.tan(theta)
            if x2> max_x:
                x2,y2 = max_x, max_x*math.tan(theta) + CC[i,1] -CC[i,0]*math.tan(theta) 
            if y2< min_y:
                x2,y2 = (min_y - CC[i,1] + math.tan(theta)*CC[i,0])/math.tan(theta), min_y
            if y2> max_y:
                x2,y2 = (max_y - CC[i,1] + math.tan(theta)*CC[i,0])/math.tan(theta), max_y    
                         
            coords[i,:] = [x1,y1,x2,y2]   
        
        self.coords = coords                  
        return coords  
            
    def fracs_plot(self):
        nf = len(self.centers)
        plt.scatter(self.centers[:,0],self.centers[:,1])
        for i in range(nf):
            plt.plot([self.coords[i,0],self.coords[i,2] ], [self.coords[i,1], self.coords[i,3]], color='k', linestyle='-', linewidth=2)
#        plt.axes.set_xlim(left = min(self.grid[:,0]),right = max(self.grid[:,0])) 
#        plt.axes.set_ylim(left = min(self.grid[:,1]),right = max(self.grid[:,1]))   
        plt.show()
        
    theta = np.zeros((1000,1))
    for i in range(1000):
       theta[i] = Angles(1.6)             
            
            
        
    

               
 