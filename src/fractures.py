# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
import math
import random

def Angle(mu):
        """
        von Mises - Fisher distribution:
            generating an angle 'theta' which is a deviation from some mean angle 'mu' (expressed in radians between 0 and 2pi)
            'kappa' - inverse measure of the dispersion, kappa = 0 --> uniform random dist from (0,2pi)
            von Mises is a special case of vMF
            
            f(theta|mu,kappa) = exp(kappa*cos(theta - mu))/ 2*pi I_0(kappa)
            
            Parashar, Rishi, and Donald M. Reeves. "On iterative techniques for computing flow in large two-dimensional discrete fracture networks." 
            Journal of computational and applied mathematics 236.18 (2012): 4712-4724.
            Baghbanan, Alireza, and Lanru Jing. "Hydraulic properties of fractured rock masses with correlated fracture length and aperture." 
            International Journal of Rock Mechanics and Mining Sciences 44.5 (2007): 704-719.
            
        Fisher distribution:
            f(theta|mu,kappa) = exp(kappa*cos(kappa mu.T*theta))/ 4*pi sinh(kappa)
            
            Hyman, J. D., et al. "Fracture size and transmissivity correlations: Implications for transport simulations in sparse 
            three‐dimensional discrete fracture networks following a truncated power law distribution of fracture size."
            Water Resources Research 52.8 (2016): 6472-6489.
                
        """

        kappa = 100
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
        r = mean_length * powerlaw.rvs(a = 5,size = amount)                              
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
        self.typecc  = type_cc    # for placing the fractures, only one type - uniform, at the moment    
        self.coords  = np.empty((0,4),float) # start and end of the fracture line
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
        min_x      = self.dx.min()# + 0.025 # Intersecton of fracture and boundary creates errors in gmsh file
        min_y      = self.dx.min()# + 0.025
        max_x      = self.dy.max()# - 0.025
        max_y      = self.dy.max()# - 0.025
        
        for i in range(len(centers)):
            l_i    = max(0.05,L[i]) # Minimal length of the fracture
            theta  = Angle(mean_angle)
            a      = 0.5*math.sin(theta)*l_i
            b      = 0.5*math.cos(theta)*l_i 
            x1,y1  = centers[i,0] - b, centers[i,1] - a
            x2,y2  = centers[i,0] + b, centers[i,1] + a
        
        # Clip the first point:
            if x1 < min_x:
                x1,y1 = min_x, min_x*math.tan(theta) + centers[i,1] -centers[i,0]*math.tan(theta)
            if x1 > max_x:
                x1,y1 = max_x, max_x*math.tan(theta) + centers[i,1] -centers[i,0]*math.tan(theta) 
            if y1 < min_y:
                x1,y1 = (min_y - centers[i,1] + math.tan(theta)*centers[i,0])/math.tan(theta), min_y
            if y1 > max_y:
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
                    
            coords_set[i,:] = (x1,y1,x2,y2) 
        
        self.coords = np.concatenate((self.coords,coords_set),axis = 0)
        self.coords = np.asarray(self.coords) # Adding the set into the self.coords
        return coords_set            
    
    def set_conds(self, coords, log_mean_cs = -2.5, var_cs = 0.2, sigma = 0.5):
        nf = len(coords)
        frac_cs   = np.random.lognormal(log_mean_cs,var_cs,nf) # Lognormal distribution for aperture
        frac_cond = (frac_cs**2)/12                            # Cubic law for conductivity
        frac_sig  = sigma * np.ones(shape=(nf,))               # Sigma fixed

        frac_char = np.transpose(np.vstack((frac_cond, frac_cs, frac_sig)))
        frac_char = np.hstack((coords,frac_char))
        return frac_char      
               
    def fracs_plot(self,coords):
        nf = len(coords)
        for i in range(nf):
            plt.plot([coords[i,0],coords[i,2] ], [coords[i,1], coords[i,3]], color='k', linestyle='-', linewidth=2)
#        plt.axes.set_xlim(left = min(self.grid[:,0]),right = max(self.grid[:,0])) 
#        plt.axes.set_ylim(left = min(self.grid[:,1]),right = max(self.grid[:,1]))   
    plt.show()