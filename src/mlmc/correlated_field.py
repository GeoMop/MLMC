import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd  


class SpatialCorrelatedField(object):
    """
    Generating realizations of a spatially correlated random field F for a fixed set of points at X.
    E[F(x)]       = mu(x) 
    Cov[x_i,x_j]  = E[(F(x_i) - mu(x))(F(x_j) - mu(x))]  
    Cov[x_i,x_i]  = E[(F(x_i) - mu(x))(F(x_i) - mu(x))] = sigma 
    
    Here: Covariance matrix based on exponential Gaussian model:
          Cov_x_i,j = exp(-0.5*(1/c)*(x_i - x_j)^2)
          
    SVD decomposition:
        Considering first m vectors, such that lam(m)/lam(0) <0.1
  """

    def  __init__(self, sigma, corr, correlation_type='gauss'):
        """
        :param sigma: scalar, float, standard deviance of single uncorelated field value.
        :param corr: scalar, float, correlation length
        :param mu:
        :param correlation_type:
        """
        self.sigma = sigma
        self.corr = corr
        self.typ = correlation_type
        #self.points    = self.get_points() 
  

    def set_points(self, points, mu = 0.0):
        """
        :param points: list of x,y,z coordinates :type array, matrix [N x 3], x,y,z
        :param mu: Scalar or numpy vector of same length as the number of points.
        :return: None
        """
        self.points = points
        assert type(mu) == float or mu.shape == (len(points), )
        self.mu = mu

     
    def cov_matrix(self): 
     # Creates the covariance matrix for set of points
     n     = len(self.points)
     C     = np.zeros((n,n))
     c      = self.corr*np.array([1.,1.,1.])
     
     for i in range(n):
           point  = (self.points[i,:]) 
           if self.typ == 'gauss':
               x      = -0.5*np.square(self.points - np.tile(point,(n,1))).dot(1./c)
           elif self.typ== 'exp':
               x      = -0.5*abs(self.points - np.tile(point,(n,1))).dot(1./c)    
           C[:,i] = (self.sigma)*np.exp(x)  
           
     return C            
     
    def svd_dcmp(self,full = 0):
        # points ... list of x,y,z coordinates :type array, matrix [N x 3], x,y,z

        # Does decomposition of covariance matrix defined by set of points 
        # C ~= U*diag(ev) * V, L = U*sqrt(ev)
        
        C           = self.cov_matrix()
        if full == 1:
            U,ev,VT    = np.linalg.svd(C)
            m          = len(ev) 
        elif full==0:
            m            = int(np.floor(2*np.sqrt(len(self.points))))
            threshold    = 1
            while threshold > 0.05:          
                U, ev, VT    = randomized_svd(C, n_components=m, n_iter=3,random_state=None)
                threshold    = ev[-1]/ev[0]
                m            = int(np.floor(1.5*m))
                     
        s            = np.sqrt(ev[0:m])
        self.L       = U[:,0:m].dot(sp.diag(s))
        
        return self.L, ev[0:m]
                        
    def values(self):
    # Generates the actual field values, field = mu + L*m, where m ~ iid from N(0,1)  
        if hasattr(self,'L')== False:
             self.svd_dcmp(full = True)    
        m         = np.random.normal(0, 1,self.L.shape[1]) 
        return   self.L.dot(m) + self.mu
        
          
#=====================================================================
# Example:
"""
F    = SpatialCorrelatedField()
L,ev = F.svd_dcmp(F.points)
F.values()
""" 
                      
                 
        