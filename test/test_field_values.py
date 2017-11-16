# TEST OF CONSISTENCY in the field values generated
import numpy as np
from spat_corr_field import SpatialCorrelatedField
import matplotlib.pyplot as plt

# ===========  A structure grid of points: =====================================
#
hx = np.linspace(0,16,num = 17) # x-direction coordinates
hy = np.linspace(0,16,num = 17)  # y-direction coordinates
hz = np.linspace(0,16,num = 17)  # z-direction coordinates

[X,Y,Z] = np.meshgrid(hx,hy,hz) 
N       = len(hx)*len(hy)*len(hz)
points  = np.column_stack((X.reshape([N,], order='C'),Y.reshape([N,], order='C'),Z.reshape([N,], order='C')))
n       = len(points)

# ================= Random fields generated for the same subset =================
""" 
 A full covariance matrix for all the grid points and another one for only a 
subset of grid points is created (N = 200). A full SVD decomposiiton and  a truncated 
one are performed to derive L (full C) and Lr (reduced C) matrix.
For the same subset o fpoints mulyiple realizations are produced, for the same
field characteristics (mu, sigma, corr).
Form the full field only the subset point values are extracted and statistically 
compared against the reduced C, Lr realizations.
""" 
N       = 200    # size of the subset of points
Xp      = np.zeros([N,3])
k       = 500   # the number of realizations 

fp_ave  = np.zeros([k,]); fp_var  = np.zeros([k,]); Fp_all  = np.zeros((N,k))
fr_ave  = np.zeros([k,]); fr_var  = np.zeros([k,]); Fr_all  = np.zeros((N,k)) 
sind    = np.zeros([N,]);     fp  =  np.zeros([N,]) 

# The subset of points + indexes:
for i in range(N): 
        sind[i] = np.random.randint(0,n)
        Xp[i,:] = points[int(sind[i]),:]

# The field and two different SVD decompositions:
sigma     = 3.3
corr      = 8
mu        = 5        
pole      = SpatialCorrelatedField(sigma, corr, mu,'gauss')
L,ev      = pole.svd_dcmp(points, full=True)                
Lr,evr    = pole.svd_dcmp(Xp) 

# Testing 
for j in range(k):
      m           = np.random.normal(0, 1,len(ev))
      F           = L.dot(m) + mu
      for i in range(N):    # extract points
        fp[i]     = F[int(sind[i])]
        fp_ave[j] = fp.mean()
        fp_var[j] = fp.var()
      
      m           = np.random.normal(0, 1,len(evr))
      fr          = Lr.dot(m) + mu 
      fr_ave[j]   = fr.mean()
      fr_var[j]   = fr.var()
      
# ============================ Comparison ======================================    
fig = plt.figure(figsize=plt.figaspect(0.5))
ax  = fig.add_subplot(1, 2, 1)
ax.set_title('The average value',size = 11)
plt.plot(fp_ave, label = 'From the field on fine grid')
plt.plot(fr_ave, label = 'With reduced covariance matrix')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

ax  = fig.add_subplot(1, 2, 2)
ax.set_title('The variance',size = 11)
plt.plot(fp_var, label = 'From the field on fine grid')
plt.plot(fr_var, label = 'With reduced covariance matrix')  
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)      

plt.show()