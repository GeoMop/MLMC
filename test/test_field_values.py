# TEST OF CONSISTENCY in the field values generated

import numpy as np

from correlated_field import SpatialCorrelatedField
from .points_select import Points_select

#from tabulate import tabulate

# ===========  A structure grid of points: =====================================
size    = [32,32,32]
ncells  = [17,17,13]
points  = Points_select('grid',size, ncells)
n       = len(points)

# ================= Random fields generated for the same subset =================
""" 
 A full covariance matrix for all the grid points and another one for only a 
subset of grid points is created (N fixed). A full SVD decomposiiton and  a truncated 
one are performed to derive L (full C) and Lr (reduced C) matrix.
For the same subset of points multiple realizations are produced with the same
field characteristics (mu, sigma, corr).
From the full field only the subset point values are extracted and statistically 
compared against the reduced Cr, Lr realizations.
""" 


mylist  = np.zeros([10,9])

for l in range(5): 
        N       = 200*(l+1)    # size of the subset of points
        Xp      = np.zeros([N,3])
        k       = 1000   # the number of realizations 
        
        fp_ave  = np.zeros([k,]); fp_var  = np.zeros([k,]); 
        fr_ave  = np.zeros([k,]); fr_var  = np.zeros([k,]); 
        sind    = np.zeros([N,]);     fp  = np.zeros([N,]) 
        
        # The subset of points from the oriignal full field F
        for i in range(N): 
                sind[i] = np.random.randint(0,n)
                Xp[i,:] = points[int(sind[i]),:]       
        # The field and two different SVD decompositions: ==============================
        sigma     = 2.3
        corr      = 8.6
        mu        = 0
                
        F     = SpatialCorrelatedField(sigma, corr,'gauss')
        F.set_points(points)
        L,ev  = F.svd_dcmp(full=True)
        
        f     = SpatialCorrelatedField(sigma, corr,'gauss') 
        f.set_points(Xp)
        Lr,evr    = f.svd_dcmp() 
        
        #=========== Testing ===========================================================
        for j in range(k):
            # Random full field and N points out of it xtracted
            m           = np.random.normal(0, 1,len(ev))
            Fp          = L.dot(m) + mu
            for i in range(N):    # extract points
                fp[i]     = Fp[int(sind[i])]
            fp_ave[j] = fp.mean()
            fp_var[j] = fp.var()
            
            # Random field generated with reduced KL, just for the N points
            m           = np.random.normal(0, 1,len(evr))
            fr          = Lr.dot(m) + mu 
            fr_ave[j]   = fr.mean()
            fr_var[j]   = fr.var()
        
        mylist[2*l,:]   = [1,corr,k,N,len(ev),round(fp_var.mean(),3),sigma,round(fp_ave.mean(),3),mu]
        mylist[2*l+1,:] = [2,corr,k,N,len(evr),round(fr_var.mean(),3),sigma,round(fr_ave.mean(),3),mu]

print('f ' 'corr ',' K   ', 'N  ', 'N_red  ', 'sig_est', 'sig','mean_est','mean')
print tabulate(mylist)      
      
# ============================ Comparison ====================================== 
"""   
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
"""