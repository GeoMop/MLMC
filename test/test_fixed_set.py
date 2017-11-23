# Testing the COV matrices estimates for a fixed subset of points
import numpy as np
from spat_corr_field import SpatialCorrelatedField
from tabulate import tabulate
import matplotlib.pyplot as plt
import time
from Points_select import Points_select

# =========== Grid of points: =================================================
size    = [30,30,24]
ncells  = [16,16,12]
points  = Points_select('grid',size, ncells)
n       = len(points)
k       = 0
mylist  = np.zeros([5,6])

for j in np.logspace(0.2,0.6,num = 5):               
        # ========== Setup the full C and field ========================================
        # A field with covariance matrix C, her full decomposition, defined on a regular structured grid
        sigma     = 4.6
        corr      = 10*round(j,1)
        mu        = 0.0
        field     = SpatialCorrelatedField(sigma, corr, 'gauss')
        field.set_points(points)
        
        F         = field.values()
        f3d       = np.reshape(F,((ncells[0],ncells[1],ncells[2])),order='C')
        C         = field.cov_matrix()
        
        # =========== SVD decomposition (full and truncated)==========================================
        start  = time.time()
        L,ev   = field.svd_dcmp(full = True)
        end    = time.time()
        t_full = end-start
        
        start  = time.time()
        Lr,evr = field.svd_dcmp()
        end    = time.time()
        t_red  = end-start
        
        # =========== Testing ==========================================================
        # Multiple realizations of the field for the "small" subset of points
        # Comparing the MC derived Cov matrix with the analytical ful one
        K = 1000;
        mean_est = 0
        cov_est = np.zeros((len(points),len(points)))
        for i in range(K):
            m     = np.random.normal(0,1,len(evr))
            ft    = Lr.dot(np.hstack(m)) + mu   # field based on reduced cov matrix
        
            cov_est  = cov_est + (ft - mu)*(np.vstack(ft) - mu)  # estimate of cov matrix based on the subset points
            mean_est = mean_est + ft.mean()  #average of the field
            
        cov_est  /= K     # the "averaged" covariance matrix based on points realizations
        mean_est /= K
        sigma_est = np.mean(np.diag(cov_est))
        
        mylist[k,:] = [corr,K,len(evr),round(sigma_est,3),round(mean_est,3), np.linalg.norm(C - cov_est)]
        print('time for C_full:', t_full, 'time for C_red:',t_red)
        k           = k+1
               
print 'corr','K', 'n_red', 'sig_est', 'ave_est','||C - C_est||'
print tabulate(mylist)

# =========== Some visualization ===============================================
fig = plt.figure(figsize=plt.figaspect(0.25))
ax  = fig.add_subplot(1, 4, 1)
ax.set_title('The field on full grid (slice)',size = 9)
plt.imshow(f3d[:,:,4])
#plt.scatter(subset[:,0],subset[:,1],subset[:,2], c=ft)
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1, 4,2)
ax.set_title('The difference C - C_est',size = 9)
plt.imshow(C-cov_est)
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1, 4,3)
ax.set_title('The reduced covariance',size = 9)
plt.imshow(C)
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1,4,4)
ax.set_title('C_red based on the fields from red. SVD decompositon',size = 9)
plt.imshow(cov_est)
plt.colorbar(orientation='horizontal')
plt.show()

#fig2 = plt.figure(figsize=plt.figaspect(0.7))
#plt.plot(mylist)


