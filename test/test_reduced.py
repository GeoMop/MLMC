# Testing the COV matrices estimates for reduced subset of points
import numpy as np
from spat_corr_field import SpatialCorrelatedField
import matplotlib.pyplot as plt

# =========== Grid of points: =================================================
hx = np.linspace(0,16,num = 17) # x-direction coordinates
hy = np.linspace(0,16,num = 17)  # y-direction coordinates
hz = np.linspace(0,12,num = 13)  # z-direction coordinates

X, Y, Z = np.meshgrid(hx, hy, hz)
N = len(hx)*len(hy)*len(hz)
points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
n = len(points)

# ========== Setup the full C and field ========================================
# A field with covariance matrix C, her full decomposition, defined on a regular structured grid
sigma     = 7
corr      = 5.4
mu        = 0.0
field      = SpatialCorrelatedField(sigma, corr, 'gauss')
L, ev      = field.svd_dcmp(points, full=True)
m         = np.random.normal(0, 1,L.shape[1]) 
F         = L.dot(m) + mu
f3d       = np.reshape(F,((len(hx),len(hy), len(hz))),order='C')
C         = field.cov_matrix(points)

# =========== Subset of points =================================================
# The same field but defined just for a set of points (subset)
subset    = field.points
C_red     = field.cov_matrix(subset)
Lr, evr   = field.svd_dcmp(subset, full = True)


# TODO:
# - Comparison of full vs. truncated SVD in terms of covariance matrix estimates.
# - make timing of SVD and MC simulation
# - move random point selection out of the SpatialCorrelatedField class


# =========== Testing ==========================================================
# Multiple realizations of the field for the "small" subset of points
K = 10000;
mean_est = 0
cov_est = np.zeros((len(subset),len(subset)))
for i in range(K):
     m    = np.random.normal(0,1,len(evr))
     f    = Lr.dot(np.hstack(m)) + mu   # field based on reduced cov matrix
     cov_est  = cov_est + (f - mu)*(np.vstack(f) - mu)  # estimate of cov matrix based on the subset points
     mean_est = mean_est + f.mean()  #average of the field
     
cov_est /= K     # the "averaged" covariance matrix based on points realizations
mean_est /= K

sigma_est = np.mean(np.diag(cov_est))
print("The estimated sigma is {}".format(sigma_est) )
print("The estimated average is {}".format(mean_est) )


# =========== Some visualization ===============================================
fig = plt.figure(figsize=plt.figaspect(0.25))
ax  = fig.add_subplot(1, 5, 1)
ax.set_title('Full covariance',size = 9)
plt.imshow(C)
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1, 5, 2)
ax.set_title('The field on full grid (slice)',size = 9)
plt.imshow(f3d[:,:,6])
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1, 5,3)
ax.set_title('One of the realizations',size = 9)
plt.scatter(subset[:,0],subset[:,1],subset[:,2], c=f)
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1, 5,4)
ax.set_title('The reduced covariance',size = 9)
plt.imshow(C_red)
plt.colorbar(orientation='horizontal')
ax  = fig.add_subplot(1, 5,5)
ax.set_title('The difference:C_red - C_est',size = 9)
plt.imshow(cov_est - C_red)
plt.colorbar(orientation='horizontal')
plt.show()



