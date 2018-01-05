# Time comparison
import time
import numpy as np
from spat_corr_field import SpatialCorrelatedField
import random
from tabulate import tabulate
import matplotlib.pyplot as plt
from Points_select import Points_select
        

#============ Field produced ==============================               
sigma     = 2
corr      = 5.4
mu        = 0.0
field     = SpatialCorrelatedField(sigma, corr, 'gauss')
size      = [20,20,20]

#=========== Time of reduced versus full decompositions for increaseing number of points ============
v   = np.r_[6:13]
t1  = np.zeros([len(v),]); t2 = np.zeros([len(v),]);
nr  = np.zeros([len(v),]); nf = np.zeros([len(v),]);
out = np.zeros([len(v),4])
for i in range(len(v)):
    N      = 2**(v[i])
    subset = Points_select('rand',size, N)
    field.set_points(subset)
    start  = time.time()
    L,ev   = field.svd_dcmp()
    end    = time.time()
    t1[i]  = end-start
    nr[i]  = len(ev)

    start  = time.time()    
    Lf,evf = field.svd_dcmp(full=True)
    end    = time.time()
    t2[i]  = end-start
    nf[i]  = N    
    
    out[i,:] = [N,len(ev),round(t1[i],2),round(t2[i],2)]
    
# ========= Visuals ===========================
print 'N  ','N_red','t_red ','t_full'
print tabulate(out)
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Time against size of cov matrix (N)',size = 11)
ax.set_xlabel('N (size)')
ax.set_ylabel('time [s]')
plt.xscale('log')
curve1, = plt.plot(nf,t1,color = 'green',marker = '^',linestyle='none', label = 'Reduced size (N red)')
curve2, = plt.plot(nf,t2,color = 'blue',marker = '^',linestyle='none', label = 'Full size') 
plt.legend(handles=[curve1,curve2])
for i, txt in enumerate(nr):
    plt.annotate(txt, (nf[i],t1[i]))
   
plt.show()
