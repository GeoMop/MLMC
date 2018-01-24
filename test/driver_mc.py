import MCwork
from MCwork import FlowMC, confidence_interval, mc_stats
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fileDir = os.path.dirname(os.path.realpath(__file__))

yaml_path = '02_mysquare.yaml' 
mesh_path = 'square_mesh.msh'

run2      = FlowMC(yaml_path, mesh_path)

n_realzs  = 20
f         = np.zeros(n_realzs,)
iter_n    = 0

for i in range(n_realzs):
   run2.add_field('vodivost',0.7,0.15,0.3)
   run2.Flow_run(yaml_path)
   f[i] = run2.extract_value()
   iter_n += 1
   plt.plot(iter_n,f[0:iter_n-1].mean(),'*')
    
# Postprocessing f:
sns.distplot(f)
plt.show()    
mc_stats(f)
