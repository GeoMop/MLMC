from MCwork import FlowMC
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import mquantiles

fileDir = os.path.dirname(os.path.realpath(__file__))

yaml_path = '02_mysquare.yaml' 
mesh_path = 'square_mesh.msh'

run2      = FlowMC(yaml_path, mesh_path)

n_realzs  = 400 
f         = np.zeros(n_realzs,)

for i in range(n_realzs):
    run2.add_field('vodivost',0.7,0.15,0.3)
    run2.Flow_run(yaml_path)
    f[i] = run2.extract_value()
    
# Postprocessing f:
sns.distplot(f)
mquantiles(f,[0.01,0.05,0.5,0.95,0.99])
plt.show()    