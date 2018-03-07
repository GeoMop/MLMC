from MCwork import FlowMC,  mc_stats
import os
import numpy as np
import matplotlib.pyplot as plt
from create_msh import create_msh, adjust_yaml

fileDir = os.path.dirname(os.path.realpath(__file__))
os.chdir(fileDir)
os.chdir('.\\1_Flow_continuum')
yaml_path = '01_mysquare.yaml'

n_realzs   = 33
f1,f2, f3  = np.zeros(n_realzs,), np.zeros(n_realzs,), np.zeros(n_realzs,)

msh_path, fraclist  = create_msh([], 0.1,0.025)
run1      = FlowMC(yaml_path, msh_path) 
iter_n    = 0
fig1      = plt.figure(1)
for i in range(n_realzs):
    run1.add_field('vodivost',0.7,0.15,0.5)
    run1.Flow_run(yaml_path)
    f1[i] = run1.extract_value()
    iter_n += 1
    plt.plot(iter_n,f1[0:iter_n-1].mean(),'b*')
plt.show()

msh_path, fraclist = create_msh([], 0.05,0.025)
run2      = FlowMC(yaml_path, msh_path)
iter_n    = 0
fig2      = plt.figure(2)
for i in range(n_realzs):
    run2.add_field('vodivost',0.7,0.15,0.5)
    run2.Flow_run(yaml_path)
    f2[i] = run2.extract_value()
    iter_n += 1
    plt.plot(iter_n,f2[0:iter_n-1].mean(),'r*')
plt.show()
 
msh_path, fraclist = create_msh([], 0.035,0.025)
run3      = FlowMC(yaml_path, msh_path)   
iter_n    = 0 
fig3      = plt.figure(3)
for i in range(n_realzs):
    run3.add_field('vodivost',0.7,0.15,0.5)
    run3.Flow_run(yaml_path)
    f3[i] = run3.extract_value()
    iter_n += 1
    plt.plot(iter_n,f3[0:iter_n-1].mean(),'m^')  
plt.show()
          
# Postprocessing f:         
mc_stats(f1)
mc_stats(f2)
mc_stats(f3)

