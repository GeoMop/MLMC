import frac_geom
import os
import numpy as np
from fractures import Fractures
import matplotlib.pyplot as plt
from create_msh import  adjust_yaml
from MCwork import FlowMC, mc_stats

n_realzs  = 16
f         = np.zeros(n_realzs,)
iter_n    = 0
failure   = 0

fig, axes = plt.subplots(nrows = 4, ncols = 4)
for i in range(n_realzs): 
    box        = ([0,0],[1.,1.])
    mesh_step  = 0.1
    frac_step  = 0.03
    frac       = Fractures(np.array((0,1)),np.array((0,1)),'uniform')
    set_1      = frac.add_fracset(0.35,0.7,3) 
    while set_1 == []:
        set_1      = frac.add_fracset(0.35,0.7,3) 
        set_2      = frac.add_fracset(1.75,0.4,3)
    frac_chars = frac.set_conds(frac.coords,log_mean_cs = -2.5,var_cs = 0.2, sigma = 0.9) 
    # sets       = np.concatenate(frac.coords,axis = 0)
    pukliny    = [[] for x in range(len(frac.coords))]
    for i in range(len(frac.coords)):
        pukliny[i] = (frac.coords[i,0:2],frac.coords[i,2:4])
   
    frac_geom.make_frac_mesh(box, mesh_step, pukliny, frac_step)
    
    # plt.subplot(4,4,i+1)
    # frac.fracs_plot(set_1)
    # frac.fracs_plot(set_2)
    # plt.show() 
    
    msh_path = 'fractured_2D.msh'
    n_f      = len(frac.coords)
    fraclist = [' ']*n_f
    for i in range(n_f):
          fraclist[i] = 'frac_' + str(i)
    yaml_path           = adjust_yaml(fraclist,frac_chars, mtrx_cond = 0.2, separate = True) 
    
    run1      = FlowMC(yaml_path, msh_path) # MC simulator   
    run1.Flow_run(yaml_path)
    f[i] = run1.extract_value()
    iter_n += 1
    plt.title(str(f[i]))