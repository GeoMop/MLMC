import fractures
import numpy as np
from fractures import Fractures
from MCwork import FlowMC, confidence_interval, mc_stats
import imp
from create_msh import create_msh, adjust_yaml
import matplotlib.pyplot as plt

imp.reload(fractures)
n_realzs  = 16
f         = np.zeros(n_realzs,)
iter_n    = 0
fig, axes = plt.subplots(nrows = 4, ncols = 4)
for i in range(n_realzs):
    frac      = Fractures(np.array((0,1)),np.array((0,1)),'uniform')
    # Assigns fracture sets of given angle, mean_length and density per unit square:
    set_1     = frac.add_fracset(0.35,0.4,2) 
    set_2     = frac.add_fracset(1.75,0.2,4)
    print(frac.coords)
    
    plt.subplot(4,4,i+1)
    frac.fracs_plot(set_1)
    frac.fracs_plot(set_2) 
    
    # Create geo + msh file, with physical lines for fractures
    msh_path, fraclist = create_msh(frac.coords)
    yaml_path          = adjust_yaml(fraclist, separate = True) # Adding region "frac" with all the fractures in
    
    run1      = FlowMC(yaml_path, msh_path) # MC simulator   
    run1.Flow_run(yaml_path)
    f[i] = run1.extract_value()
    iter_n += 1
    plt.title(str(f[i]))

mc_stats(f)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                