import os,path
import sys
import fractures
from fractures import Fractures
from scipy.stats import vonmises_line
import MCwork
from MCwork import FlowMC, confidence_interval, mc_stats
import imp

imp.reload(fractures)
yaml_path = 'Flow_02_test/02_mysquare.yaml' 
mesh_path = 'Flow_02_test/square_mesh.msh'

run1      = FlowMC(yaml_path, mesh_path)

frac      = Fractures(run1.points,'uniform')
CC        = frac.get_centers(4) # rate refers to  more or less number of fractures in 1 x 1 size square
endings   = frac.get_coords(CC)

frac.fracs_plot()
