import filecmp
import os.path
import numpy as np
import geoInterface

interface = geoInterface.PyIntGeo(geoInterface.PyIntGeo.modPath+'/Tests/')
test = np.array([[0, 0, 0], [0, 5, 0], [5, 5, 0], [5, 0, 0]])
interface.generate_boundary(test)
test2 = np.array([[0, 1.6, 1, 1], [4, 3.4, 1, 1], [5, 1, 1, 2], [3, 3, 1, 2], [2,4,1,3],[3,2,1,3]])
interface.write_tectonics(test2)

sourcefiles = ['boundary_source', 'tectonics_generator_Claire_2D', 'intersect_2d_area_with_tects']
filenew = 'frac_list'
interface.geo_handler(sourcefiles, filenew)

#os.system('call gmsh-3.0.5-Windows\gmsh.exe frac_list.geo -2')

