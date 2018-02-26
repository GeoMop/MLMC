"""
Pat a Mat routine to create the geo file for the fractured grid, some
limitations regarding fractures crossing the edges to not to let gmsh fail
"""
import os
import random
import numpy

def create_msh(frac_set):
    n_f       = len(frac_set)
    points    = [' ']*2*n_f
    lines     = [' ']*n_f
    fraclist  = [' ']*n_f
    
    os.system('del basic.geo')
    os.system('copy basic_nofrac.geo basic.geo')
    
    retez1 = 'Point('
    retez2 = ') =' 
    retez3 = 'Line('
    retez4 = 'Physical Line("frac_'
    
    with open('basic.geo', "a") as file:
        file.write("\n")
        for i in range(n_f):
            points[2*i]   = retez1 + str(5 + 2*i) + retez2 + '{' + str(round(frac_set[i,0],2)) + ',' + str(round(frac_set[i,1],2)) + ',0,cl2};'
            points[2*i+1] = retez1 + str(5 + 2*i+1) + retez2 + '{' + str(round(frac_set[i,2],2)) + ',' + str(round(frac_set[i,3],2)) + ',0,cl2};'    
            file.write(points[2*i] + "\n")
            file.write(points[2*i+1] + "\n")
            
        file.write("\n")
        for i in range(n_f): 
            lines[i] = retez3 + str(20 + i) + retez2 + '{' + str(6+2*i-1) + ',' + str(6+2*i) + '};'
            file.write(lines[i] + "\n")  
            
        file.write("\n")
        for i in range(n_f):
            phylines = retez4  + str(i+1) + '") = {' + str(20 + i) + '};'
            fraclist[i] = 'frac_' + str(i+1)
            file.write(phylines + "\n")
            
        surf = 'Line{20:' + str(20 + n_f-1) + '} In Surface{32};'
        file.write("\n" + surf)            
        
    os.system('call gmsh basic.geo -2 -o basic.msh')
    msh_file = "basic.msh"
    
    return msh_file, fraclist 
    
def adjust_yaml(fraclist, separate): # Adds the particular number of fracture into yaml source file,
    # which are part of region: fracs
    os.system('del frac_new.yaml')
    n_f = len(fraclist)
    with open('101_frac_square.yaml', "r") as in_file:
        content = in_file.readlines()
    
    count = 0    
    with open('frac_new.yaml', "w") as out_file:
        for line in content:
            out_file.write(line)
            count += 1
            if count == 9 :
               for i in range(n_f):
                  out_file.write('          - ' + str(fraclist[i]) + "\n") 
    
    with open('frac_new.yaml', "r") as in_file:
        content = in_file.readlines()                            
    if separate:
        frac_cs = numpy.zeros(shape=(n_f,))
        frac_cond = numpy.zeros(shape=(n_f,))
        frac_sig = numpy.zeros(shape=(n_f,))
        for i in range(n_f):            
            frac_cs[i] = round(random.gauss(0.05,0.005),2)
            frac_cond[i] = 0.1 + 3*frac_cs[i]
            frac_sig[i] = 0.5
        count = 0    
        with open('frac_new.yaml', "w") as out_file:
            for line in content:
                out_file.write(line)
                count += 1
                if count == 20 + n_f:
                    for i in range(n_f):
                        out_file.write('      - region: ' + str(fraclist[i]) + "\n") 
                        out_file.write('        conductivity: ' + str(frac_cond[i].tolist()) + "\n") 
                        out_file.write('        cross_section: ' + str(frac_cs[i].tolist()) + "\n")
                        out_file.write('        sigma: ' + str(frac_sig[i].tolist()) + "\n")

    
    yaml_file = 'frac_new.yaml'              
    return yaml_file                   