"""
Pat a Mat routine to create the geo file for the fractured grid, some
limitations regarding fractures crossing the edges to not to let gmsh fail
"""
import os

def create_msh(frac_set, cl1, cl2):
    
    first  = 'cl1 = ' + str(cl1) + '; \n'
    second = 'cl2 = ' + str(cl2) + '; \n'
    with open('basic_nofrac.geo', "r") as in_file:
        content = in_file.readlines()
    with open('basic.geo', "w") as out_file:
        out_file.write(first)
        out_file.write(second)
        for line in content:
            out_file.write(line)
    
    if len(frac_set) == 0:
        os.system('call gmsh basic.geo -2 -o basic.msh')
        msh_file = "basic.msh"
        fraclist = []
            
    else:        
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
    
def adjust_yaml(fraclist, chars, mtrx_cond, separate ): # Adds the particular number of fracture into yaml source file,
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
        count = 0    
        with open('frac_new.yaml', "w") as out_file:
            for line in content:
                out_file.write(line)
                count += 1
                if count == 17 + n_f:
                    out_file.write('      - region: bulk_0' + '\n')
                    out_file.write('        conductivity: ' + str(mtrx_cond) + '\n')
                    out_file.write('        anisotropy: 1 ' + '\n')
                    for i in range(n_f):
                        out_file.write('      - region: ' + str(fraclist[i]) + "\n") 
                        out_file.write('        conductivity: ' + str(chars[i,4].tolist()) + "\n") 
                        out_file.write('        cross_section: ' + str(chars[i,5].tolist()) + "\n")
                        out_file.write('        sigma: ' + str(chars[i,6].tolist()) + "\n")

    
    yaml_file = 'frac_new.yaml'              
    return yaml_file 