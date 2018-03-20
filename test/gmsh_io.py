""" Module containing an expanded python gmsh class"""
from __future__ import print_function

import struct



class GmshIO:
    """This is a class for storing nodes and elements. Based on Gmsh.py

    Members:
    nodes -- A dict of the form { nodeID: [ xcoord, ycoord, zcoord] }
    elements -- A dict of the form { elemID: (type, [tags], [nodeIDs]) }
    physical -- A dict of the form { name: (id, dim) }

    Methods:
    read([file]) -- Parse a Gmsh version 1.0 or 2.0 mesh file
    write([file]) -- Output a Gmsh version 2.0 mesh file
    """

    def __init__(self, filename=None):
        """Initialise Gmsh data structure"""
        self.reset()
        self.filename = filename
        if self.filename:
            self.read()

    def reset(self):
        """Reinitialise Gmsh data structure"""
        self.nodes = {}
        self.elements = {}
        self.physical = {}

    def read(self, mshfile=None):
        """Read a Gmsh .msh file.
        
        Reads Gmsh format 1.0 and 2.0 mesh files, storing the nodes and
        elements in the appropriate dicts.
        """

        if not mshfile:
            mshfile = open(self.filename,'r')

        readmode = 0
        print('Reading %s'%mshfile.name)
        line='a'
        while line:
            line=mshfile.readline()
            line = line.strip()
            if line.startswith('$'):
                if line == '$NOD' or line == '$Nodes':
                    readmode = 1
                elif line == '$ELM':
                    readmode = 2
                elif line == '$Elements':
                    readmode = 3
                elif line == '$MeshFormat':
                    readmode = 4
                else:
                    readmode = 0
            elif readmode:
                columns = line.split()
                if readmode == 4:
                    if len(columns)==3:
                        vno,ftype,dsize=(float(columns[0]),
                                         int(columns[1]),
                                         int(columns[2]))
                        print(('ASCII','Binary')[ftype]+' format')
                    else:
                        endian=struct.unpack('i',columns[0])
                if readmode == 1:
                    # Version 1.0 or 2.0 Nodes
                    try:
                        if ftype==0 and len(columns)==4:
                            self.nodes[int(columns[0])] = [ float(col) for col  in  columns[1:] ]
                        elif ftype==1:
                            nnods=int(columns[0])
                            for N in range(nnods):
                                data=mshfile.read(4+3*dsize)
                                i,x,y,z=struct.unpack('=i3d',data)
                                self.nodes[i]=[x,y,z]
                            mshfile.read(1)
                    except ValueError:
                        print('Node format error: '+line, ERROR)
                        readmode = 0
                elif ftype==0 and  readmode > 1 and len(columns) > 5:
                    # Version 1.0 or 2.0 Elements 
                    try:
                        columns = [ int(col) for col in columns ]
                    except ValueError:
                        print('Element format error: '+line, ERROR)
                        readmode = 0
                    else:
                        (id, type) = columns[0:2]
                        if readmode == 2:
                            # Version 1.0 Elements
                            tags = columns[2:4]
                            nodes = columns[5:]
                        else:
                            # Version 2.0 Elements
                            ntags = columns[2]
                            tags = columns[3:3+ntags]
                            nodes = columns[3+ntags:]
                        self.elements[id] = (type, tags, nodes)
                elif readmode == 3 and ftype==1:
                    # el_type : num of nodes per element
                    tdict={1:2,2:3,3:4,4:4,5:5,6:6,7:5,8:3,9:6,10:9,11:10,15:1}
                    try:
                        neles=int(columns[0])
                        k=0
                        while k<neles:
                            etype,ntype,ntags=struct.unpack('=3i',
                                                            mshfile.read(3*4))
                            k+=1
                            for j in range(ntype):
                                mysize=1+ntags+tdict[etype]
                                data=struct.unpack('=%di'%mysize,
                                                   mshfile.read(4*mysize))
                                self.elements[data[0]]=(etype,
                                                        data[1:1+ntags],
                                                        data[1+ntags:])
                    except:
                        raise
                    mshfile.read(1)
                            
        print('  %d Nodes'%len(self.nodes))
        print('  %d Elements'%len(self.elements))

        mshfile.close()

    def write_ascii(self, mshfile=None):
        """Dump the mesh out to a Gmsh 2.0 msh file."""

        if not mshfile:
            mshfile = open(self.filename, 'w')


        print('$MeshFormat\n2.2 0 8\n$EndMeshFormat', file=mshfile)
        print('$PhysicalNames\n%d'%len(self.physical), file=mshfile)
        for name in sorted(self.physical.keys()):
            value = self.physical[name]
            region_id, dim = value
            print('%d %d "%s"'%(dim, region_id, name), file=mshfile)
        print('$EndPhysicalNames', file=mshfile)
        print('$Nodes\n%d'%len(self.nodes), file=mshfile)
        for node_id in sorted(self.nodes.keys()):
            coord = self.nodes[node_id]
            print(node_id,' ',' '.join([str(c) for c in  coord]), sep="",
                  file=mshfile)
        print('$EndNodes',file=mshfile)
        print('$Elements\n%d'%len(self.elements),file=mshfile)
        for ele_id in sorted(self.elements.keys()):
            elem = self.elements[ele_id]
            (ele_type, tags, nodes) = elem
            print(ele_id,' ',ele_type,' ',len(tags),' ',
                  ' '.join([str(c) for c in tags]),' ',
                  ' '.join([str(c) for c in nodes]), sep="", file=mshfile)
        print('$EndElements',file=mshfile)

    def write_binary(self, filename=None):
        """Dump the mesh out to a Gmsh 2.0 msh file."""

        if not filename:
            filename = self.filename

        mshfile = open(filename, 'wr')

        mshfile.write("$MeshFormat\n2.2 1 8\n")
        mshfile.write(struct.pack('@i',1))
        mshfile.write("\n$EndMeshFormat\n")
        mshfile.write("$Nodes\n%d\n"%(len(self.nodes)))
        for node_id, coord in self.nodes.items():
            mshfile.write(struct.pack('@i',node_id))
            mshfile.write(struct.pack('@3d',*coord))
        mshfile.write("\n$EndNodes\n")
        mshfile.write("$Elements\n%d\n"%(len(self.elements)))
        for ele_id, elem in self.elements.items():
            (ele_type, tags, nodes) = elem
            mshfile.write(struct.pack('@i',ele_type))
            mshfile.write(struct.pack('@i',1))
            mshfile.write(struct.pack('@i',len(tags)))
            mshfile.write(struct.pack('@i',ele_id))
            for c in tags:
                mshfile.write(struct.pack('@i',c))
            for c in nodes:
                mshfile.write(struct.pack('@i',c))
        mshfile.write("\n$EndElements\n")
                      
        mshfile.close()
        
    def write_fields(self,mshfile,fieldvalues,name='vodivost'):
        """
        Creates msh file for Flow model
        """
        if not mshfile:
            mshfile = open(self.filename, 'w')
        with open(mshfile,"w") as fout:
         fout.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
         fout.write('$ElementData\n')
         fout.write('1\n"'+str(name)+'"')
         fout.write('\n0\n3\n0\n1\n')
         fout.write('%d\n'%len(self.elements))
         for i in range(len(self.elements)):
             fout.write(str(i+1) +'\t'+ str(fieldvalues[i]) + '\n') 
         fout.write('$EndElementData\n')
        fout.close()       
        
