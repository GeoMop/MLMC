"""Module containing an expanded python gmsh class"""
from __future__ import print_function

import struct
import numpy as np
import enum


# class ElementType(enum.IntEnum):
#     simplex_1d = 1
#     simplex_2d = 2
#     simplex_3d = 4
#
# element_sizes = {
#     1: 1,
#     2: 2,
#     4: 3
# }
#

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
        self.element_data = {}

    def read_element_data_head(self, mshfile):

        columns = mshfile.readline().strip().split()
        n_str_tags = int(columns[0])
        assert (n_str_tags == 1)
        field = mshfile.readline().strip().strip('"')

        columns = mshfile.readline().strip().split()
        n_real_tags = int(columns[0])
        assert (n_real_tags == 1)
        columns = mshfile.readline().strip().split()
        time = float(columns[0])

        columns = mshfile.readline().strip().split()
        n_int_tags = int(columns[0])
        assert (n_int_tags == 3)
        columns = mshfile.readline().strip().split()
        t_idx = float(columns[0])
        columns = mshfile.readline().strip().split()
        n_comp = float(columns[0])
        columns = mshfile.readline().strip().split()
        n_elem = float(columns[0])
        return field, time, t_idx, n_comp, n_elem


    def read(self, mshfile=None):
        """Read a Gmsh .msh file.

        Reads Gmsh format 1.0 and 2.0 mesh files, storing the nodes and
        elements in the appropriate dicts.
        """

        if not mshfile:
            mshfile = open(self.filename, 'r')

        readmode = 0
        print('Reading %s' % mshfile.name)
        line = 'a'
        while line:
            line = mshfile.readline()
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
                elif line == '$PhysicalNames':
                    readmode = 5
                elif line == '$ElementData':
                    field, time, t_idx, n_comp, n_ele = self.read_element_data_head(mshfile)
                    field_times = self.element_data.setdefault(field, {})
                    assert t_idx not in field_times
                    self.current_elem_data = {}
                    self.current_n_components = n_comp
                    field_times[t_idx] = (time, self.current_elem_data)
                    readmode = 6
                else:
                    readmode = 0
            elif readmode:
                columns = line.split()
                if readmode == 6:
                    ele_idx = int(columns[0])
                    comp_values = [float(col) for col in columns[1:]]
                    assert len(comp_values) == self.current_n_components
                    self.current_elem_data[ele_idx] = comp_values

                if readmode == 5:
                    if len(columns) == 3:
                        self.physical[str(columns[2])] = (int(columns[1]), int(columns[0]))

                if readmode == 4:
                    if len(columns) == 3:
                        vno, ftype, dsize = (float(columns[0]),
                                             int(columns[1]),
                                             int(columns[2]))
                        print(('ASCII', 'Binary')[ftype] + ' format')
                    else:
                        endian = struct.unpack('i', columns[0])
                if readmode == 1:
                    # Version 1.0 or 2.0 Nodes
                    try:
                        if ftype == 0 and len(columns) == 4:
                            self.nodes[int(columns[0])] = [float(col) for col in columns[1:]]
                        elif ftype == 1:
                            nnods = int(columns[0])
                            for N in range(nnods):
                                data = mshfile.read(4 + 3 * dsize)
                                i, x, y, z = struct.unpack('=i3d', data)
                                self.nodes[i] = [x, y, z]
                            mshfile.read(1)
                    except ValueError:
                        print('Node format error: ' + line, ERROR)
                        readmode = 0
                elif ftype == 0 and readmode > 1 and len(columns) > 5:
                    # Version 1.0 or 2.0 Elements
                    try:
                        columns = [int(col) for col in columns]
                    except ValueError:
                        print('Element format error: ' + line, ERROR)
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
                            tags = columns[3:3 + ntags]
                            nodes = columns[3 + ntags:]
                        self.elements[id] = (type, tags, nodes)
                elif readmode == 3 and ftype == 1:
                    # el_type : num of nodes per element
                    tdict = {1: 2, 2: 3, 3: 4, 4: 4, 5: 5, 6: 6, 7: 5, 8: 3, 9: 6, 10: 9, 11: 10, 15: 1}
                    try:
                        neles = int(columns[0])
                        k = 0
                        while k < neles:
                            etype, ntype, ntags = struct.unpack('=3i',
                                                                mshfile.read(3 * 4))
                            k += 1
                            for j in range(ntype):
                                mysize = 1 + ntags + tdict[etype]
                                data = struct.unpack('=%di' % mysize,
                                                     mshfile.read(4 * mysize))
                                self.elements[data[0]] = (etype,
                                                          data[1:1 + ntags],
                                                          data[1 + ntags:])
                    except:
                        raise
                    mshfile.read(1)

        print('  %d Nodes' % len(self.nodes))
        print('  %d Elements' % len(self.elements))

        mshfile.close()

    def write_ascii(self, mshfile=None):
        """Dump the mesh out to a Gmsh 2.0 msh file."""

        if not mshfile:
            mshfile = open(self.filename, 'w')

        print('$MeshFormat\n2.2 0 8\n$EndMeshFormat', file=mshfile)
        print('$PhysicalNames\n%d' % len(self.physical), file=mshfile)
        for name in sorted(self.physical.keys()):
            value = self.physical[name]
            region_id, dim = value
            print('%d %d "%s"' % (dim, region_id, name), file=mshfile)
        print('$EndPhysicalNames', file=mshfile)
        print('$Nodes\n%d' % len(self.nodes), file=mshfile)
        for node_id in sorted(self.nodes.keys()):
            coord = self.nodes[node_id]
            print(node_id, ' ', ' '.join([str(c) for c in coord]), sep="",
                  file=mshfile)
        print('$EndNodes', file=mshfile)
        print('$Elements\n%d' % len(self.elements), file=mshfile)
        for ele_id in sorted(self.elements.keys()):
            elem = self.elements[ele_id]
            (ele_type, tags, nodes) = elem
            print(ele_id, ' ', ele_type, ' ', len(tags), ' ',
                  ' '.join([str(c) for c in tags]), ' ',
                  ' '.join([str(c) for c in nodes]), sep="", file=mshfile)
        print('$EndElements', file=mshfile)

    def write_binary(self, filename=None):
        """Dump the mesh out to a Gmsh 2.0 msh file."""

        if not filename:
            filename = self.filename

        mshfile = open(filename, 'wr')

        mshfile.write("$MeshFormat\n2.2 1 8\n")
        mshfile.write(struct.pack('@i', 1))
        mshfile.write("\n$EndMeshFormat\n")
        mshfile.write("$Nodes\n%d\n" % (len(self.nodes)))
        for node_id, coord in self.nodes.items():
            mshfile.write(struct.pack('@i', node_id))
            mshfile.write(struct.pack('@3d', *coord))
        mshfile.write("\n$EndNodes\n")
        mshfile.write("$Elements\n%d\n" % (len(self.elements)))
        for ele_id, elem in self.elements.items():
            (ele_type, tags, nodes) = elem
            mshfile.write(struct.pack('@i', ele_type))
            mshfile.write(struct.pack('@i', 1))
            mshfile.write(struct.pack('@i', len(tags)))
            mshfile.write(struct.pack('@i', ele_id))
            for c in tags:
                mshfile.write(struct.pack('@i', c))
            for c in nodes:
                mshfile.write(struct.pack('@i', c))
        mshfile.write("\n$EndElements\n")

        mshfile.close()

    def write_element_data(self, f, ele_ids, name, values):
        """
        Write given element data to the MSH file. Write only a single '$ElementData' section.
        :param f: Output file stream.
        :param ele_ids: Iterable giving element ids of N value rows given in 'values'
        :param name: Field name.
        :param values: np.array (N, L); N number of elements, L values per element (components)
        :return:

        TODO: Generalize to time dependent fields.
        """
        n_els = values.shape[0]
        n_comp = np.atleast_1d(values[0]).shape[0]
        np.reshape(values, (n_els, n_comp))
        header_dict = dict(
            field=str(name),
            time=0,
            time_idx=0,
            n_components=n_comp,
            n_els=n_els
        )

        header = "1\n" \
                 "\"{field}\"\n" \
                 "1\n" \
                 "{time}\n" \
                 "3\n" \
                 "{time_idx}\n" \
                 "{n_components}\n" \
                 "{n_els}\n".format(**header_dict)

        f.write('$ElementData\n')
        f.write(header)
        assert len(values.shape) == 2
        for ele_id, value_row in zip(ele_ids, values):
            value_line = " ".join([str(val) for val in value_row])
            f.write("{:d} {}\n".format(int(ele_id), value_line))
        f.write('$EndElementData\n')

    def write_fields(self, msh_file, ele_ids, fields):
        """
        Creates input data msh file for Flow model.
        :param msh_file: Target file (or None for current mesh file)
        :param ele_ids: Element IDs in computational mesh corrsponding to order of
        field values in element's barycenter.
        :param fields: {'field_name' : values_array, ..}
        """
        if not msh_file:
            msh_file = open(self.filename, 'w')
        with open(msh_file, "w") as fout:
            fout.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
            for name, values in fields.items():
                self.write_element_data(fout, ele_ids, name, values)


    def read_element_data(self):
        """
        Write given element data to the MSH file. Write only a single '$ElementData' section.
        :param f: Output file stream.
        :param ele_ids: Iterable giving element ids of N value rows given in 'values'
        :param name: Field name.
        :param values: np.array (N, L); N number of elements, L values per element (components)
        :return:

        TODO: Generalize to time dependent fields.
        """

        n_els = values.shape[0]
        n_comp = np.atleast_1d(values[0]).shape[0]
        np.reshape(values, (n_els, n_comp))
        header_dict = dict(
            field=str(name),
            time=0,
            time_idx=0,
            n_components=n_comp,
            n_els=n_els
        )

        header = "1\n" \
                 "\"{field}\"\n" \
                 "1\n" \
                 "{time}\n" \
                 "3\n" \
                 "{time_idx}\n" \
                 "{n_components}\n" \
                 "{n_els}\n".format(**header_dict)

        f.write('$ElementData\n')
        f.write(header)
        assert len(values.shape) == 2
        for ele_id, value_row in zip(ele_ids, values):
            value_line = " ".join([str(val) for val in value_row])
            f.write("{:d} {}\n".format(int(ele_id), value_line))
        f.write('$EndElementData\n')
