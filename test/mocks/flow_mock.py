"""
Compute an integral of the conductivity field.
Filed file extracted from the input YAML.
"""
import os
import sys
import yaml
import numpy as np
import argparse

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append( os.path.join(src_path, '..', '..' , 'src') )

import gmsh_io

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input_dir", help="Input dir.", default=None)
parser.add_argument("-o", "--output", dest="output_dir",  help="Output dir.", default=None)
parser.add_argument("-s", "--solve", dest="yaml_file", help="Main input file.")

options, args = parser.parse_known_args()

def default_ctor(loader, tag_suffix, node):
    ctor = yaml.constructor.BaseConstructor()
    dict = ctor.construct_mapping(node)
    #assert isinstance(node, yaml.MappingNode)
    return dict

yaml.add_multi_constructor('', default_ctor)
with open(options.yaml_file, "r") as f:

    input_content = yaml.load(f)

mesh_file = input_content['problem']['mesh']['mesh_file']
mesh = gmsh_io.GmshIO()
with open(mesh_file, "r") as f:
    mesh.read(f)

input_fields = input_content['problem']['flow_equation']['input_fields']
for in_field in input_fields:
    if 'conductivity' in in_field:
        conductivity_file = in_field['conductivity']['gmsh_file']
        conductivity_name = in_field['conductivity']['field_name']

conductivity_file = conductivity_file.replace("${INPUT}", options.input_dir)
field_mesh = gmsh_io.GmshIO()
with open(conductivity_file, "r") as f:
    field_mesh.read(f)

time_idx = 0
time, value_table = field_mesh.element_data[conductivity_name][time_idx]
total = 0.0
for iele, el in mesh.elements.items():
    type, tags, nodes = el
    coords = np.array([ mesh.nodes[inode] for inode in nodes ])
    if coords.shape == (3, 3):
        volume = np.linalg.det(coords[1:, :2] - coords[0,:2])/6
        total += volume * value_table[iele][0]

src_path = os.path.dirname(os.path.abspath(__file__))
balance_mock = os.path.join(src_path, "water_balance_mock.yaml")
with open(balance_mock, "r") as f:
    content = f.read()

content = content.replace("$OUTFLOW$", str(-total))

balance_out = os.path.join(options.output_dir, "water_balance.yaml")
with open(balance_out, "w") as f:
    f.write(content)