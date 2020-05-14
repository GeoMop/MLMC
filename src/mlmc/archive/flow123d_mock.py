"""
Mockup for the Flow123d simulator.
"""

import argparse
import ruamel.yaml as yaml

parser = argparse.ArgumentParser(description='Flow123d mockup.')
parser.add_argument('yaml_file', metavar='<main YAML>', type=string, nargs=1,
                    help='Main input YAML file.')

args = parser.parse_args()

with open(args.yaml_file, 'r') as f:
    input = yaml.load(f)
    mesh_file = input.mesh.mesh_file

def conductivity_field_average(gmsh):
    pass
