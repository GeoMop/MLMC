import os
import os.path
import subprocess
import time as t
import sys
# src_path = os.path.dirname(os.path.abspath(__file__))
# print("src path ", src_path)
# sys.path.append(os.path.join(src_path, '..', '..', 'src'))
#from gmsh_api import gmsh

import numpy as np
import json
import glob
from datetime import datetime as dt
import shutil
import copy
import mlmc.simulation as simulation
import mlmc.sample as sample
import mlmc.correlated_field as correlated_field
import gmsh_io as gmsh_io

# src_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(src_path, '..'))


# import dfn.src.fracture_homo_cube as frac


class FieldGenerator:
    MESH_FILE_VAR = 'mesh_file'
    # Timestep placeholder given as O(h), h = mesh step
    TIMESTEP_H1_VAR = 'timestep_h1'
    # Timestep placeholder given as O(h^2), h = mesh step
    TIMESTEP_H2_VAR = 'timestep_h2'

    YAML_TEMPLATE = 'flow_input.yaml.tmpl'
    YAML_FILE = 'flow_input.yaml'
    FIELDS_FILE = 'fields_sample.msh'

    def __init__(self, gmsh=None):
        self.mesh_file = None
        self.bulk_fields = None
        self.fracture_fields = None
        self.gmsh = gmsh

        # self.mesh_file

        self.set_fields()

    def set_fields(self):
        conductivity = dict(
            mu=0.0,
            sigma=1.0,
            corr_exp='gauss',
            dim=2,
            corr_length=0.5,
            log=True
        )
        cond_field = correlated_field.SpatialCorrelatedField(**conductivity)
        self.cond_fields = correlated_field.Fields([correlated_field.Field("conductivity", cond_field)])

        # self.fracture_fields = correlated_field.Fields([correlated_field.Field("conductivity", cond_field)])

    def make_mesh(self, mesh_file, geo_file, step):
        """
        Make the mesh, mesh_file: <geo_base>_step.msh.
        Make substituted yaml: <yaml_base>_step.yaml,
        using common fields_step.msh file for generated fields.
        :return:
        """

        subprocess.call([self.gmsh, "-2", '-clscale', str(step), '-o', mesh_file, geo_file])

    def generate_fields(self, mesh_file):
        self._extract_mesh(mesh_file)
        return self._make_fields()

    def _extract_mesh(self, mesh_file):
        """
        Extract mesh from file
        :param mesh_file: Mesh file path
        :return: None
        """
        mesh = gmsh_io.GmshIO(mesh_file)
        is_bc_region = {}
        self.region_map = {}
        for name, (id, _) in mesh.physical.items():
            unquoted_name = name.strip("\"'")
            is_bc_region[id] = (unquoted_name[0] == '.')
            self.region_map[unquoted_name] = id

        bulk_elements = []
        for id, el in mesh.elements.items():
            _, tags, i_nodes = el
            region_id = tags[0]
            if not is_bc_region[region_id]:
                bulk_elements.append(id)

        n_bulk = len(bulk_elements)
        centers = np.empty((n_bulk, 3))
        self.ele_ids = np.zeros(n_bulk, dtype=int)
        self.point_region_ids = np.zeros(n_bulk, dtype=int)

        for i, id_bulk in enumerate(bulk_elements):
            _, tags, i_nodes = mesh.elements[id_bulk]
            region_id = tags[0]
            centers[i] = np.average(np.array([mesh.nodes[i_node] for i_node in i_nodes]), axis=0)
            self.point_region_ids[i] = region_id
            self.ele_ids[i] = id_bulk

        min_pt = np.min(centers, axis=0)
        max_pt = np.max(centers, axis=0)
        diff = max_pt - min_pt
        min_axis = np.argmin(diff)
        non_zero_axes = [0, 1, 2]
        # TODO: be able to use this mesh_dimension in fields
        if diff[min_axis] < 1e-10:
            non_zero_axes.pop(min_axis)
        self.points = centers[:, non_zero_axes]

    def substitute_yaml(self, yaml_tmpl, yaml_out, time_step_h1, time_step_h2, mesh_file, field_tmpl, fields_file):
        """
        Create substituted YAML file from the template.
        :return:
        """
        param_dict = {}
        for field_name in self.cond_fields.names:
            param_dict[field_name] = field_tmpl % (fields_file, field_name)
        param_dict[self.MESH_FILE_VAR] = mesh_file
        param_dict[self.TIMESTEP_H1_VAR] = time_step_h1
        param_dict[self.TIMESTEP_H2_VAR] = time_step_h2
        used_params = substitute_placeholders(yaml_tmpl, yaml_out, param_dict)
        self.cond_fields.set_outer_fields(used_params)

    def _make_fields(self):
        self.cond_fields.set_points(self.points, self.point_region_ids, self.region_map)
        return self.cond_fields.sample()


def substitute_placeholders(file_in, file_out, params):
    """
    Substitute for placeholders of format '<name>' from the dict 'params'.
    :param file_in: Template file.
    :param file_out: Values substituted.
    :param params: { 'name': value, ...}
    """
    used_params = []
    with open(file_in, 'r') as src:
        text = src.read()
    for name, value in params.items():
        placeholder = '<%s>' % name
        n_repl = text.count(placeholder)
        if n_repl > 0:
            used_params.append(name)
            text = text.replace(placeholder, str(value))
    with open(file_out, 'w') as dst:
        dst.write(text)
    return used_params


if __name__ == "__main__":
    gen = Generator()
    gen.make_mesh()
