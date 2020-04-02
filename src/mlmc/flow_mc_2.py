"""
This file is used in 02_cond test.
TODO:
- modify test to use standard flow_mc base simulation.
- remove this file
"""
import os
import os.path
import subprocess
import time as t
import gmsh_io
import numpy as np
import json
import glob
from datetime import datetime as dt
import shutil
import copy
import mlmc.simulation as simulation
import mlmc.sample as sample
from mlmc.generate_fields import FieldGenerator


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


def force_mkdir(path, force=False):
    """
    Make directory 'path' with all parents,
    remove the leaf dir recursively if it already exists.
    :param path: path to directory
    :param force: if dir already exists then remove it and create new one
    :return: None
    """
    if force:
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.makedirs(path, mode=0o775, exist_ok=True)


class FlowSim(simulation.Simulation):
    # placeholders in YAML
    total_sim_id = 0
    # MESH_FILE_VAR = 'mesh_file'
    # # Timestep placeholder given as O(h), h = mesh step
    # TIMESTEP_H1_VAR = 'timestep_h1'
    # # Timestep placeholder given as O(h^2), h = mesh step
    # TIMESTEP_H2_VAR = 'timestep_h2'

    # files
    GEO_FILE = 'mesh.geo'
    MESH_FILE = 'mesh.msh'
    YAML_TEMPLATE = 'flow_input.yaml.tmpl'
    YAML_FILE = 'flow_input.yaml'
    FIELDS_FILE = 'fields_sample.msh'

    """
    Gather data for single flow call (coarse/fine)
    """

    def __init__(self, mesh_step, level_id=None, config=None, clean=False, parent_fine_sim=None):
        """

        :param config: configuration of the simulation, processed keys:
            env - Environment object.
            fields - FieldSet object
            yaml_file: Template for main input file. Placeholders:
                <mesh_file> - replaced by generated mesh
                <FIELD> - for FIELD be name of any of `fields`, replaced by the FieldElementwise field with generated
                 field input file and the field name for the component.
                 (TODO: allow relative paths, not tested but should work)
            geo_file: Path to the geometry file. (TODO: default is <yaml file base>.geo
        :param mesh_step: Mesh step, decrease with increasing MC Level.
        :param parent_fine_sim: Allow to set the fine simulation on previous level (Sim_f_l) which corresponds
        to 'self' (Sim_c_l+1) as a coarse simulation. Usually Sim_f_l and Sim_c_l+1 are same simulations, but
        these need to be different for advanced generation of samples (zero-mean control and antithetic).
        """
        if level_id is not None:
            self.sim_id = level_id
        else:
            self.sim_id = FlowSim.total_sim_id
            FlowSim.total_sim_id += 1

        self.env = config['env']

        # self.field_config = config['field_name']
        # self._fields_inititialied = False
        # self._fields = copy.deepcopy(config['fields'])
        self.time_factor = config.get('time_factor', 1.0)
        self.base_yaml_file = config['yaml_file']
        self.base_geo_file = config['geo_file']
        self.field_template = config.get('field_template',
                                         "!FieldElementwise {mesh_data_file: $INPUT_DIR$/%s, field_name: %s}")

        # print("init fields template ", self.field_template)

        self.step = mesh_step
        # Pbs script creater
        self.pbs_creater = self.env["pbs"]

        # Set in _make_mesh
        self.points = None
        # Element centers of computational mesh.
        self.ele_ids = None
        # Element IDs of computational mesh.
        self.n_fine_elements = 0
        # Fields samples
        self._input_sample = {}

        # TODO: determine minimal element from mesh
        self.time_step_h1 = self.time_factor * self.step
        self.time_step_h2 = self.time_factor * self.step * self.step

        # Prepare base workdir for this mesh_step
        output_dir = config['output_dir']
        self.work_dir = os.path.join(output_dir, 'sim_%d_step_%f' % (self.sim_id, self.step))
        force_mkdir(self.work_dir, clean)

        self.mesh_file = os.path.join(self.work_dir, self.MESH_FILE)

        self.coarse_sim = None
        self.coarse_sim_set = False

        super(simulation.Simulation, self).__init__()

    def n_ops_estimate(self):
        """
        Number of operations
        :return: int
        """
        return self.n_fine_elements

    # def _substitute_yaml(self, yaml_tmpl, yaml_out):
    #     """
    #     Create substituted YAML file from the template.
    #     :return:
    #     """
    #     param_dict = {}
    #     field_tmpl = self.field_template
    #     for field_name in self._fields.names:
    #         param_dict[field_name] = field_tmpl % (self.FIELDS_FILE, field_name)
    #     param_dict[self.MESH_FILE_VAR] = self.mesh_file
    #     param_dict[self.TIMESTEP_H1_VAR] = self.time_step_h1
    #     param_dict[self.TIMESTEP_H2_VAR] = self.time_step_h2
    #     used_params = substitute_placeholders(yaml_tmpl, yaml_out, param_dict)
    #     self._fields.set_outer_fields(used_params)

    def set_coarse_sim(self, coarse_sim=None):
        """
        Set coarse simulation ot the fine simulation so that the fine can generate the
        correlated input data sample for both.

        Here in particular set_points to the field generator
        :param coarse_sim
        """
        self.coarse_sim = coarse_sim
        self.coarse_sim_set = True
        #self.n_fine_elements = len(self.points)

    def _make_fields(self):
        if self.coarse_sim is None:
            self._fields.set_points(self.points, self.point_region_ids, self.region_map)
        else:
            coarse_centers = self.coarse_sim.points
            both_centers = np.concatenate((self.points, coarse_centers), axis=0)
            both_regions_ids = np.concatenate((self.point_region_ids, self.coarse_sim.point_region_ids))
            assert self.region_map == self.coarse_sim.region_map
            self._fields.set_points(both_centers, both_regions_ids, self.region_map)

        self._fields_inititialied = True

    # Needed by Level
    def generate_random_sample(self):
        """
        Generate random field, both fine and coarse part.
        Store them separeted.
        :return:
        """
        # Prepare mesh
        geo_file = os.path.join(self.work_dir, self.GEO_FILE)
        shutil.copyfile(self.base_geo_file, geo_file)

        field_gen = FieldGenerator(self.env["gmsh"])
        field_gen.make_mesh(self.mesh_file, geo_file, self.step)

        yaml_template = os.path.join(self.work_dir, self.YAML_TEMPLATE)
        shutil.copyfile(self.base_yaml_file, yaml_template)
        self.yaml_file = os.path.join(self.work_dir, self.YAML_FILE)

        field_gen.substitute_yaml(yaml_template, self.yaml_file, self.time_step_h1, self.time_step_h2,
                         self.mesh_file, self.field_template, self.FIELDS_FILE)
        #self._substitute_yaml(yaml_template, self.yaml_file)
        
        fields_sample = field_gen.generate_fields(self.mesh_file)

        # Common computational mesh for all samples.
        # self._make_mesh(geo_file, self.mesh_file)

        # Prepare main input YAML
        
        self.points = field_gen.points
        self.ele_ids = field_gen.ele_ids
        
        #self._extract_mesh(self.mesh_file)
        #self._make_fields()
        self.n_fine_elements = len(self.points)

        #fields_sample = self._fields.sample()
        self._input_sample = {name: values[:self.n_fine_elements, None] for name, values in fields_sample.items()}
        if self.coarse_sim is not None:
            self.coarse_sim._input_sample = {name: values[self.n_fine_elements:, None] for name, values in
                                             fields_sample.items()}

    def simulation_sample(self, sample_tag, sample_id, start_time=0):
        """
        Evaluate model using generated or set input data sample.
        :param sample_tag: A unique ID used as work directory of the single simulation run.
        :return: tuple (sample tag, sample directory path)
        TODO:
        - different mesh and yaml files for individual levels/fine/coarse
        - reuse fine mesh from previous level as coarse mesh

        1. create work dir
        2. write input sample there
        3. call flow through PBS or a script that mark the folder when done
        """
        out_subdir = os.path.join("samples", str(sample_tag))
        sample_dir = os.path.join(self.work_dir, out_subdir)

        force_mkdir(sample_dir, True)
        fields_file = os.path.join(sample_dir, self.FIELDS_FILE)

        gmsh_io.GmshIO().write_fields(fields_file, self.ele_ids, self._input_sample)
        prepare_time = (t.time() - start_time)
        package_dir = self.run_sim_sample(out_subdir)

        return sample.Sample(directory=sample_dir, sample_id=sample_id,
                             job_id=package_dir, prepare_time=prepare_time)

    def run_sim_sample(self, out_subdir):
        """
        Add simulations realization to pbs file
        :param out_subdir: MLMC output directory
        :return: Package directory (directory with pbs job data)
        """
        lines = [
            'cd {work_dir}',
            'date +%y.%m.%d_%H:%M:%S',
            'time -p {flow123d} --yaml_balance -i {output_subdir} -s {work_dir}/flow_input.yaml  -o {output_subdir} >{work_dir}/{output_subdir}/flow.out',
            'date +%y.%m.%d_%H:%M:%S',
            'touch {output_subdir}/FINISHED',
            'echo \\"Finished simulation:\\" \\"{flow123d}\\" \\"{work_dir}\\" \\"{output_subdir}\\"',
            '']

        # Add flow123d realization to pbs script
        package_dir = self.pbs_creater.add_realization(self.n_fine_elements, lines,
                                                       output_subdir=out_subdir,
                                                       work_dir=self.work_dir,
                                                       flow123d=self.env['flow123d'])

        return package_dir

    def get_run_time(self, sample_dir):
        """
        Get flow123d sample running time from profiler
        :param sample_dir: Sample directory
        :return: float
        """
        profiler_file = os.path.join(sample_dir, "profiler_info_*.json")
        profiler = glob.glob(profiler_file)[0]

        try:
            with open(profiler, "r") as f:
                prof_content = json.load(f)

            run_time = float(prof_content['children'][0]['cumul-time-sum'])
        except:
            print("Extract run time failed")

        return run_time


