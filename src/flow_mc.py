import os
import os.path
import yaml
import subprocess
import gmsh_io as gmsh_io
import numpy as np
import shutil
import random
# from operator import add
import simulation as simulation


# from unipath import Path

class Environment:
    def __init__(self, flow123d, gmsh, pbs=None):
        """

        :param flow123d: Flow123d executable path.
        :param gmsh: Gmsh executable path.
        :param pbs: None or dictionary with pbs options :
            n_nodes = 1
            n_cores = 1
            mem - reserved memory for single flow run = 2GB
            queue = 'charon'
        """
        self.flow123d = flow123d
        self.gmsh = gmsh
        self.pbs = pbs


def substitute_placeholders(file_in, file_out, params):
    """
    Substitute for placeholders of format '<name>' from the dict 'params'.
    :param file_in: Template file.
    :param file_out: Values substituted.
    :param params: { 'name': value, ...}
    """
    with open(file_in, 'r') as src:
        text = src.read()
    for name, value in params.items():
        text = text.replace('<%s>' % name, str(value))
    with open(file_out, 'w') as dst:
        dst.write(text)


def force_mkdir(path):
    """
    Make directory 'path' with all parents,
    remove the leaf dir recoursively if it already exists.
    :param path:
    :return:
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


class Pbs:
    __instance = None

    @staticmethod
    def get_instance():
        if Pbs.__instance is None:
            Pbs()
        return Pbs.__instance

    def __init__(self):
        self.steps_sum = 2000
        self.pbs_script = ["#!/bin/bash",
                      '#PBS -S /bin/bash',
                      '#PBS -l select={n_nodes}:ncpus={n_cores}:mem={mem}',
                      '#PBS -q {queue}',
                      '#PBS -N Flow123d',
                      '#PBS -j oe', ]
        self.pbs_script_heading = None
        self.pbs_script_setting = False
        self.pbs_qsub = False

        if Pbs.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Pbs.__instance = self

    def pbs_common_setting(self, **kwargs):
        self.pbs_script = [line.format(**kwargs) for line in self.pbs_script]
        self.pbs_script_heading = self.pbs_script
        self.pbs_script_setting = True

    def add_realization(self, **kwargs):
        lines = [
            'cd {work_dir}',
            'time -p {flow123d} --yaml_balance -i {output_subdir} -s {work_dir}/flow_input.yaml  -o {output_subdir} >{work_dir}/{output_subdir}/flow.out 2>&1',
            'cd {output_subdir}',
            'touch FINISHED',]
        lines = [line.format(**kwargs) for line in lines]
        self.pbs_script , "\n".join(lines)

    def execute(self, dir):
        pbs_file = os.path.join(dir, "pbs_script.sh")
        with open(pbs_file, "w") as f:
            f.write(self.pbs_script)
        os.chmod(pbs_file, 0o774)  # Make exectutable to allow direct call.
        if self.pbs_qsub is True:
            subprocess.call("qsub " + pbs_file, shell=True)
        else:
            subprocess.call(pbs_file)

        self.clear_script()

    def clear_script(self):
        self.pbs_script = self.pbs_script_heading


class FlowSim(simulation.Simulation):
    total_sim_id = 0
    MESH_FILE_PLACEHOLDER = 'mesh_file'
    GEO_FILE = 'mesh.geo'
    MESH_FILE = 'mesh.msh'
    YAML_TEMPLATE = 'flow_input.yaml.tmpl'
    YAML_FILE = 'flow_input.yaml'
    FIELDS_FILE = 'fields_sample.msh'

    """
    Gather data for single flow call (coarse/fine)
    """

    def __init__(self, flow_dict, mesh_step, parent_fine_sim=None):
        """

        :param flow_dict: configuration of the simulation, processed keys:
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
        self.sim_id = FlowSim.total_sim_id
        # Pbs script creater
        self.pbs_creater = Pbs.get_instance()

        FlowSim.total_sim_id += 1
        self.env = flow_dict['env']
        self.fields = flow_dict['fields']
        self.base_yaml_file = flow_dict['yaml_file']
        self.base_geo_file = flow_dict['geo_file']
        self.mesh_step = mesh_step

        # TODO pass to own method
        pbs = self.env["pbs"] or {}
        if len(pbs) > 0:
            self.pbs_creater.pbs_qsub = True
        if self.pbs_creater.pbs_script_setting is False:
            self.pbs_creater.pbs_common_setting(n_nodes=pbs.get('n_nodes', 1),
                                                n_cores=pbs.get('n_cores', 1),
                                                mem=pbs.get('mem', 2),
                                                queue=pbs.get('queue', "charon"))

        # Set in _make_mesh
        self.points = None
        # Element centers of computational mesh.
        self.ele_ids = None
        # Element IDs of computational mesh.

        # Prepare base workdir for this mesh_step
        base_dir = os.path.dirname(self.base_yaml_file)
        self.work_dir = os.path.join(base_dir, 'sim_%d_step_%f' % (self.sim_id, self.mesh_step))
        force_mkdir(self.work_dir)

        # Prepare mesh
        geo_file = os.path.join(self.work_dir, self.GEO_FILE)
        shutil.copy(self.base_geo_file, geo_file)
        self.mesh_file = os.path.join(self.work_dir, self.MESH_FILE)
        # Common computational mesh for all samples.
        self._make_mesh(geo_file, self.mesh_file)

        # Prepare main input YAML
        yaml_template = os.path.join(self.work_dir, self.YAML_TEMPLATE)
        shutil.copy(self.base_yaml_file, yaml_template)
        self.yaml_file = os.path.join(self.work_dir, self.YAML_FILE)
        self._substitute_yaml(yaml_template, self.yaml_file)

        super(simulation.Simulation, self).__init__()

    def n_ops_estimate(self):
        return self.mesh_step

    def _make_mesh(self, geo_file, mesh_file):
        """
        Make the mesh, mesh_file: <geo_base>_step.msh.
        Make substituted yaml: <yaml_base>_step.yaml,
        using common fields_step.msh file for generated fields.
        :return:
        """
        subprocess.call([self.env['gmsh'], "-2", '-clscale', str(self.mesh_step), '-o', mesh_file, geo_file])

        mesh = gmsh_io.GmshIO(mesh_file)
        n_ele = len(mesh.elements)
        self.points = np.zeros((n_ele, 2))
        self.ele_ids = np.zeros(n_ele)
        i = 0
        for id, el in mesh.elements.items():
            type, tags, i_nodes = el
            self.points[i] = np.average(np.array([mesh.nodes[i_node] for i_node in i_nodes]), axis=0)[0:2]
            self.ele_ids[i] = id
            i += 1

    def _substitute_yaml(self, yaml_tmpl, yaml_out):
        """
        Create substituted YAML file from the tamplate.
        :return:
        """
        param_dict = {}
        field_tmpl = "!FieldElementwise {gmsh_file: \"${INPUT}/%s\", field_name: %s}"
        for field in self.fields.names():
            param_dict[field] = field_tmpl % (self.FIELDS_FILE, field)
        param_dict[self.MESH_FILE_PLACEHOLDER] = self.mesh_file
        substitute_placeholders(yaml_tmpl, yaml_out, param_dict)

    def set_previous_fine_sim(self, coarse_sim):
        """
        Set coarse simulation ot the fine simulation so that the fine can generate the
        correlated input data sample for both.

        Here in particular set_points to the field generator
        :param coarse_sim
        """
        coarse_centers = coarse_sim.points
        both_centers = np.concatenate((self.points, coarse_centers), axis=0)
        self.n_fine_elements = self.points.shape[0]
        self.fields.set_points(both_centers)

    # Needed by Level
    def generate_random_sample(self):
        """
        Generate random field, both fine and coarse part.
        Store them separeted.
        :return:
        """
        # assert self._is_fine_sim
        fields_sample = self.fields.sample()
        self._input_sample = {name: values[:self.n_fine_elements] for name, values in fields_sample.items()}
        self._coarse_sample = {name: values[self.n_fine_elements:] for name, values in fields_sample.items()}

    def get_coarse_sample(self):
        """
        Return coarse part of generated field.
        :return:
        """
        return self._coarse_sample

    def cycle(self, sample_tag):
        """
        Evaluate model using generated or set input data sample.
        :param sample_tag: A unique ID used as work directory of the single simulation run.
        :return: run_token, a dict representing executed simulation. This is passed to extract_result
        to get the result or None if simulation is not finished yet.
        TODO:
        - different mesh and yaml files for individual levels/fine/coarse
        - reuse fine mesh from previous level as coarse mesh

        1. create work dir
        2. write input sample there
        3. call flow through PBS or a script that mark the folder when done
        """
        out_subdir = os.path.join("samples", str(sample_tag))
        sample_dir = os.path.join(self.work_dir, out_subdir)
        force_mkdir(sample_dir)
        fields_file = os.path.join(sample_dir, self.FIELDS_FILE)

        gmsh_io.GmshIO().write_fields(fields_file, self.ele_ids, self._input_sample)

        if self.pbs_creater.steps_sum <= 0:
            self.pbs_creater.execute(self.work_dir)

        self.pbs_creater.add_realization(output_subdir=out_subdir,
                                         work_dir=self.work_dir,
                                         flow123d=self.env['flow123d'])
        self.pbs_creater.steps_sum -= len(self.points)
        return sample_tag, sample_dir

    def extract_result(self, run_token):
        """
        Extract the observed value from the Flow123d output.
        Get sample from the field restriction, write to the GMSH file, call flow.
        :param fields:
        :return:

        TODO: Pass an extraction function as other FlowSim parameter. This function will take the
        balance data and retun observed values.
        """
        sample_tag, sample_dir = run_token

        if os.path.exists(os.path.join(sample_dir, "FINISHED")):

            # extract the flux
            balance_file = os.path.join(sample_dir, "water_balance.yaml")
            with open(balance_file, "r") as f:
                balance = yaml.load(f)

            # TODO: we need to move this part out of the library as soon as possible
            # it has to be changed for every new input file or different observation.
            # However in Analysis it is already done in general way.
            flux_regions = ['.bc_outflow']
            total_flux = 0.0
            found = False
            for flux_item in balance['data']:
                if flux_item['time'] > 0:
                    break

                if flux_item['region'] in flux_regions:
                    flux = float(flux_item['data'][0])
                    flux_in = float(flux_item['data'][1])
                    if flux_in > 1e-10:
                        raise Exception("Possitive inflow at outlet region.")
                    total_flux += flux  # flux field
                    found = True

            if not found:
                raise Exception("Observation region not found.")
            return total_flux
        else:
            return None

