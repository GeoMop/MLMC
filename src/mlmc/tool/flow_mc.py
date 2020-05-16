import os
import os.path
import subprocess
import numpy as np
import json
import glob
import shutil
import copy
import yaml
from typing import List
from mlmc.level_simulation import LevelSimulation
from mlmc.tool import gmsh_io
from mlmc.sim.simulation import Simulation
from mlmc.sim.simulation import QuantitySpec
from mlmc.random import correlated_field as cf


def create_corr_field():
    return cf.Fields([
        cf.Field('conductivity', cf.FourierSpatialCorrelatedField('gauss', dim=2, corr_length=0.125, log=True)),
    ])


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


class FlowSim(Simulation):
    # placeholders in YAML
    total_sim_id = 0
    MESH_FILE_VAR = 'mesh_file'
    # Timestep placeholder given as O(h), h = mesh step
    TIMESTEP_H1_VAR = 'timestep_h1'
    # Timestep placeholder given as O(h^2), h = mesh step
    TIMESTEP_H2_VAR = 'timestep_h2'

    # files
    GEO_FILE = 'mesh.geo'
    MESH_FILE = 'mesh.msh'
    YAML_TEMPLATE = 'flow_input.yaml.tmpl'
    YAML_FILE = 'flow_input.yaml'
    FIELDS_FILE = 'fields_sample.msh'

    """
    Gather data for single flow call (coarse/fine)
    """

    def __init__(self, mesh_step_range, config=None, clean=False):
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
        to 'self' (Sim_c_l+1) as a coarse simulation. Usually Sim_f_l and Sim_c_l+1 are same simulations, but
        these need to be different for advanced generation of samples (zero-mean control and antithetic).
        """
        self.need_workspace = True

        self.env = config['env']

        # self.field_config = config['field_name']
        #self._fields_inititialied = False
        self._fields = create_corr_field()#copy.deepcopy(config['fields'])
        self.time_factor = config.get('time_factor', 1.0)
        self.base_yaml_file = config['yaml_file']
        self.base_geo_file = config['geo_file']
        self.field_template = config.get('field_template',
                                         "!FieldElementwise {mesh_data_file: $INPUT_DIR$/%s, field_name: %s}")
        self.flow123d = config['flow123d']
        self.gmsh = config['gmsh']

        self.work_dir = config['work_dir']

        self.step_range = mesh_step_range
        self.clean = clean

        # Set in _make_mesh
        # self.points = None
        # # Element centers of computational mesh.
        # self.ele_ids = None
        # # Element IDs of computational mesh.
        # self.n_fine_elements = 0
        # # Fields samples
        # self._input_sample = {}

        # Prepare base workdir for this mesh_step
        #self.work_dir = config['work_dir']
        # self.work_dir = os.path.join(output_dir, 'sim_%d_step_%f' % (self.sim_id, self.step))

        super(Simulation, self).__init__()

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        :param fine_level_params:
        :param coarse_level_params:
        :return: LevelSimulation object
        """
        fine_step = fine_level_params[0]
        coarse_step = coarse_level_params[0]
        # TODO: determine minimal element from mesh
        self.time_step_h1 = self.time_factor * fine_step
        self.time_step_h2 = self.time_factor * fine_step * fine_step

        common_files_dir = os.path.join(self.work_dir, "l_step_{}_common_files".format(fine_step))
        force_mkdir(common_files_dir, True)

        self.mesh_file = os.path.join(common_files_dir, self.MESH_FILE)

        # self.coarse_sim = None
        # self.coarse_sim_set = False

        if self.clean:
            # Prepare mesh
            geo_file = os.path.join(common_files_dir, self.GEO_FILE)
            shutil.copyfile(self.base_geo_file, geo_file)

            # Common computational mesh for all samples.
            self._make_mesh(geo_file, self.mesh_file, fine_step)

            # Prepare main input YAML
            yaml_template = os.path.join(common_files_dir, self.YAML_TEMPLATE)
            shutil.copyfile(self.base_yaml_file, yaml_template)
            self.yaml_file = os.path.join(common_files_dir, self.YAML_FILE)
            self._substitute_yaml(yaml_template, self.yaml_file)

        fine_mesh_data = self.extract_mesh(self.mesh_file)

        coarse_mesh_file = None
        if coarse_step != 0:
            coarse_sim_common_files_dir = os.path.join(self.work_dir, "l_step_{}_common_files".format(coarse_step))
            coarse_mesh_file = os.path.join(coarse_sim_common_files_dir, self.MESH_FILE)
            #coarse_mesh_data = self.extract_mesh(coarse_mesh_file)

        #self._make_fields(fine_mesh_data, coarse_mesh_file)

        # FlowSim.make_fields(fields, coarse_step, fine_points, coarse_points, fine_point_region_ids, coarse_point_region_ids,
        #             fine_region_map, coarse_region_map)

        config = dict()
        config["fine"] = {}
        config["coarse"] = {}
        config["fine"]["step"] = fine_step
        config["coarse"]["step"] = coarse_step
        config["fields_used_params"] = self._fields_used_params
        config["coarse_mesh_file"] = coarse_mesh_file
        config["gmsh"] = self.env['gmsh']
        config["flow123d"] = self.env['flow123d']

        common_files = [geo_file, self.mesh_file, yaml_template, self.yaml_file]

        return LevelSimulation(config_dict=config, task_size=len(fine_mesh_data['points']), calculate=FlowSim.calculate,
                               common_files=common_files)

    @staticmethod
    def make_fields(fields, fine_mesh_data, coarse_mesh_data):
        if coarse_mesh_data is None:
            fields.set_points(fine_mesh_data['points'], fine_mesh_data['point_region_ids'],
                                    fine_mesh_data['region_map'])
        else:
            coarse_centers = coarse_mesh_data['points']
            both_centers = np.concatenate((fine_mesh_data['points'], coarse_centers), axis=0)
            both_regions_ids = np.concatenate((fine_mesh_data['point_region_ids'], coarse_mesh_data['point_region_ids']))
            assert fine_mesh_data['region_map'] == coarse_mesh_data['region_map']
            fields.set_points(both_centers, both_regions_ids, fine_mesh_data['region_map'])

        return fields

    @staticmethod
    def calculate(config, seed):
        """
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration
        :return:
        """
        print("os.getcwd() ", os.getcwd())
        #fields = config["fields"]

        # Init correlation field objects
        fields = create_corr_field()
        fields.set_outer_fields(config["fields_used_params"])

        coarse_step = config["coarse"]["step"]
        flow123d = config["flow123d"]

        fine_mesh_data = FlowSim.extract_mesh(FlowSim.MESH_FILE)
        print("fine mesh data ", fine_mesh_data)

        coarse_mesh_data = None
        if coarse_step != 0:
            coarse_mesh_data = FlowSim.extract_mesh(config["coarse_mesh_file"])

        fields = FlowSim.make_fields(fine_mesh_data, coarse_mesh_data)

        np.random.seed(seed)
        fine_input_sample, coarse_input_sample = FlowSim.generate_random_sample(fields, coarse_step=coarse_step,
                                                                                n_fine_elements=len(fine_mesh_data['points']))

        #out_subdir = os.path.join("samples", str(sample_tag))
        #sample_dir = os.path.join(self.work_dir, out_subdir)

        #force_mkdir(sample_dir, True)
        # PbsJob changes sample directory (see PbsJob.calculate_samples())
        fields_file = os.path.join(os.getcwd(), FlowSim.FIELDS_FILE)

        gmsh_io.GmshIO().write_fields(fields_file, fine_mesh_data['ele_ids'], fine_input_sample)

        subprocess.call([flow123d, "--yaml_balance", '-i', os.getcwd(), '-s', "{}/flow_input.yaml".format(os.getcwd()),
                         "-o", os.getcwd(), ">{}/flow.out".format(os.getcwd())])

        sample_dir = os.getcwd()
        res,time = FlowSim._extract_result(sample_dir)
        print("Fine res: {}, time:{}".format(res, time))

        # Rename fields_sample.msh to fine_fields_sample.msh
        subprocess.call(["mv", fields_file, os.path.join(os.getcwd(), "fine_" + FlowSim.FIELDS_FILE)])

        gmsh_io.GmshIO().write_fields(fields_file, coarse_mesh_data['ele_ids'], coarse_input_sample)

        subprocess.call([flow123d, "--yaml_balance", '-i', os.getcwd(), '-s', "{}/flow_input.yaml".format(os.getcwd()),
                         "-o", os.getcwd(), ">{}/flow.out".format(os.getcwd())])


        res, time = FlowSim._extract_result(sample_dir)
        print("Fine res: {}, time:{}".format(res, time))

        # if np.isnan(fine_result) or np.isnan(coarse_result):
        #     raise Exception("result is nan")

        # quantity_format = FlowSimSimulation.result_format()
        #
        # results = []
        # for result in [fine_result, coarse_result]:
        #     quantities = []
        #     for quantity in quantity_format:
        #         locations = np.array([result + i for i in range(len(quantity.locations))])
        #         times = np.array([locations for _ in range(len(quantity.times))])
        #         quantities.append(times)
        #
        #     results.append(np.array(quantities))
        #
        # return results[0].flatten(), results[1].flatten()

    @staticmethod
    def generate_random_sample(fields, coarse_step, n_fine_elements):
        """
        Generate random field, both fine and coarse part.
        Store them separeted.
        :return:
        """
        # assert self._is_fine_sim

        fields_sample = fields.sample()
        fine_input_sample = {name: values[:n_fine_elements, None] for name, values in fields_sample.items()}
        if coarse_step != 0:
            coarse_input_sample = {name: values[n_fine_elements:, None] for name, values in
                                             fields_sample.items()}

        return fine_input_sample, coarse_input_sample

    def _make_mesh(self, geo_file, mesh_file, fine_step):
        """
        Make the mesh, mesh_file: <geo_base>_step.msh.
        Make substituted yaml: <yaml_base>_step.yaml,
        using common fields_step.msh file for generated fields.
        :return:
        """
        subprocess.call([self.env['gmsh'], "-2", '-clscale', str(fine_step), '-o', mesh_file, geo_file])

    @staticmethod
    def extract_mesh(mesh_file):
        """
        Extract mesh from file
        :param mesh_file: Mesh file path
        :return: None
        """
        mesh = gmsh_io.GmshIO(mesh_file)
        is_bc_region = {}
        region_map = {}
        for name, (id, _) in mesh.physical.items():
            unquoted_name = name.strip("\"'")
            is_bc_region[id] = (unquoted_name[0] == '.')
            region_map[unquoted_name] = id

        bulk_elements = []
        for id, el in mesh.elements.items():
            _, tags, i_nodes = el
            region_id = tags[0]
            if not is_bc_region[region_id]:
                bulk_elements.append(id)

        n_bulk = len(bulk_elements)
        centers = np.empty((n_bulk, 3))
        ele_ids = np.zeros(n_bulk, dtype=int)
        point_region_ids = np.zeros(n_bulk, dtype=int)

        for i, id_bulk in enumerate(bulk_elements):
            _, tags, i_nodes = mesh.elements[id_bulk]
            region_id = tags[0]
            centers[i] = np.average(np.array([mesh.nodes[i_node] for i_node in i_nodes]), axis=0)
            point_region_ids[i] = region_id
            ele_ids[i] = id_bulk

        min_pt = np.min(centers, axis=0)
        max_pt = np.max(centers, axis=0)
        diff = max_pt - min_pt
        min_axis = np.argmin(diff)
        non_zero_axes = [0, 1, 2]
        # TODO: be able to use this mesh_dimension in fields
        if diff[min_axis] < 1e-10:
            non_zero_axes.pop(min_axis)
        points = centers[:, non_zero_axes]

        return {'points': points, 'point_region_ids': point_region_ids, 'ele_ids': ele_ids, 'region_map':region_map}

    def _substitute_yaml(self, yaml_tmpl, yaml_out):
        """
        Create substituted YAML file from the tamplate.
        :return:
        """
        param_dict = {}
        field_tmpl = self.field_template
        for field_name in self._fields.names:
            param_dict[field_name] = field_tmpl % (self.FIELDS_FILE, field_name)
        param_dict[self.MESH_FILE_VAR] = self.mesh_file
        param_dict[self.TIMESTEP_H1_VAR] = self.time_step_h1
        param_dict[self.TIMESTEP_H2_VAR] = self.time_step_h2
        used_params = substitute_placeholders(yaml_tmpl, yaml_out, param_dict)

        self._fields_used_params = used_params

    @staticmethod
    def make_fields(fields, coarse_step, fine_points, coarse_points, fine_point_region_ids, coarse_point_region_ids,
                    fine_region_map, coarse_region_map):
        # One level Monte Carlo
        if coarse_step == 0:
            fields.set_points(fine_points, fine_point_region_ids, fine_region_map)
        else:
            coarse_centers = coarse_points
            both_centers = np.concatenate((fine_points, coarse_centers), axis=0)
            both_regions_ids = np.concatenate((fine_point_region_ids, coarse_point_region_ids))
            assert fine_region_map == coarse_region_map
            fields.set_points(both_centers, both_regions_ids, fine_region_map)

    @staticmethod
    def _extract_result(sample_dir):
        """
        Extract the observed value from the Flow123d output.
        :param sample: str, path to sample directory
        :return: None, inf or water balance result (float) and overall sample time
        """
        if os.path.exists(os.path.join(sample_dir, "FINISHED")):
            # try:
            # extract the flux
            balance_file = os.path.join(sample_dir, "water_balance.yaml")

            with open(balance_file, "r") as f:
                balance = yaml.load(f)

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

            # Get flow123d computing time
            run_time = FlowSim.get_run_time(sample_dir)

            if not found:
                raise Exception
            return -total_flux, run_time
        else:
            return None, 0

    @staticmethod
    def result_format() -> List[QuantitySpec]:
        """
        Result format
        :return:
        """
        spec1 = QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1, 2, 3], locations=['10', '20'])
        spec2 = QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=['30', '40'])
        # spec1 = QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1, 2, 3], locations=[(1, 2, 3), (4, 5, 6)])
        # spec2 = QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=[(7, 8, 9), (10, 11, 12)])
        return [spec1, spec2]

    @staticmethod
    def get_run_time(sample_dir):
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


