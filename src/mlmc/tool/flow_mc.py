import os
import os.path
import subprocess
import numpy as np
import shutil
import ruamel.yaml as yaml
from typing import List
import gstools
from mlmc.level_simulation import LevelSimulation
from mlmc.tool import gmsh_io
from mlmc.sim.simulation import Simulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.random import correlated_field as cf


def create_corr_field(model='gauss', corr_length=0.125, dim=2, log=True, sigma=1, mode_no=1000):
    """
    Create random fields
    :return:
    """
    if model == 'fourier':
        return cf.Fields([
            cf.Field('conductivity', cf.FourierSpatialCorrelatedField('gauss', dim=dim,
                                                                      corr_length=corr_length,
                                                                      log=log, sigma=sigma)),
        ])

    elif model == 'svd':
        conductivity = dict(
            mu=0.0,
            sigma=sigma,
            corr_exp='exp',
            dim=dim,
            corr_length=corr_length,
            log=log
        )
        return cf.Fields([cf.Field("conductivity", cf.SpatialCorrelatedField(**conductivity))])

    elif model == 'exp':
        model = gstools.Exponential(dim=dim, len_scale=corr_length)
    elif model == 'TPLgauss':
        model = gstools.TPLGaussian(dim=dim,  len_scale=corr_length)
    elif model == 'TPLexp':
        model = gstools.TPLExponential(dim=dim,  len_scale=corr_length)
    elif model == 'TPLStable':
        model = gstools.TPLStable(dim=dim,  len_scale=corr_length)
    else:
        model = gstools.Gaussian(dim=dim,  len_scale=corr_length)

    return cf.Fields([
        cf.Field('conductivity', cf.GSToolsSpatialCorrelatedField(model, log=log, sigma=sigma, mode_no=mode_no)),
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

    Usage:
    mlmc.sampler.Sampler uses instance of FlowSim, it calls once level_instance() for each level step (The level_instance() method
     is called as many times as the number of levels), it takes place in main process

    mlmc.tool.pbs_job.PbsJob uses static methods in FlowSim, it calls calculate(). That's where the calculation actually runs,
    it takes place in PBS process
       It also extracts results and passes them back to PbsJob, which handles the rest 

    """

    def __init__(self, config=None, clean=None):
        """
        Simple simulation using flow123d
        :param config: configuration of the simulation, processed keys:
            env - Environment object.
            fields - FieldSet object
            yaml_file: Template for main input file. Placeholders:
                <mesh_file> - replaced by generated mesh
                <FIELD> - for FIELD be name of any of `fields`, replaced by the FieldElementwise field with generated
                 field input file and the field name for the component.
            geo_file: Path to the geometry file.
        :param clean: bool, if True remove existing simulation files - mesh files, ...
        """
        self.need_workspace = True
        # This simulation requires workspace
        self.env = config['env']
        # Environment variables, flow123d, gmsh, ...
        self._fields_params = config['fields_params']
        self._fields = create_corr_field(**config['fields_params'])
        self._fields_used_params = None
        # Random fields instance
        self.time_factor = config.get('time_factor', 1.0)
        # It is used for minimal element from mesh determination (see level_instance method)

        self.base_yaml_file = config['yaml_file']
        self.base_geo_file = config['geo_file']
        self.field_template = config.get('field_template',
                                         "!FieldElementwise {mesh_data_file: $INPUT_DIR$/%s, field_name: %s}")
        self.work_dir = config['work_dir']
        self.clean = clean

        super(Simulation, self).__init__()

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Called from mlmc.Sampler, it creates single instance of LevelSimulation (mlmc.)
        :param fine_level_params: in this version, it is just fine simulation step
        :param coarse_level_params: in this version, it is just coarse simulation step
        :return: mlmc.LevelSimulation object, this object is serialized in SamplingPoolPbs and deserialized in PbsJob,
         so it allows pass simulation data from main process to PBS process
        """
        fine_step = fine_level_params[0]
        coarse_step = coarse_level_params[0]

        # TODO: determine minimal element from mesh
        self.time_step_h1 = self.time_factor * fine_step
        self.time_step_h2 = self.time_factor * fine_step * fine_step

        # Set fine simulation common files directory
        # Files in the directory are used by each simulation at that level
        common_files_dir = os.path.join(self.work_dir, "l_step_{}_common_files".format(fine_step))
        force_mkdir(common_files_dir, force=self.clean)

        self.mesh_file = os.path.join(common_files_dir, self.MESH_FILE)

        if self.clean:
            # Prepare mesh
            geo_file = os.path.join(common_files_dir, self.GEO_FILE)
            shutil.copyfile(self.base_geo_file, geo_file)
            self._make_mesh(geo_file, self.mesh_file, fine_step)  # Common computational mesh for all samples.

            # Prepare main input YAML
            yaml_template = os.path.join(common_files_dir, self.YAML_TEMPLATE)
            shutil.copyfile(self.base_yaml_file, yaml_template)
            yaml_file = os.path.join(common_files_dir, self.YAML_FILE)
            self._substitute_yaml(yaml_template, yaml_file)

        # Mesh is extracted because we need number of mesh points to determine task_size parameter (see return value)
        fine_mesh_data = self.extract_mesh(self.mesh_file)

        # Set coarse simulation common files directory
        # Files in the directory are used by each simulation at that level
        coarse_sim_common_files_dir = None
        if coarse_step != 0:
            coarse_sim_common_files_dir = os.path.join(self.work_dir, "l_step_{}_common_files".format(coarse_step))

        # Simulation config
        # Configuration is used in mlmc.tool.pbs_job.PbsJob instance which is run from PBS process
        # It is part of LevelSimulation which is serialized and then deserialized in mlmc.tool.pbs_job.PbsJob
        config = dict()
        config["fine"] = {}
        config["coarse"] = {}
        config["fine"]["step"] = fine_step
        config["coarse"]["step"] = coarse_step
        config["fine"]["common_files_dir"] = common_files_dir
        config["coarse"]["common_files_dir"] = coarse_sim_common_files_dir

        config[
            "fields_used_params"] = self._fields_used_params  # Params for Fields instance, which is createed in PbsJob
        config["gmsh"] = self.env['gmsh']
        config["flow123d"] = self.env['flow123d']
        config['fields_params'] = self._fields_params

        # Auxiliary parameter which I use to determine task_size (should be from 0 to 1, if task_size is above 1 then pbs job is scheduled)
        job_weight = 17000000  # 4000000 - 20 min, 2000000 - cca 10 min

        return LevelSimulation(config_dict=config,
                               task_size=len(fine_mesh_data['points']) / job_weight,
                               calculate=FlowSim.calculate,
                               # method which carries out the calculation, will be called from PBS processs
                               need_sample_workspace=True  # If True, a sample directory is created
                               )

    @staticmethod
    def calculate(config, seed):
        """
        Method that actually run the calculation, it's called from mlmc.tool.pbs_job.PbsJob.calculate_samples()
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration, LevelSimulation.config_dict (set in level_instance)
        :param seed: random seed, int
        :return: List[fine result, coarse result], both flatten arrays (see mlmc.sim.synth_simulation.calculate())
        """
        # Init correlation field objects
        fields = create_corr_field(**config['fields_params'])  # correlated_field.Fields instance
        fields.set_outer_fields(config["fields_used_params"])

        coarse_step = config["coarse"]["step"]  # Coarse simulation step, zero if one level MC
        flow123d = config["flow123d"]  # Flow123d command

        # Extract fine mesh
        fine_common_files_dir = config["fine"]["common_files_dir"]  # Directory with fine simulation common files
        fine_mesh_data = FlowSim.extract_mesh(os.path.join(fine_common_files_dir, FlowSim.MESH_FILE))

        # Extract coarse mesh
        coarse_mesh_data = None
        coarse_common_files_dir = None
        if coarse_step != 0:
            coarse_common_files_dir = config["coarse"][
                "common_files_dir"]  # Directory with coarse simulation common files
            coarse_mesh_data = FlowSim.extract_mesh(os.path.join(coarse_common_files_dir, FlowSim.MESH_FILE))

        # Create fields both fine and coarse
        fields = FlowSim.make_fields(fields, fine_mesh_data, coarse_mesh_data)

        # Set random seed, seed is calculated from sample id, so it is not user defined
        np.random.seed(seed)
        # Generate random samples
        fine_input_sample, coarse_input_sample = FlowSim.generate_random_sample(fields, coarse_step=coarse_step,
                                                                                n_fine_elements=len(
                                                                                    fine_mesh_data['points']))

        # Run fine sample
        fields_file = os.path.join(os.getcwd(), FlowSim.FIELDS_FILE)
        fine_res = FlowSim._run_sample(fields_file, fine_mesh_data['ele_ids'], fine_input_sample, flow123d,
                                       fine_common_files_dir)

        # Rename fields_sample.msh to fine_fields_sample.msh, we might remove it
        for filename in os.listdir(os.getcwd()):
            if not filename.startswith("fine"):
                shutil.move(os.path.join(os.getcwd(), filename), os.path.join(os.getcwd(), "fine_" + filename))

        # Run coarse sample
        coarse_res = np.zeros(len(fine_res))
        if coarse_input_sample:
            coarse_res = FlowSim._run_sample(fields_file, coarse_mesh_data['ele_ids'], coarse_input_sample, flow123d,
                                             coarse_common_files_dir)

        return fine_res, coarse_res

    @staticmethod
    def make_fields(fields, fine_mesh_data, coarse_mesh_data):
        """
        Create random fields that are used by both coarse and fine simulation
        :param fields: correlated_field.Fields instance
        :param fine_mesh_data: Dict contains data extracted from fine mesh file (points, point_region_ids, region_map)
        :param coarse_mesh_data: Dict contains data extracted from coarse mesh file (points, point_region_ids, region_map)
        :return: correlated_field.Fields
        """
        # One level MC has no coarse_mesh_data
        if coarse_mesh_data is None:
            fields.set_points(fine_mesh_data['points'], fine_mesh_data['point_region_ids'],
                              fine_mesh_data['region_map'])
        else:
            coarse_centers = coarse_mesh_data['points']
            both_centers = np.concatenate((fine_mesh_data['points'], coarse_centers), axis=0)
            both_regions_ids = np.concatenate(
                (fine_mesh_data['point_region_ids'], coarse_mesh_data['point_region_ids']))
            assert fine_mesh_data['region_map'] == coarse_mesh_data['region_map']
            fields.set_points(both_centers, both_regions_ids, fine_mesh_data['region_map'])

        return fields

    @staticmethod
    def _run_sample(fields_file, ele_ids, fine_input_sample, flow123d, common_files_dir):
        """
        Create random fields file, call Flow123d and extract results
        :param fields_file: Path to file with random fields
        :param ele_ids: Element IDs in computational mesh
        :param fine_input_sample: fields: {'field_name' : values_array, ..}
        :param flow123d: Flow123d command
        :param common_files_dir: Directory with simulations common files (flow_input.yaml, )
        :return: simulation result, ndarray
        """
        gmsh_io.GmshIO().write_fields(fields_file, ele_ids, fine_input_sample)

        subprocess.call(
            [flow123d, "--yaml_balance", '-i', os.getcwd(), '-s', "{}/flow_input.yaml".format(common_files_dir),
             "-o", os.getcwd(), ">{}/flow.out".format(os.getcwd())])

        return FlowSim._extract_result(os.getcwd())

    @staticmethod
    def generate_random_sample(fields, coarse_step, n_fine_elements):
        """
        Generate random field, both fine and coarse part.
        Store them separeted.
        :return: Dict, Dict
        """
        fields_sample = fields.sample()
        fine_input_sample = {name: values[:n_fine_elements, None] for name, values in fields_sample.items()}
        coarse_input_sample = {}
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
        if self.env['gmsh_version'] == 2:
            subprocess.call(
                [self.env['gmsh'], "-2", '-format', 'msh2', '-clscale', str(fine_step), '-o', mesh_file, geo_file])
        else:
            subprocess.call([self.env['gmsh'], "-2", '-clscale', str(fine_step), '-o', mesh_file, geo_file])

    @staticmethod
    def extract_mesh(mesh_file):
        """
        Extract mesh from file
        :param mesh_file: Mesh file path
        :return: Dict
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

        return {'points': points, 'point_region_ids': point_region_ids, 'ele_ids': ele_ids, 'region_map': region_map}

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
    def _extract_result(sample_dir):
        """
        Extract the observed value from the Flow123d output.
        :param sample_dir: str, path to sample directory
        :return: None, inf or water balance result (float) and overall sample time
        """
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
        # run_time = FlowSim.get_run_time(sample_dir)

        if not found:
            raise Exception
        return np.array([-total_flux])

    @staticmethod
    def result_format() -> List[QuantitySpec]:
        """
        Define simulation result format
        :return: List[QuantitySpec, ...]
        """
        spec1 = QuantitySpec(name="conductivity", unit="m", shape=(1, 1), times=[1], locations=['0'])
        # spec2 = QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=['30', '40'])
        return [spec1]

    # @staticmethod
    # def get_run_time(sample_dir):
    #     """
    #     Get flow123d sample running time from profiler
    #     :param sample_dir: Sample directory
    #     :return: float
    #     """
    #     profiler_file = os.path.join(sample_dir, "profiler_info_*.json")
    #     profiler = glob.glob(profiler_file)[0]
    #
    #     try:
    #         with open(profiler, "r") as f:
    #             prof_content = json.load(f)
    #
    #         run_time = float(prof_content['children'][0]['cumul-time-sum'])
    #     except:
    #         print("Extract run time failed")
    #
    #     return run_time


