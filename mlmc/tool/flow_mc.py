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
    Create correlated random-field provider (cf.Fields) according to selected backend.

    :param model: One of 'fourier', 'svd', 'exp', 'TPLgauss', 'TPLexp', 'TPLStable', or others (defaults to 'gauss').
    :param corr_length: Correlation length (used by GSTools or SVD implementations).
    :param dim: Spatial dimension of the field (1, 2 or 3).
    :param log: If True, generate log-normal field (exponentiate underlying Gaussian field).
    :param sigma: Standard deviation for the generated field.
    :param mode_no: Number of Fourier modes
    :return: cf.Fields instance that can generate random field samples.
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
    Replace placeholders of form '<name>' in a template file with corresponding values.

    :param file_in: Path to the template file containing placeholders.
    :param file_out: Path where the substituted output will be written.
    :param params: Dictionary mapping placeholder names to replacement values, e.g. {'mesh_file': 'mesh.msh'}.
    :return: List of parameter names that were actually used (replaced) in the template.
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
    Create directory tree; optionally remove existing leaf directory first.

    :param path: Directory path to create (parents created as needed).
    :param force: If True and the directory already exists, remove it (recursively) before creating.
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
    TIMESTEP_H1_VAR = 'timestep_h1'  # O(h)
    TIMESTEP_H2_VAR = 'timestep_h2'  # O(h^2)

    # filenames used in workspace and job directories
    GEO_FILE = 'mesh.geo'
    MESH_FILE = 'mesh.msh'
    YAML_TEMPLATE = 'flow_input.yaml.tmpl'
    YAML_FILE = 'flow_input.yaml'
    FIELDS_FILE = 'fields_sample.msh'

    def __init__(self, config=None, clean=None):
        """
        Initialize FlowSim instance that runs flow123d simulations using generated random fields.

        :param config: Dict with keys:
            - env: dict of environment executables (flow123d, gmsh, gmsh_version, etc.)
            - fields_params: parameters forwarded to create_corr_field
            - yaml_file: base YAML template path
            - geo_file: geometry (.geo) file path
            - work_dir: base working directory for generated level common files
            - field_template: optional template string for field definition in YAML
            - time_factor: optional multiplier for timestep selection (default 1.0)
        :param clean: If True, regenerate common files (mesh, yaml) for the given level.
        """
        self.need_workspace = True  # this simulation needs per-sample work directories
        self.env = config['env']
        self._fields_params = config['fields_params']
        self._fields = create_corr_field(**config['fields_params'])
        self._fields_used_params = None
        self.time_factor = config.get('time_factor', 1.0)
        self.base_yaml_file = config['yaml_file']
        self.base_geo_file = config['geo_file']
        self.field_template = config.get('field_template',
                                         "!FieldElementwise {mesh_data_file: $INPUT_DIR$/%s, field_name: %s}")
        self.work_dir = config['work_dir']
        self.clean = clean

        super(Simulation, self).__init__()  # keep compatibility with parent initialization


    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Create a LevelSimulation object for given fine/coarse steps.

        This method is called in the main process (Sampler) and must prepare
        common files (mesh, YAML) for that level. The returned LevelSimulation
        is serialized and sent to PBS jobs (PbsJob) for actual execution.

        :param fine_level_params: list with single element [fine_step] (mesh step)
        :param coarse_level_params: list with single element [coarse_step] (mesh step) or [0] for one-level MC
        :return: LevelSimulation configured with task size and calculate method
        """
        fine_step = fine_level_params[0]
        coarse_step = coarse_level_params[0]

        # Set time steps used in YAML substitution (O(h) and O(h^2) placeholders)
        self.time_step_h1 = self.time_factor * fine_step
        self.time_step_h2 = self.time_factor * fine_step * fine_step

        # Directory to store files common to all samples at this fine level
        common_files_dir = os.path.join(self.work_dir, "l_step_{}_common_files".format(fine_step))
        force_mkdir(common_files_dir, force=self.clean)

        self.mesh_file = os.path.join(common_files_dir, self.MESH_FILE)

        if self.clean:
            # Create computational mesh from geometry template
            geo_file = os.path.join(common_files_dir, self.GEO_FILE)
            shutil.copyfile(self.base_geo_file, geo_file)
            self._make_mesh(geo_file, self.mesh_file, fine_step)

            # Prepare main YAML by substituting placeholders
            yaml_template = os.path.join(common_files_dir, self.YAML_TEMPLATE)
            shutil.copyfile(self.base_yaml_file, yaml_template)
            yaml_file = os.path.join(common_files_dir, self.YAML_FILE)
            self._substitute_yaml(yaml_template, yaml_file)

        # Extract mesh metadata to determine task_size (number of points affects job weight)
        fine_mesh_data = self.extract_mesh(self.mesh_file)

        # Set coarse sim common files dir if coarse level exists
        coarse_sim_common_files_dir = None
        if coarse_step != 0:
            coarse_sim_common_files_dir = os.path.join(self.work_dir, "l_step_{}_common_files".format(coarse_step))

        # Prepare configuration dict that will be serialized in LevelSimulation
        config = dict()
        config["fine"] = {}
        config["coarse"] = {}
        config["fine"]["step"] = fine_step
        config["coarse"]["step"] = coarse_step
        config["fine"]["common_files_dir"] = common_files_dir
        config["coarse"]["common_files_dir"] = coarse_sim_common_files_dir

        config["fields_used_params"] = self._fields_used_params
        config["gmsh"] = self.env['gmsh']
        config["flow123d"] = self.env['flow123d']
        config['fields_params'] = self._fields_params

        # job_weight is used to convert mesh size into a normalized task_size
        job_weight = 17000000

        return LevelSimulation(config_dict=config,
                               task_size=len(fine_mesh_data['points']) / job_weight,
                               calculate=FlowSim.calculate,
                               need_sample_workspace=True
                               )

    @staticmethod
    def calculate(config, seed):
        """
        Execute one MLMC sample calculation (fine and optional coarse) inside PBS job.

        :param config: Configuration dict from LevelSimulation.config_dict (contains common_files dirs, steps, fields params)
        :param seed: Random seed for the sample generation (derived from sample id)
        :return: Tuple (fine_result_array, coarse_result_array), both numpy arrays (coarse may be zeros for one-level MC)
        """
        # Initialize fields object in the worker process
        fields = create_corr_field(**config['fields_params'])
        fields.set_outer_fields(config["fields_used_params"])

        coarse_step = config["coarse"]["step"]
        flow123d = config["flow123d"]

        # Extract fine mesh structure and optionally coarse mesh structure
        fine_common_files_dir = config["fine"]["common_files_dir"]
        fine_mesh_data = FlowSim.extract_mesh(os.path.join(fine_common_files_dir, FlowSim.MESH_FILE))

        coarse_mesh_data = None
        coarse_common_files_dir = None
        if coarse_step != 0:
            coarse_common_files_dir = config["coarse"]["common_files_dir"]
            coarse_mesh_data = FlowSim.extract_mesh(os.path.join(coarse_common_files_dir, FlowSim.MESH_FILE))

        # Prepare combined fields object that has points for both fine and coarse meshes
        fields = FlowSim.make_fields(fields, fine_mesh_data, coarse_mesh_data)

        # Sample random field realizations reproducibly
        np.random.seed(seed)
        fine_input_sample, coarse_input_sample = FlowSim.generate_random_sample(
            fields, coarse_step=coarse_step, n_fine_elements=len(fine_mesh_data['points'])
        )

        # Run fine-level simulation
        fields_file = os.path.join(os.getcwd(), FlowSim.FIELDS_FILE)
        fine_res = FlowSim._run_sample(fields_file, fine_mesh_data['ele_ids'], fine_input_sample, flow123d,
                                       fine_common_files_dir)

        # Move generated files to have 'fine_' prefix so they don't collide
        for filename in os.listdir(os.getcwd()):
            if not filename.startswith("fine"):
                shutil.move(os.path.join(os.getcwd(), filename), os.path.join(os.getcwd(), "fine_" + filename))

        # Run coarse-level simulation if coarse sample exists
        coarse_res = np.zeros(len(fine_res))
        if coarse_input_sample:
            coarse_res = FlowSim._run_sample(fields_file, coarse_mesh_data['ele_ids'], coarse_input_sample, flow123d,
                                             coarse_common_files_dir)

        return fine_res, coarse_res

    @staticmethod
    def make_fields(fields, fine_mesh_data, coarse_mesh_data):
        """
        Assign evaluation points to fields and return the Fields object prepared for sampling.

        :param fields: correlated_field.Fields instance (with local field definitions)
        :param fine_mesh_data: Dict returned by extract_mesh() for the fine mesh
        :param coarse_mesh_data: Dict returned by extract_mesh() for the coarse mesh (or None for one-level)
        :return: the same cf.Fields object with points set for sampling
        """
        # If no coarse mesh, just register fine mesh points
        if coarse_mesh_data is None:
            fields.set_points(fine_mesh_data['points'], fine_mesh_data['point_region_ids'],
                              fine_mesh_data['region_map'])
        else:
            # Concatenate fine and coarse points to compute joint fields (ensures consistent sampling)
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
        Write random fields to Gmsh file, call flow123d, and extract sample results.

        :param fields_file: Path where fields will be written (in current working directory)
        :param ele_ids: Array of element ids for which field values are provided
        :param fine_input_sample: Dict mapping field names to arrays of shape (n_elements, 1)
        :param flow123d: Path/command to flow123d executable
        :param common_files_dir: Directory containing common YAML and other input files for the level
        :return: numpy.ndarray with extracted simulation result (e.g., water balance)
        """
        gmsh_io.GmshIO().write_fields(fields_file, ele_ids, fine_input_sample)

        subprocess.call(
            [flow123d, "--yaml_balance", '-i', os.getcwd(), '-s', "{}/flow_input.yaml".format(common_files_dir),
             "-o", os.getcwd(), ">{}/flow.out".format(os.getcwd())])

        return FlowSim._extract_result(os.getcwd())

    @staticmethod
    def generate_random_sample(fields, coarse_step, n_fine_elements):
        """
        Generate random field samples for the fine and (optionally) coarse meshes.

        :param fields: cf.Fields object (already configured with points)
        :param coarse_step: coarse-level step (0 for no coarse sample)
        :param n_fine_elements: Number of elements that belong to fine mesh (used to split combined sample)
        :return: Tuple (fine_input_sample: dict, coarse_input_sample: dict)
                 Each dict maps field name -> array shaped (n_elements, 1).
        """
        fields_sample = fields.sample()
        # Fine inputs are first n_fine_elements rows; coarse are the remainder (if any)
        fine_input_sample = {name: values[:n_fine_elements, None] for name, values in fields_sample.items()}
        coarse_input_sample = {}
        if coarse_step != 0:
            coarse_input_sample = {name: values[n_fine_elements:, None] for name, values in
                                   fields_sample.items()}

        return fine_input_sample, coarse_input_sample

    def _make_mesh(self, geo_file, mesh_file, fine_step):
        """
        Invoke Gmsh to produce a mesh with the requested geometric scale (clscale).

        :param geo_file: Path to the .geo file used to generate the mesh
        :param mesh_file: Path where the .msh output will be written
        :param fine_step: Mesh step (controls element size via -clscale)
        :return: None
        """
        if self.env['gmsh_version'] == 2:
            subprocess.call(
                [self.env['gmsh'], "-2", '-format', 'msh2', '-clscale', str(fine_step), '-o', mesh_file, geo_file])
        else:
            subprocess.call([self.env['gmsh'], "-2", '-clscale', str(fine_step), '-o', mesh_file, geo_file])

    @staticmethod
    def extract_mesh(mesh_file):
        """
        Parse a Gmsh mesh file and extract points (element centers), element ids and region mapping.

        :param mesh_file: Path to .msh file to parse (Gmsh 2/4 depending on GmshIO implementation)
        :return: Dict with keys:
                 - 'points': np.ndarray of shape (n_elements, dim) with element center coordinates
                 - 'point_region_ids': np.ndarray of region id per element
                 - 'ele_ids': np.ndarray of original element ids
                 - 'region_map': dict mapping region name -> region id
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
        # If mesh is effectively 2D (one axis collapsed), remove that axis from point coordinates
        if diff[min_axis] < 1e-10:
            non_zero_axes.pop(min_axis)
        points = centers[:, non_zero_axes]

        return {'points': points, 'point_region_ids': point_region_ids, 'ele_ids': ele_ids, 'region_map': region_map}

    def _substitute_yaml(self, yaml_tmpl, yaml_out):
        """
        Build YAML input file for flow123d by substituting placeholders for mesh and fields.

        :param yaml_tmpl: Path to YAML template with placeholders like '<mesh_file>' and '<FIELDNAME>'.
        :param yaml_out: Path to output YAML file that will be used by flow123d.
        :return: None (also populates self._fields_used_params with names of substituted fields)
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
        Extract the observed quantity (e.g., water balance flux) from a flow123d run directory.

        :param sample_dir: Directory where flow123d output (water_balance.yaml) is located.
        :return: numpy.ndarray with a single value [-total_flux] representing outflow (negative sign).
                 Raises Exception if expected data is not found or inflow at outlet is positive.
        """
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
                total_flux += flux
                found = True

        if not found:
            raise Exception("No outlet flux found in water_balance.yaml")
        return np.array([-total_flux])

    @staticmethod
    def result_format() -> List[QuantitySpec]:
        """
        Describe the simulation output format as a list of QuantitySpec objects.

        :return: List[QuantitySpec] describing each output quantity (name, unit, shape, times, locations)
        """
        spec1 = QuantitySpec(name="conductivity", unit="m", shape=(1, 1), times=[1], locations=['0'])
        return [spec1]
