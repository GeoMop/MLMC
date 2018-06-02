import os
import os.path
import yaml
import subprocess
import gmsh_io
import numpy as np
import shutil
# from operator import add
import mlmc.simulation as simulation
import mlmc.correlated_field as correlated_field

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
    remove the leaf dir recoursively if it already exists.
    :param path:
    :return:
    """
    if force:
        if os.path.isdir(path):
            shutil.rmtree(path)
    os.makedirs(path, mode=0o775, exist_ok=True)


class FlowSim(simulation.Simulation):
    # placeholders in YAML
    total_sim_id = 0
    MESH_FILE_VAR = 'mesh_file'
    TIMESTEP_H1_VAR = 'timestep_h1'
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

    def __init__(self, mesh_step, config = None, clean=False, parent_fine_sim=None):
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
        self.sim_id = FlowSim.total_sim_id
        FlowSim.total_sim_id += 1
        self.env = config['env']
        #self.field_config = config['field_name']
        self._fields_inititialied = False
        self._fields = config['fields']
        self.time_factor = config.get('time_factor', 1.0)
        self.base_yaml_file = config['yaml_file']
        self.base_geo_file = config['geo_file']
        self.field_template = config.get('field_template',
                                            "!FieldElementwise {gmsh_file: \"${INPUT}/%s\", field_name: %s}")
        self.step = mesh_step
        # Pbs script creater
        self.pbs_creater = self.env["pbs"]

        # Set in _make_mesh
        self.points = None
        # Element centers of computational mesh.
        self.ele_ids = None
        # Element IDs of computational mesh.

        # TODO: determine minimal element from mesh
        self.time_step_h1 = self.time_factor * self.step
        self.time_step_h2 = self.time_factor * self.step * self.step

        # Prepare base workdir for this mesh_step
        output_dir = config['output_dir']
        self.work_dir = os.path.join(output_dir, 'sim_%d_step_%f' % (self.sim_id, self.step))
        force_mkdir(self.work_dir, clean)

        self.mesh_file = os.path.join(self.work_dir, self.MESH_FILE)
        if clean:
            # Prepare mesh
            geo_file = os.path.join(self.work_dir, self.GEO_FILE)
            shutil.copy(self.base_geo_file, geo_file)
            
            # Common computational mesh for all samples.
            self._make_mesh(geo_file, self.mesh_file)

            # Prepare main input YAML
            yaml_template = os.path.join(self.work_dir, self.YAML_TEMPLATE)
            shutil.copy(self.base_yaml_file, yaml_template)
            self.yaml_file = os.path.join(self.work_dir, self.YAML_FILE)
            self._substitute_yaml(yaml_template, self.yaml_file)
        self._extract_mesh(self.mesh_file)
            
        super(simulation.Simulation, self).__init__()

    def n_ops_estimate(self):
        return self.n_fine_elements

    def _make_mesh(self, geo_file, mesh_file):
        """
        Make the mesh, mesh_file: <geo_base>_step.msh.
        Make substituted yaml: <yaml_base>_step.yaml,
        using common fields_step.msh file for generated fields.
        :return:
        """
        subprocess.call([self.env['gmsh'], "-2", '-clscale', str(self.step), '-o', mesh_file, geo_file])

    def _extract_mesh(self, mesh_file):
    
        mesh = gmsh_io.GmshIO(mesh_file)
        is_bc_region = {}
        self.region_map = {}
        for name, (id, dim) in mesh.physical.items():
            unquoted_name = name.strip("\"'")
            is_bc_region[id] = (unquoted_name[0] == '.')
            self.region_map[unquoted_name] = id
         

        bulk_elements = []
        for id, el in mesh.elements.items():
            type, tags, i_nodes = el
            region_id = tags[0]
            if not is_bc_region[region_id]:
                bulk_elements.append(id)

        n_bulk = len(bulk_elements)
        centers = np.empty((n_bulk, 3))
        self.ele_ids = np.zeros(n_bulk, dtype=int)
        self.point_region_ids = np.zeros(n_bulk, dtype=int)

        for i, id_bulk in enumerate(bulk_elements):
            type, tags, i_nodes = mesh.elements[id_bulk]
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

    def _substitute_yaml(self, yaml_tmpl, yaml_out):
        """
        Create substituted YAML file from the tamplate.
        :return:
        """
        param_dict = {}
        #field_tmpl = "!FieldElementwise {gmsh_file: \"${INPUT}/%s\", field_name: %s}"
        field_tmpl = self.field_template
        for field_name in self._fields.names:
            param_dict[field_name] = field_tmpl % (self.FIELDS_FILE, field_name)
        param_dict[self.MESH_FILE_VAR] = self.mesh_file
        param_dict[self.TIMESTEP_H1_VAR] = self.time_step_h1
        param_dict[self.TIMESTEP_H2_VAR] = self.time_step_h2
        used_params = substitute_placeholders(yaml_tmpl, yaml_out, param_dict)
        self._fields.set_outer_fields(used_params)

    def set_coarse_sim(self, coarse_sim=None):
        """
        Set coarse simulation ot the fine simulation so that the fine can generate the
        correlated input data sample for both.

        Here in particular set_points to the field generator
        :param coarse_sim
        """
        self.coarse_sim = coarse_sim
        self.n_fine_elements = len(self.points)


      
    def _make_fields(self):



        if self.coarse_sim is  None:
            self._fields.set_points(self.points, self.point_region_ids, self.region_map)
        else:
            coarse_centers = self.coarse_sim.points
            both_centers = np.concatenate((self.points, coarse_centers), axis=0)
            both_regions_ids = np.concatenate( (self.point_region_ids, self.coarse_sim.point_region_ids) )
            assert self.region_map == self.coarse_sim.region_map
            self._fields.set_points(both_centers, both_regions_ids, self.region_map)
        

            
            
    # Needed by Level
    def generate_random_sample(self):
        """
        Generate random field, both fine and coarse part.
        Store them separeted.
        :return:
        """
        # assert self._is_fine_sim
        if not self._fields_inititialied:
            self._make_fields()
        fields_sample = self._fields.sample()
        self._input_sample = {name: values[:self.n_fine_elements, None] for name, values in fields_sample.items()}
        if self.coarse_sim is not None:
            self.coarse_sim._input_sample = {name: values[self.n_fine_elements:, None] for name, values in fields_sample.items()}

    # def get_coarse_sample(self):
    #     """
    #     Return coarse part of generated field.
    #     :return:
    #     """
    #     return self._coarse_sample

    def simulation_sample(self, sample_tag):
        """
        Evaluate model using generated or set input data sample.
        :param sample_tag: A unique ID used as work directory of the single simulation run.
        :param last_sim: No further simulation will be performed at current level
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

        # Add flow123d realization to pbs script
        self.pbs_creater.add_realization(self.n_fine_elements, output_subdir=out_subdir,
                                         work_dir=self.work_dir,
                                         flow123d=self.env['flow123d'])
        return sample_tag, sample_dir

    def extract_result(self, sample_tuple):
        """
        Extract the observed value from the Flow123d output.
        Get sample from the field restriction, write to the GMSH file, call flow.
        :param fields:
        :return:

        TODO: Pass an extraction function as other FlowSim parameter. This function will take the
        balance data and retun observed values.
        """
        sample_dir = sample_tuple[1]
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
            return -total_flux

        else:
            return None
