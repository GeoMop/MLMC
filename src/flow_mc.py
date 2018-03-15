import os
import os.path
import yaml
import subprocess
import mlmc.gmsh_io as gmsh_io
import numpy as np
import shutil

#from operator import add
import mlmc.simulation


#from unipath import Path

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




class FlowSim(mlmc.simulation.Simulation):
    total_sim_id = 0
    MESH_FILE_PLACEHOLDER = 'mesh_file'
    GEO_FILE='mesh.geo'
    MESH_FILE='mesh.msh'
    YAML_TEMPLATE='flow_input.yaml.tmpl'
    YAML_FILE = 'flow_input.yaml'
    FIELDS_FILE = 'fields_sample.msh'

    """
    Gather data for single flow call (coarse/fine)
    """

    def __init__(self, flow_dict, mesh_step):
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
        """
        self.sim_id = FlowSim.total_sim_id
        FlowSim.total_sim_id += 1
        self.env  = flow_dict['env']
        self.fields = flow_dict['fields']
        self.base_yaml_file = flow_dict['yaml_file']
        self.base_geo_file = flow_dict['geo_file']
        self.mesh_step = mesh_step

        # Set in _make_mesh
        self.points = None
        # Element centers of computational mesh.
        self.ele_ids = None
        # Element IDs of computational mesh.

        # Prepare base workdir for this mesh_step
        base_dir = os.path.dirname(self.base_yaml_file)
        self.work_dir = os.path.join(base_dir, 'sim_%d_step_%f'%(self.sim_id, self.mesh_step))
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
        self.points = np.zeros((n_ele, 3))
        self.ele_ids = np.zeros(n_ele)
        i = 0
        for id, el in mesh.elements.items():
            type, tags, i_nodes = el
            self.points[i] = np.average(np.array([mesh.nodes[i_node] for i_node in i_nodes]), axis=0)
            self.ele_ids[i] = id
            i += 1


    def _substitute_yaml(self, yaml_tmpl, yaml_out):
        """
        Create substituted YAML file from the tamplate.
        :return:
        """
        param_dict = {}
        field_tmpl = "!FieldElementwise {gmsh_file: ${INPUT}/%s, field_name: %s}"
        for field in self.fields.names():
            param_dict[field] = field_tmpl%(self.FIELDS_FILE, field)
        param_dict[self.MESH_FILE_PLACEHOLDER] = self.mesh_file
        substitute_placeholders(yaml_tmpl, yaml_out, param_dict)


    def set_coarse_sim(self, coarse_sim):
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
    def random_array(self):
        """
        Generate random field, both fine and coarse part.
        Store them separeted.
        :return:
        """
        assert self._is_fine_sim
        fields_sample = self.fields.sample()
        self._input_sample = {name: values[:self.n_fine_elements] for name, values in fields_sample }
        self._coarse_sample = {name: values[self.n_fine_elements:] for name, values in fields_sample }


    def get_random_array(self):
        """
        Return coarse part of generated field.
        :return:
        """
        return self._coarse_sample


    @staticmethod
    def _make_pbs_script(**kwargs):

        lines = [
            "#!/bin/bash",
            '# PBS -S /bin/bash',
            '# PBS -l select={n_nodes}:ncpus=${n_cores}:mem={mem}gb',
            '# PBS -q {queue}',
            '# PBS -N Flow123d',
            '# PBS -j oe',
            '# PBS -e {work_dir}/{output_subdir}'
            '# PBS -o {work_dir}/{output_subdir}',
            'cd {work_dir}',
            'time -p {flow123d} --yaml_balance -i {output_subdir} -o {output_subdir} {main_input} >{work_dir}/{output_subdir}/flow.out',
            'touch FINISHED']

        lines = [ line.format(kwargs) for line in lines ]
        return "\n".join(lines)

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
        sample_dir = os.path.join(self.work_dir, sample_tag)
        force_mkdir(sample_dir)
        fields_file = os.path.join(self.sample_dir, self.FIELDS_FILE)
        gmsh_io.GmshIO().write_fields(fields_file, self.ele_ids, self._input_sample)

        pbs = self.env.pbs or {}    # Empty dict for None.
        pbs_file = os.path.join(sample_dir, "pbs_script.sh")
        with open(pbs_file, "w") as f:
            pbs_script = self.make_pbs_script(
                n_nodes = pbs.get('n_nodes', 1),
                n_cores = pbs.get('n_cores', 1),
                mem = pbs.get('mem', 2),
                queue = pbs.get('queue', "charon"),
                output_dir = sample_dir,
                work_dir = self.work_dir,
                flow123d = self.env['flow123d']
            )
            f.write(pbs_script)
        os.chmod(pbs_file, 774) # Make exectutable to allow direct call.

        if self.env.pbs is None:
            # Direct execution.
            subprocess.call(pbs_file)
        else:
            # Batch execution.
            subprocess.call("qsub", shell=True)
        return (sample_tag, sample_dir)

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

            flux_regions = ['bc_outlet']
            total_flux = 0.0
            found = False
            for flux_item in balance['data']:
                if flux_item['time'] > 0:
                    break
                if flux_item['region'] in flux_regions:
                    flux = flux_item['data'][0]
                    flux_in = flux_item['data'][1]
                    if flux_in > 1e-10:
                        raise Exception("Possitive inflow at outlet region.")
                    total_flux += flux  # flux field
                    found = True
            raise Exception("Observation region not found.")
            return total_flux
        else:
            return None



class FlowSimGeneric:
    """
    Class to run a single Flow123d simulation with generated input random fields.
    Setup:
    - set environment, main YAML, geometry (__init__), field set (must be a single object since we have to deal
    with cross correlations)

    Initialization:
    1. construction:
        environment (flow123d path, gmsh path, pbs setting)
        main yaml_file (with mesh and fields placeholders),
        given geometry file,
        given mesh step
    2. create the fine mesh, calling GMSH
    3. set the coarse mesh (from upper level)
    4. create the yaml file (substitute mesh files, fields)
    5. extract element centers

    Run simulation:
    4. Create realization directory, cd to relaization directory
    5. Generate random fields, write them to the GMSH file
    6. Copy yaml files and meshes there.
    7. run flow (use yaml balance) using a (PBS possibly) script that touch a given file lock at the end.

    Postprocess:
    1. test if we have results, check for the lock file.
    2. read from balance.yaml, the result
    """
    def __init__(self, env, fields, yaml_file, step_range, geo_file=None):
        """
        :param env - Environment object.
        :param fields - FieldSet object
        :param yaml_file: Path to the main YAML input.  (absolute or relative to CWD)
        :param geo_file: Path to the geometry file (default is <yaml file base>.geo
        """
        yaml_file = os.path.abspath(yaml_file)
        if geo_file is None:
            geo_file = os.path.splitext(yaml_file)[0] + '.geo'

        self.flow_setup = {
            'env': env,                 # The Environment.
            'fields': fields,           # correlated_field.FieldSet object
            'yaml_file': yaml_file,     # The template with a mesh and field placeholders
            'sim_param_range': step_range,  # Range of MLMC simulation parametr. Here the mesh step.
            'geo_file': geo_file        # The file with simulation geometry (independent of the step)
        }


    def interpolate_precision(self, t=None):
            """
            't' is a parameter from interval [0,1], where 0 corresponds to the lower bound
            and 1 to the upper bound of the simulation parameter range.
            :param t: float 0 to 1
            :return:
            TODO: move to the parent class, pass in the range as a parameter
            """
            if t is None:
                sim_param = 0
            else:
                assert 0 <= t <= 1
                # logaritmic interpolation
                sim_param = self.flow_setup['sim_param_range'][0] ** (1 - t) * self.flow_setup['sim_param_range'][1] ** t

            return FlowSim(self.flow_setup, sim_param)

        #
        # #fileDir = os.path.dirname(os.path.realpath('__file__'))
        # mesh_file = self._extract_mesh_file(yaml_file)
        # #filename = os.path.join(fileDir, mesh_file_dir)
        # gio = GmshIO()
        # with open(mesh_file) as f:
        #    gio.read(f)
        # coord = np.zeros((len(gio.elements),3))
        # for i, one_el in enumerate(gio.elements.values()):
        #     i_nodes = one_el[2]
        #     coord[i] = np.average(np.array([ gio.nodes[i_node] for i_node in i_nodes]), axis=0)
        # self.points = coord
        # self.yaml_dir = os.path.join(fileDir, yaml_file)
        # self.mesh_dir = os.path.join(fileDir, mesh_file_dir)
        # self.gio      = gio


    #
    # def add_field(self, placeholder):
    #     """
    #     Specify a generated field.
    #     :param placeholder: The placeholder for the generated FieldElementwise field in the YAMLfile, e.g.
    #     ...
    #       input_fields:
    #         - region: BULK
    #           conductivity: !FieldElementwise
    #             <PLACEHOLDER>
    #
    #     :return:
    #     """
    #
    #
    # def initialize(self, mesh_step):
    #     """
    #     Set the mesh step, generate the meshes and substitute to the yaml files.
    #     :param mesh_step: The mesh step of the fine mesh.
    #     :return:
    #     """
    #     self.fine = FlowCall(self.flow123d, self.yaml_template, self.geo_file, mesh_step)
    #     self.coarse = FlowCall(self.flow123d, self.yaml_template, self.geo_file, mesh_step * self.coarsing_factor)
    #     for flow in [self.fine, self.coarse]:
    #         flow.make_mesh()
    #
    #
    #
    # def _extract_mesh_file(self, yaml_file):
    #     with open(yaml_file, 'r') as f:
    #         input = yaml.load(f)
    #         return input.mesh.mesh_file
    #
    # def add_field(self, name, mu, sig2, corr):
    #
    #     field = SpatialCorrelatedField(corr_exp = 'gauss', dim = self.points.shape[1], corr_length = corr,aniso_correlation = None,  )
    #     field.set_points(self.points, mu = mu, sigma = sig2)
    #     hodnoty = field.sample()
    #     p = Path(self.mesh_dir)
    #     filepath = p.parent +'\\'+ name + '_values.msh'
    #     self.gio.write_fields(filepath, hodnoty, name)
    #     print ("Field created in",filepath)
    #
    #
    # def extract_value(self):
    #     p = Path(self.mesh_dir)
    #     filename = os.path.join(p.parent,'output\\mass_balance.txt')
    #     soubor   = open(filename,'r')
    #     for line in soubor:
    #         line = line.rstrip()
    #         if re.search('1', line):
    #             x = line
    #
    #     y         = x.split('"conc"',100)
    #     z         = y[1].split('\t')
    #     var_name  = -float(z[3])
    #
    #     return var_name
    #
    # def Flow_run(self,yaml_file):
    #     p = Path(self.yaml_dir)
    #     os.chdir(p.parent)
    #     #os.chdir("C:\\Users\\Klara\\Documents\\Intec\\PythonScripts\\MonteCarlo\\Flow_02_test")
    #     os.system('call fterm.bat //opt/flow123d/bin/flow123d -s ' + str(yaml_file))
    #     #os.system("call fterm.bat //opt/flow123d/bin/flow123d -s 02_mysquare.yaml")
    #     #os.system('call fterm.bat //opt/flow123d/bin/flow123d -s 02_mysquare.yaml')
    #     #os.system("call fterm.bat //opt/flow123d/bin/flow123d -s" + str(self.yaml_dir) + '"')
    #
