import os
import os.path
import yaml
import subprocess
import mlmc.gmsh_io as gmsh_io
import numpy as np

#from operator import add
import mlmc.simulation


#from unipath import Path

class Environment:
    def __init__(self, flow123d, gmsh, pbs):
        """

        :param flow123d: Flow123d executable path.
        :param gmsh: Gmsh executable path.
        :param pbs: Dictionary with pbs options:
            n_cpu
            n_nodes
            mem
            queue
        """
        self.flow123d = flow123d
        self.gmsh = gmsh
        self.pbs = pbs


def sub_file_params(file_in, file_out, params):
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



class FlowSim(mlmc.simulation.Simulation):
    """
    Gather data for single flow call (coarse/fine)
    """

    def __init__(self, flow_dict, mesh_step):
        """
        :param env - Environment object.
        :param fields - FieldSet object
        :param yaml_file: Path to the main YAML input.  (absolute or relative to CWD)
        :param geo_file: Path to the geometry file (default is <yaml file base>.geo
        """
        self.__dict__.update(flow_dict)
        # Set attributes of FlowSimGeneric (env, fields, yaml_template, sim_param_range, geo_file).
        self.mesh_step = mesh_step
        self._make_mesh()
        self._substitute_yaml()


    def _make_mesh(self):
        """
        Make the mesh, mesh_file: <geo_base>_step.msh.
        Make substituted yaml: <yaml_base>_step.yaml,
        using common fields_step.msh file for generated fields.
        :return:
        """
        subprocess.call(self.env.gmsh, -2, '-clscale', self.mesh_step, self.geo_file)
        self.mesh_file = os.path.splitext(self.geo_file)[0] + '.msh'
        mesh = gmsh_io.GmshIO(self.mesh_file)
        n_ele = len(mesh.elements)
        self.points = np.zeros(n_ele, 3)
        self.ele_ids = np.zeros(n_ele)
        i = 0
        for id, el in mesh.elements.items():
            type, tags, i_nodes = el
            self.points[i] = np.average(np.array([mesh.nodes[i_node] for i_node in i_nodes]), axis=0)
            self.ele_ids[i] = id
            i += 1


    def _substitute_yaml(self):
        """
        Create substituted YAML file from the tamplate.
        :return:
        """
        self.yaml_file = os.path.splitext(self.yaml_template)[0] + '.inst.yaml'
        self.fields_file = os.path.splitext(self.yaml_template)[0] + '.data.msh'
        param_dict = {}
        field_tmpl = "!FieldElementwise {gmsh_file: ${INPUT}/%s, field_name: %s}"
        for field in self.fields.names():
            param_dict[field] = field_tmpl%(self.fields_file, field)
        param_dict['mesh_file'] = self.mesh_file
        sub_file_params(self.yaml_template, self.yaml_file, param_dict)


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

    def cycle(self, sample_tag):
        """
        Evaluate model using generated or set input data sample.
        :param sample_tag: A unique ID used as work directory of the single simulation run.
        :return: run_token, a dict representing executed simulation. This is passed to extract_result
        to get the result or None if simulation is not finished yet.
        TODO:
        - different mesh and yaml files for individual levels/fine/coarse
        - reuse fine mesh from previous level as coarse mesh
        """
        work_dir_base = os.path.dirname(self.yaml_file)
        work_dir = os.path.join(work_dir_base, sample_tag)
        gmsh_io.GmshIO().write_fields(self.fields_file, self.ele_ids, self._input_sample)
        subprocess.call(self.env.flow123d,
                        '-i', work_dir,
                        '-o', work_dir,
                        '--yaml_balance',
                        self.yaml_file)
        return dict(tag = sample_tag, work_dir = work_dir)

    def extract_result(self, run_token):
        """
        Extract the observed value from the Flow123d output.
        Get sample from the field restriction, write to the GMSH file, call flow.
        :param fields:
        :return:
        JS TODO:
        - check that simulation is done
        - extract  boundary flux
        """
            # extract the flux
            balance_file = os.path.join(run_token['work_dir'], "water_balance.yaml")
            with open(balance_file, "r") as f:
                balance = yaml.load(f)


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
