import os
import sys
import shutil
import ruamel.yaml as yaml

from mlmc.random import correlated_field as cf
from mlmc.tool import flow_mc
from mlmc.moments import Legendre
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from test import base_process
from mlmc.tool.flow_mc import FlowSim


class CondField(base_process.Process):

    def __init__(self):
        args = self.get_arguments(sys.argv[1:])

        self.step_range = [1, 0.01]  # Each level must have a defined step, len(step_range) == number of MLMC levels

        self.work_dir = args.work_dir
        self.append = False
        self.clean = args.clean

        if args.command == 'run':
            self.run()
        else:
            self.append = True
            self.clean = False
            self.run(renew=True) if args.command == 'renew' else self.run()

    def run(self, renew=False):
        """
        Run mlmc
        :return: None
        """
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

        sampler_list = []
        for nl in [1]:  # , 2, 3, 4,5, 7, 9]:
            sampler = self.setup_config(nl, clean=True)
            # self.n_sample_estimate(mlmc)
            self.generate_jobs(sampler, n_samples=[8])
            sampler_list.append(sampler)

        self.all_collect(sampler_list)

    def setup_config(self, n_levels, clean):
        """
        Simulation dependent configuration
        :param n_levels: Number of levels
        :param clean: bool, If True remove existing files
        :return: mlmc.sampler instance
        """

        # Random fields setting
        #
        # In the previous version it was defined as:
        fields = cf.Fields([
            cf.Field('conductivity', cf.FourierSpatialCorrelatedField('gauss', dim=2, corr_length=0.125, log=True)),
        ])
        # I tried to avoid passing objects, so the new version passes dictionary.

        # fields = {"name": "conductivity", "class": "FourierSpatialCorrelatedField", "params": {"name": "gauss", "dim": 2,
        #                                                                                        "corr_length": 0.125,
        #                                                                                        "log": True}
        #           }

        #####
        # Create HDF sample storage
        #####
        sample_storage = SampleStorageHDF(file_path=os.path.join(self.work_dir, "mlmc_{}.hdf5".format(len(self.step_range))),
                                          append=self.append)

        #####
        # Create simulation factory
        #####
        # Set pbs config, flow123d, gmsh, ...
        self.set_environment_variables()

        shutil.copyfile('synth_sim_config.yaml', os.path.join(self.work_dir, 'synth_sim_config.yaml'))

        output_dir = os.path.join(self.work_dir, "output_{}".format(n_levels))
        # remove existing files
        if clean:
            self.rm_files(output_dir)

        #####
        # Init sampling pool object
        #####
        sampling_pool = self.create_pbs_sampling_pool(output_dir, clean)

        simulation_config = {
            'env': dict(flow123d=self.flow123d, gmsh=self.gmsh),  # The Environment.
            'output_dir': output_dir,
            'fields': fields,  # Dictionary of correlated fields
            'yaml_file': os.path.join(self.work_dir, '01_conductivity.yaml'),
        # The template with a mesh and field placeholders
            'sim_param_range': self.step_range,  # Range of MLMC simulation parametr. Here the mesh step.
            'geo_file': os.path.join(self.work_dir, 'square_1x1.geo'),
        # The file with simulation geometry (independent of the step)
            # 'field_template': "!FieldElementwise {mesh_data_file: \"${INPUT}/%s\", field_name: %s}"
            'field_template': "!FieldElementwise {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        }

        #FlowProcSim.total_sim_id = 0

        #self.options['output_dir'] = output_dir

        simulation_factory = FlowSim(self.step_range, config=simulation_config, clean=clean)

        # Plan and compute samples
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          step_range=self.step_range)

        # true_domain = distr.ppf([0.0001, 0.9999])
        # moments_fn = Legendre(n_moments, true_domain)

        # sampler.set_initial_n_samples([10, 4])
        # # sampler.set_initial_n_samples([1000])
        # sampler.schedule_samples()
        # sampler.ask_sampling_pool_for_samples()

        return sampler

    def set_environment_variables(self):
        """
        Set pbs config, flow123d, gmsh
        :return: None
        """
        root_dir = os.path.abspath(self.work_dir)
        while root_dir != '/':
            root_dir, tail = os.path.split(root_dir)

        if tail == 'storage':
            # Metacentrum
            self.sample_sleep = 30
            self.init_sample_timeout = 600
            self.sample_timeout = 0
            self.pbs_config['qsub'] = '/usr/bin/qsub'
            self.flow123d = 'flow123d'  # "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"
            self.gmsh = "/storage/liberec3-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
        else:
            # Local
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60
            self.pbs_config['qsub'] = None
            self.flow123d = "/home/jb/workspace/flow123d/bin/fterm flow123d dbg"
            self.gmsh = "/home/jb/local/gmsh-3.0.5-git-Linux/bin/gmsh"

    def rm_files(self, output_dir):
        """
        Rm files and dirs
        :param output_dir: Output directory path
        :return:
        """
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, mode=0o775, exist_ok=True)

    def create_pbs_sampling_pool(self, output_dir, clean):
        """
        Initialize object for PBS execution
        :param output_dir: Output directory
        :param clean: bool, if True remove existing files
        :return: None
        """
        # pbs_work_dir = os.path.join(output_dir, "scripts")
        # num_jobs = 0
        # if os.path.isdir(pbs_work_dir):
        #     num_jobs = len([_ for _ in os.listdir(pbs_work_dir)])

        #####
        # Create PBS sampling pool
        #####
        sampling_pool = SamplingPoolPBS(job_weight=20000000, work_dir=self.work_dir, clean=self.clean)

        pbs_config = dict(
            n_cores=1,
            n_nodes=1,
            select_flags=['cgroups=cpuacct'],
            mem='128mb',
            queue='charon_2h',
            home_dir='/storage/liberec3-tul/home/martin_spetlik/',
            pbs_process_file_dir='/auto/liberec3-tul/home/martin_spetlik/MLMC_new_design/src/mlmc',
            # @TODO: remove??
            python='python3',
            env_setting=['cd {work_dir}',
                         'module load python36-modules-gcc',
                         'source env/bin/activate',
                         'pip3 install /storage/liberec3-tul/home/martin_spetlik/MLMC_new_design',
                         'module use /storage/praha1/home/jan-hybs/modules',
                         'module load python36-modules-gcc',
                         'module list']
        )

        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        return sampling_pool

    def generate_jobs(self, sampler, n_samples=None):
        """
        Generate level samples
        :param n_samples: None or list, number of samples for each level
        :return: None
        """
        if n_samples is not None:
            sampler.set_initial_n_samples(n_samples)
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)

    # def process(self):
    #     """
    #     Use collected data
    #     :return: None
    #     """
    #     assert os.path.isdir(self.work_dir)
    #     mlmc_est_list = []
    #     # for nl in [ 1,3,5,7,9]:
    #
    #     import time
    #     for nl in [5]:  # high resolution fields
    #         start = time.time()
    #         sampler = self.setup_config(nl, clean=False)
    #         print("celkový čas ", time.time() - start)
    #         # Use wrapper object for working with collected data
    #         mlmc_est = Estimate(mlmc)
    #         mlmc_est_list.append(mlmc_est)


if __name__ == "__main__":
    pr = CondField()
