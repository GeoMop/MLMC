import os
import sys
import shutil
import ruamel.yaml as yaml
import numpy as np

from mlmc.random import correlated_field as cf
from mlmc.tool import flow_mc
from mlmc.moments import Legendre
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from mlmc.tool import base_process
from mlmc.tool.flow_mc import FlowSim
from mlmc.moments import Legendre
from mlmc.quantity_estimate import QuantityEstimate


class CondField(base_process.Process):

    def __init__(self):
        args = self.get_arguments(sys.argv[1:])

        self.step_range = [[1, 0.01]]  # Each level must have a defined step, len(step_range) == number of MLMC levels
        # step_range allows defined more MLMC realizations each one has step_range i.e. [1, 0.1]

        self.work_dir = args.work_dir
        self.append = False
        # add samples to existing ones
        self.clean = args.clean
        # remove HDF5 file, start from scratch

        if args.command == 'run':
            self.run()
        else:
            self.append = True  # Use 'collect' command (see base_process.Process) to add samples
            self.clean = False
            self.run(renew=True) if args.command == 'renew' else self.run()

    def run(self, renew=False):
        """
        Run MLMC
        :param renew: If True then rerun failed samples with same sample id
        :return: None
        """
        # Create working directory if necessary
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

        # Create sampler and schedule samples for each step range
        sampler_list = []
        for step_range in self.step_range:  # Length of step range determines number of levels
            if self.clean:
                # Remove HFD5 file
                if os.path.exists(os.path.join(self.work_dir, "mlmc_{}.hdf5".format(len(step_range)))):
                    os.remove(os.path.join(self.work_dir, "mlmc_{}.hdf5".format(len(step_range))))

            # Create sampler (mlmc.Sampler instance) - crucial class which actually schedule samples
            sampler = self.setup_config(step_range, clean=True)
            # Schedule samples
            self.generate_jobs(sampler, n_samples=[100, 50], renew=renew)

            sampler_list.append(sampler)

        self.all_collect(sampler_list)  # Check if all samples are finished
        self.calculate_moments(sampler_list)  # Simple moment check

    def setup_config(self, step_range, clean):
        """
        Simulation dependent configuration
        :param step_range: Simulation's step range, length of them is number of levels
        :param clean: bool, If True remove existing files
        :return: mlmc.sampler instance
        """
        # Set pbs config, flow123d, gmsh, ..., random fields are set in simulation class
        self.set_environment_variables()

        # Create Pbs sampling pool
        sampling_pool = self.create_pbs_sampling_pool()

        simulation_config = {
            'work_dir': self.work_dir,
            'env': dict(flow123d=self.flow123d, gmsh=self.gmsh),  # The Environment.
            'yaml_file': os.path.join(self.work_dir, '01_conductivity.yaml'),
            # The template with a mesh and field placeholders
            'sim_param_range': step_range,  # Range of MLMC simulation parametr. Here the mesh step.
            'geo_file': os.path.join(self.work_dir, 'square_1x1.geo'),
            # The file with simulation geometry (independent of the step)
            # 'field_template': "!FieldElementwise {mesh_data_file: \"${INPUT}/%s\", field_name: %s}"
            'field_template': "!FieldElementwise {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        }

        # Create simulation factory
        simulation_factory = FlowSim(step_range, config=simulation_config, clean=clean)

        # Create HDF sample storage
        sample_storage = SampleStorageHDF(
            file_path=os.path.join(self.work_dir, "mlmc_{}.hdf5".format(len(step_range))),
            append=self.append)

        # Create sampler, it manages sample scheduling and so on
        sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                          step_range=step_range)

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
            self.flow123d = 'flow123d'  # "/storage/praha1/home/jan_brezina/local/flow123d_2.2.0/flow123d"
            self.gmsh = "/storage/liberec3-tul/home/martin_spetlik/astra/gmsh/bin/gmsh"
        else:
            # Local
            self.sample_sleep = 1
            self.init_sample_timeout = 60
            self.sample_timeout = 60
            self.flow123d = "/home/jb/workspace/flow123d/bin/fterm flow123d dbg"
            self.gmsh = "/home/jb/local/gmsh-3.0.5-git-Linux/bin/gmsh"

    # def rm_files(self, output_dir):
    #     """
    #     Rm files and dirs
    #     :param output_dir: Output directory path
    #     :return:
    #     """
    #     if os.path.isdir(output_dir):
    #         shutil.rmtree(output_dir, ignore_errors=True)
    #     os.makedirs(output_dir, mode=0o775, exist_ok=True)

    def create_pbs_sampling_pool(self):
        """
        Initialize object for PBS execution
        :return: None
        """
        # Create PBS sampling pool
        sampling_pool = SamplingPoolPBS(job_weight=20000000, work_dir=self.work_dir, clean=self.clean)

        pbs_config = dict(
            n_cores=1,
            n_nodes=1,
            select_flags=['cgroups=cpuacct'],
            mem='128mb',
            queue='charon_2h',
            home_dir='/storage/liberec3-tul/home/martin_spetlik/',
            # pbs_process_file_dir='/auto/liberec3-tul/home/martin_spetlik/MLMC_new_design/src/mlmc',
            python='python3',
            env_setting=['cd {work_dir}',
                         'module load python36-modules-gcc',
                         'source env/bin/activate',
                         'pip3 install /storage/liberec3-tul/home/martin_spetlik/MLMC_new_design',
                         'module use /storage/praha1/home/jan-hybs/modules',
                         'module load python36-modules-gcc',
                         'module load flow123d',
                         'module list']
        )

        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        return sampling_pool

    def generate_jobs(self, sampler, n_samples=None, renew=False):
        """
        Generate level samples
        :param n_samples: None or list, number of samples for each level
        :param renew: rerun failed samples with same random seed (= same sample id)
        :return: None
        """
        if renew:
            sampler.ask_sampling_pool_for_samples()
            sampler.renew_failed_samples()
            sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)
        else:
            if n_samples is not None:
                sampler.set_initial_n_samples(n_samples)
            sampler.schedule_samples()
            sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)

    def calculate_moments(self, sampler_list):
        """
        Calculate moments through the mlmc.QuantityEstimate
        :param sampler_list: List of samplers (mlmc.Sampler)
        :return: None
        """
        # Simple moment evaluation
        for sampler in sampler_list:
            moments_fn = self.set_moments(sampler.sample_storage)

            q_estimator = QuantityEstimate(sample_storage=sampler.sample_storage, moments_fn=moments_fn,
                                           sim_steps=self.step_range)

            print("collected samples ", sampler._n_scheduled_samples)
            means, vars = q_estimator.estimate_moments(moments_fn)
            print("means ", means)
            print("vars ", vars)

            # The first moment is in any case 1 and its variance is 0
            assert means[0] == 1
            # assert np.isclose(means[1], 0, atol=1e-2)
            assert vars[0] == 0

    def set_moments(self, sample_storage):
        n_moments = 5
        true_domain = QuantityEstimate.estimate_domain(sample_storage, quantile=0.01)
        return Legendre(n_moments, true_domain)

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
