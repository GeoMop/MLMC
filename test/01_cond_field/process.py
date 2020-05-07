import os
import sys
import ruamel.yaml as yaml

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', '..', 'src'))
src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', 'src/mlmc'))

import flow_mc as flow_mc
from moments import Legendre
from sampler import Sampler
from sample_storage import Memory
from sample_storage_hdf import SampleStorageHDF
from sampling_pool_pbs import SamplingPoolPBS
from quantity_estimate import QuantityEstimate
import new_estimator



sys.path.append(os.path.join(src_path, '..'))
import base_process


class FlowProcSim(flow_mc.FlowSim):
    """
    Child from FlowSimulation that defines extract method
    """

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
            run_time = self.get_run_time(sample_dir)

            if not found:
                raise Exception
            return -total_flux, run_time
        else:
            return None, 0


class CondField(base_process.Process):

    def __init__(self):
        args = self.get_arguments(sys.argv[1:])

        self.step_range = (1, 0.01)

        self.work_dir = args.work_dir
        self.options = {'keep_collected': args.keep_collected,
                        'regen_failed': args.regen_failed}

        if args.command == 'run':
            self.run()
        elif args.command == 'collect':
            self.collect()
        elif args.command == 'process':
            self.process()

    def get_arguments(self, arguments):
        """
        Getting arguments from console
        :param arguments: list of arguments
        :return: namespace
        """
        import argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('command', choices=['run', 'collect', 'process'], help='Run, collect or process')
        parser.add_argument('work_dir', help='Work directory')
        parser.add_argument("-r", "--regen-failed", default=False, action='store_true',
                            help="Regenerate failed samples", )
        parser.add_argument("-k", "--keep-collected", default=False, action='store_true',
                            help="Keep sample dirs")

        args = parser.parse_args(arguments)
        return args

    def run(self):
        """
        Run mlmc
        :return: None
        """
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

        mlmc_list = []
        for nl in [1]:  # , 2, 3, 4,5, 7, 9]:
            mlmc = self.setup_config(nl, clean=True)
            # self.n_sample_estimate(mlmc)
            self.generate_jobs(mlmc, n_samples=[8])
            mlmc_list.append(mlmc)

        self.all_collect(mlmc_list)

    def setup_config(self, n_levels, clean):
        """
        Simulation dependent configuration
        :param n_levels: Number of levels
        :param clean: bool, If True remove existing files
        :return: mlmc.MLMC instance
        """
        fields = cf.Fields([
            cf.Field('conductivity', cf.FourierSpatialCorrelatedField('gauss', dim=2, corr_length=0.125, log=True)),
        ])

        pbs_config = dict(
            n_cores=1,
            n_nodes=1,
            select_flags=['cgroups=cpuacct'],
            mem='128mb',
            queue='charon_2h',
            home_dir='/storage/liberec3-tul/home/martin_spetlik/',
            pbs_process_file_dir='/auto/liberec3-tul/home/martin_spetlik/MLMC_new_design/src/mlmc')

        sampling_pool = SamplingPoolPBS(job_weight=200000, job_count=0, work_dir=self.work_dir)

        # Set pbs config, flow123d, gmsh, ...
        self.set_environment_variables()
        output_dir = os.path.join(self.work_dir, "output_{}".format(n_levels))
        # remove existing files
        if clean:
            self.rm_files(output_dir)

        # Init pbs object
        self.create_pbs_object(output_dir, clean)

        simulation_config = {
            'env': dict(flow123d=self.flow123d, gmsh=self.gmsh, pbs=self.pbs_obj),  # The Environment.
            'output_dir': output_dir,
            'fields': fields,  # correlated_field.FieldSet object
            'yaml_file': os.path.join(self.work_dir, '01_conductivity.yaml'),  # The template with a mesh and field placeholders
            'sim_param_range': self.step_range,  # Range of MLMC simulation parametr. Here the mesh step.
            'geo_file': os.path.join(self.work_dir, 'square_1x1.geo'),  # The file with simulation geometry (independent of the step)
            # 'field_template': "!FieldElementwise {mesh_data_file: \"${INPUT}/%s\", field_name: %s}"
            'field_template': "!FieldElementwise {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        }

        FlowProcSim.total_sim_id = 0

        self.options['output_dir'] = output_dir
        mlmc_obj = mlmc.mlmc.MLMC(n_levels, FlowProcSim.factory(self.step_range, config=simulation_config, clean=clean),
                                  self.step_range, self.options)

        sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_{}.hdf5".format(len(step_range))))


        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        sampler = Sampler()

        sampler.set_initial_n_samples([10, 4])
        # sampler.set_initial_n_samples([1000])
        sampler.schedule_samples()
        sampler.ask_sampling_pool_for_samples()

        # if clean:
        #     # Create new execution of mlmc
        #     # Create new execution of mlmc
        #     mlmc_obj.create_new_execution()
        # else:
        #     # Use existing mlmc HDF file
        #     mlmc_obj.load_from_file()
        return mlmc_obj

    def set_environment_variables(self):
        """
        Set pbs config, flow123d, gmsh
        :return: None
        """
        root_dir = os.path.abspath(self.work_dir)
        while root_dir != '/':
            root_dir, tail = os.path.split(root_dir)

        self.pbs_config = dict(
            job_weight=250000,  # max number of elements per job
            n_cores=1,
            n_nodes=1,
            select_flags=['cgroups=cpuacct'],
            mem='4gb',
            queue='charon',
            home_dir='/storage/liberec3-tul/home/martin_spetlik/')

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

    def create_pbs_object(self, output_dir, clean):
        """
        Initialize object for PBS execution
        :param output_dir: Output directory
        :param clean: bool, if True remove existing files
        :return: None
        """
        pbs_work_dir = os.path.join(output_dir, "scripts")
        num_jobs = 0
        if os.path.isdir(pbs_work_dir):
            num_jobs = len([_ for _ in os.listdir(pbs_work_dir)])

        self.pbs_obj = pbs.Pbs(pbs_work_dir,
                               job_count=num_jobs,
                               qsub=self.pbs_config['qsub'],
                               clean=clean)
        self.pbs_obj.pbs_common_setting(flow_3=True, **self.pbs_config)

    def generate_jobs(self, mlmc, n_samples=None):
        """
        Generate level samples
        :param n_samples: None or list, number of samples for each level
        :return: None
        """
        if n_samples is not None:
            mlmc.set_initial_n_samples(n_samples)
        mlmc.refill_samples()

        if self.pbs_obj is not None:
            self.pbs_obj.execute()
        mlmc.wait_for_simulations(sleep=self.sample_sleep, timeout=self.sample_timeout)

    def process(self):
        """
        Use collected data
        :return: None
        """
        assert os.path.isdir(self.work_dir)
        mlmc_est_list = []
        # for nl in [ 1,3,5,7,9]:

        import time
        for nl in [5]:  # high resolution fields
            start = time.time()
            mlmc = self.setup_config(nl, clean=False)
            print("celkový čas ", time.time() - start)
            # Use wrapper object for working with collected data
            mlmc_est = Estimate(mlmc)
            mlmc_est_list.append(mlmc_est)



if __name__ == "__main__":
    pr = CondField()
