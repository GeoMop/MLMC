import os
import sys
import shutil
from scipy import stats

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', 'src/mlmc'))
from new_synth_simulation import SimulationTest
from synth_simulation_with_workspace import SimulationTestUseWorkspace

from sampler import Sampler
from sample_storage_hdf import HDF5
from sample_storage import Memory
from sampling_pool import ProcessPool
from sampling_pool_pbs import SamplingPoolPBS


def sampler_test(hdf=False):

    n_levels = 1
    failed_fraction = 0#0.2

    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    distr = stats.norm()
    step_range = (0.1, 0.006)

    # User configure and create simulation instance
    simulation_config = dict(distr=distr, complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')
    #simulation_config = {"config_yaml": 'synth_sim_config.yaml'}
    simulation_factory = SimulationTest(simulation_config)

    #mlv = MLView(n_levels, simulation_factory, step_range)
    if hdf:
        sample_storage = HDF5(work_dir=work_dir)
    else:
        sample_storage = Memory()
    sampling_pool = ProcessPool(4)

    # Plan and compute samples
    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      n_levels=n_levels, step_range=step_range)

    sampler.determine_level_n_samples()
    sampler.create_simulations()
    sampler.ask_simulations_for_samples()

    # After crash
    # sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, config=mlv)
    # sampler.load_from_storage()
    #
    # This should happen automatically
    # sampler.determine_level_n_samples()
    # sampler.create_simulations()


def sampler_test_with_sim_workspace():

    n_levels = 1
    failed_fraction = 0  # 0.2

    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '')

    distr = stats.norm()
    step_range = (0.1, 0.006)

    print("distr ", distr)

    # User configure and create simulation instance
    simulation_config = {"config_yaml": os.path.join(work_dir, 'synth_sim_config.yaml')}
    simulation_factory = SimulationTestUseWorkspace(simulation_config)

    # mlv = MLView(n_levels, simulation_factory, step_range)
    sample_storage = Memory()
    sampling_pool = ProcessPool(4)

    # Plan and compute samples
    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      n_levels=n_levels, step_range=step_range, work_dir=work_dir)

    sampler.determine_level_n_samples()
    sampler.create_simulations()
    sampler.ask_simulations_for_samples()

    # After crash
    # sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, config=mlv)
    # sampler.load_from_storage()
    #
    # This should happen automatically
    # sampler.determine_level_n_samples()
    # sampler.create_simulations()


def sampler_test_pbs():
    n_levels = 2
    failed_fraction = 0  # 0.2

    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '')

    distr = stats.norm()
    step_range = (0.1, 0.006)

    print("distr ", distr)

    # User configure and create simulation instance
    simulation_config = {"config_yaml": 'synth_sim_config.yaml'}
    simulation_factory = SimulationTestUseWorkspace(simulation_config)

    # mlv = MLView(n_levels, simulation_factory, step_range)
    sample_storage = Memory()
    sampling_pool = SamplingPoolPBS(job_weight=200000, job_count=0)

    pbs_config = dict(
        job_weight=250000,  # max number of elements per job
        n_cores=1,
        n_nodes=1,
        select_flags=['cgroups=cpuacct'],
        mem='4gb',
        queue='charon',
        home_dir='/storage/liberec3-tul/home/martin_spetlik/',
        pbs_process_file_dir='/home/martin/Documents/MLMC_new_design/src/mlmc')

    sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

    # Plan and compute samples
    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      n_levels=n_levels, step_range=step_range, work_dir=work_dir)

    sampler.determine_level_n_samples()
    sampler.create_simulations()
    sampler.ask_simulations_for_samples()


def sampler_test_thread():
    n_levels = 2
    failed_fraction = 0  # 0.2

    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '')

    distr = stats.norm()
    step_range = (0.1, 0.006)

    print("distr ", distr)

    # User configure and create simulation instance
    simulation_config = {"config_yaml": 'synth_sim_config.yaml'}
    simulation_factory = SimulationTestUseWorkspace(simulation_config)

    # mlv = MLView(n_levels, simulation_factory, step_range)
    sample_storage = Memory()
    sampling_pool = SamplingPoolPBS(job_weight=200000, job_count=0)

    pbs_config = dict(
        job_weight=250000,  # max number of elements per job
        n_cores=1,
        n_nodes=1,
        select_flags=['cgroups=cpuacct'],
        mem='4gb',
        queue='charon',
        home_dir='/storage/liberec3-tul/home/martin_spetlik/',
        pbs_process_file_dir='/home/martin/Documents/MLMC_new_design/src/mlmc')

    sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

    # Plan and compute samples
    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      n_levels=n_levels, step_range=step_range, work_dir=work_dir)

    sampler.determine_level_n_samples()
    sampler.create_simulations()
    sampler.ask_simulations_for_samples()




if __name__ == "__main__":
    #sampler_test(hdf=True)
    sampler_test_with_sim_workspace()
    #sampler_test_pbs()
    #sampler_test_thread()
