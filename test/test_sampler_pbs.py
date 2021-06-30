import os
import sys
import shutil
import ruamel.yaml as yaml
import numpy as np
from scipy import stats
import argparse
import pytest

from mlmc.moments import Legendre
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from mlmc.estimator import Estimate
from mlmc.sim.synth_simulation import SynthSimulationWorkspace
import mlmc.quantity.quantity


@pytest.mark.pbs
def test_sampler_pbs(work_dir, clean=False, debug=False):
    np.random.seed(3)
    n_moments = 5
    distr = stats.norm(loc=1, scale=2)
    step_range = [0.5, 0.01]
    n_levels = 5

    # if clean:
    #     if os.path.isdir(work_dir):
    #         shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, mode=0o775, exist_ok=True)

    assert step_range[0] > step_range[1]
    level_parameters = []
    for i_level in range(n_levels):
        if n_levels == 1:
            level_param = 1
        else:
            level_param = i_level / (n_levels - 1)
        level_parameters.append([step_range[0] ** (1 - level_param) * step_range[1] ** level_param])

    failed_fraction = 0
    simulation_config = dict(distr='norm', complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')

    with open(os.path.join(work_dir, 'synth_sim_config.yaml'), "w") as file:
        yaml.dump(simulation_config, file, default_flow_style=False)

    simulation_config = {"config_yaml": os.path.join(work_dir, 'synth_sim_config.yaml')}
    simulation_factory = SynthSimulationWorkspace(simulation_config)

    if clean and os.path.exists(os.path.join(work_dir, "mlmc_{}.hdf5".format(len(step_range)))):
        os.remove(os.path.join(work_dir, "mlmc_{}.hdf5".format(len(step_range))))

    if clean and os.path.exists(os.path.join(work_dir, "output")):
        shutil.rmtree(os.path.join(work_dir, "output"), ignore_errors=True)

    sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_{}.hdf5".format(len(step_range))))
    sampling_pool = SamplingPoolPBS(work_dir=work_dir, clean=clean)
    #sampling_pool = OneProcessPool()

    shutil.copyfile(os.path.join(work_dir, 'synth_sim_config.yaml'),
                    os.path.join(sampling_pool._output_dir, 'synth_sim_config.yaml'))

    pbs_config = dict(
        n_cores=1,
        n_nodes=1,
        select_flags=['cgroups=cpuacct'],
        mem='2Gb',
        queue='charon',
        pbs_name='flow123d',
        walltime='72:00:00',
        optional_pbs_requests=[],  # e.g. ['#PBS -m ae', ...]
        home_dir='/auto/liberec3-tul/home/martin_spetlik/',
        python='python3',
        env_setting=['cd $MLMC_WORKDIR',
                     'module load python36-modules-gcc',
                     'source env/bin/activate',
                     # 'pip3 install /storage/liberec3-tul/home/martin_spetlik/MLMC_new_design',
                     'module use /storage/praha1/home/jan-hybs/modules',
                     'module load python36-modules-gcc',
                     'module load flow123d',
                     'module list']
    )

    sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

    # Plan and compute samples
    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool,
                      sim_factory=simulation_factory,
                      level_parameters=level_parameters)

    true_domain = distr.ppf([0.0001, 0.9999])
    moments_fn = Legendre(n_moments, true_domain)

    sampler.set_initial_n_samples([1e7, 5e6, 1e6, 5e5, 1e4])
    #sampler.set_initial_n_samples([1e1, 1e1, 1e1, 1e1, 1e1])
    #sampler.set_initial_n_samples([4, 4, 4, 4, 4])
    sampler.schedule_samples()
    n_running = sampler.ask_sampling_pool_for_samples()

    quantity = mlmc.quantity.quantity.make_root_quantity(storage=sample_storage,
                                                         q_specs=sample_storage.load_result_format())
    length = quantity['length']
    time = length[1]
    location = time['10']
    value_quantity = location[0]

    estimator = Estimate(quantity=value_quantity, sample_storage=sample_storage, moments_fn=moments_fn)


    # target_var = 1e-3
    # sleep = 0
    # add_coef = 0.1
    #
    # # @TODO: test
    # # New estimation according to already finished samples
    # variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
    # n_estimated = new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
    #                                                                    n_levels=sampler.n_levels)
    # # Loop until number of estimated samples is greater than the number of scheduled samples
    # while not sampler.process_adding_samples(n_estimated, sleep, add_coef):
    #     # New estimation according to already finished samples
    #     variances, n_ops = q_estimator.estimate_diff_vars_regression(sampler._n_scheduled_samples)
    #     n_estimated = new_estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
    #                                                                        n_levels=sampler.n_levels)

    #print("collected samples ", sampler._n_created_samples)
    means, vars = estimator.estimate_moments(moments_fn)

    # print("means ", means)
    # print("vars ", vars)
    # assert means[0] == 1
    # assert np.isclose(means[1], 0, atol=1e-2)
    # assert vars[0] == 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', help='Work directory')

    parser.add_argument("-c", "--clean", default=False, action='store_true',
                        help="Clean before run, used only with 'run' command")
    parser.add_argument("-d", "--debug", default=False, action='store_true',
                        help="Keep sample directories")

    args = parser.parse_args(arguments)

    test_sampler_pbs(os.path.abspath(args.work_dir), clean=args.clean, debug=args.debug)
