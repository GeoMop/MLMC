import os
import shutil
import time as t
import numpy as np
from scipy import stats
import pytest
import ruamel.yaml as yaml
import mlmc.quantity.quantity
from test.synth_sim_for_tests import SynthSimulationWorkspaceForTests
from mlmc.sampler import Sampler
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampling_pool import OneProcessPool, ProcessPool
from mlmc.moments import Legendre
from mlmc.estimator import Estimate


np.random.seed(1234)
ref_means = [1., -0.03814235, -0.42411443, 0.05103307, 0.2123083]
ref_vars = [0., 0.02713652, 0.00401481, 0.04033169, 0.01629022]

# Set work dir
os.chdir(os.path.dirname(os.path.realpath(__file__)))
work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
os.makedirs(work_dir)

# Create simulations
distr = stats.norm(loc=1, scale=2)
step_range = [[0.01], [0.001], [0.0001]]
failed_fraction = 0

simulation_config = dict(distr='norm', complexity=2, nan_fraction=failed_fraction, sim_method='_sample_fn')

with open('synth_sim_config_test.yaml', "w") as file:
    yaml.dump(simulation_config, file, default_flow_style=False)
shutil.copyfile('synth_sim_config_test.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))
sim_config_workspace = {"config_yaml": os.path.join(work_dir, 'synth_sim_config.yaml')}


@pytest.mark.parametrize("sampling_pool, simulation_factory",
                         [(OneProcessPool(work_dir=work_dir), SynthSimulationWorkspaceForTests(sim_config_workspace)),
                          (ProcessPool(4, work_dir=work_dir), SynthSimulationWorkspaceForTests(sim_config_workspace)),
                          (OneProcessPool(work_dir=work_dir, debug=True),
                           SynthSimulationWorkspaceForTests(sim_config_workspace)),
                          (ProcessPool(4, work_dir=work_dir, debug=True),
                           SynthSimulationWorkspaceForTests(sim_config_workspace)),
                          ])
def test_sampling_pools(sampling_pool, simulation_factory):
    n_moments = 5
    np.random.seed(123)
    t.sleep(5)

    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    if simulation_factory.need_workspace:
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        shutil.copyfile('synth_sim_config_test.yaml', os.path.join(work_dir, 'synth_sim_config.yaml'))

    sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, "mlmc_{}.hdf5".format(len(step_range))))
    # Plan and compute samples
    sampler = Sampler(sample_storage=sample_storage, sampling_pool=sampling_pool, sim_factory=simulation_factory,
                      level_parameters=step_range)

    true_domain = distr.ppf([0.0001, 0.9999])
    moments_fn = Legendre(n_moments, true_domain)

    sampler.set_initial_n_samples([10, 10, 10])
    sampler.schedule_samples()
    sampler.ask_sampling_pool_for_samples()

    quantity = mlmc.quantity.quantity.make_root_quantity(storage=sample_storage,
                                                         q_specs=sample_storage.load_result_format())
    length = quantity['length']
    time = length[1]
    location = time['10']
    value_quantity = location[0]

    estimator = Estimate(quantity=value_quantity, sample_storage=sample_storage, moments_fn=moments_fn)
    means, vars = estimator.estimate_moments(moments_fn)

    assert means[0] == 1
    assert vars[0] == 0
    assert np.allclose(np.array(ref_means), np.array(means), atol=1e-5)
    assert np.allclose(np.array(ref_vars), np.array(ref_vars), atol=1e-5)

    if sampling_pool._debug:
        assert 'output' in next(os.walk(work_dir))[1]
