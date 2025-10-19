import os
import ruamel.yaml as ruyaml
import numpy as np
from typing import List
import scipy.stats as stats
from mlmc.sim.simulation import Simulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.level_simulation import LevelSimulation


class SynthSimulation(Simulation):
    """
    Artificial (synthetic) simulation used for testing and examples.

    The simulation generates random samples from a specified distribution and
    optionally injects numerical error / NaN failures according to configuration.
    It implements the Simulation interface: provides `level_instance`, `calculate`,
    and `result_format` methods and a simple cost estimator `n_ops_estimate`.
    """

    n_nans = 0
    nan_fraction = 0
    len_results = 0
    result_dict = {}

    # Artificial simulation. Just random parameter + numerical error."""
    def __init__(self, config=None):
        """
        Initialize the synthetic simulation.

        :param config: Dict, optional configuration with keys:
                       - 'distr': a scipy.stats distribution object (default: stats.norm())
                       - 'complexity': exponent used for cost estimate (default: 2)
                       - 'nan_fraction': fraction of samples that should be returned as NaN
                       If config is None, a default normal distribution is used.
        """
        super().__init__()
        if config is None:
            config = dict(distr=stats.norm(), complexity=2)
        self.config = config

        # Static counters / settings used across instances
        SynthSimulation.n_nans = 0
        SynthSimulation.nan_fraction = config.get('nan_fraction', 0.0)
        SynthSimulation.len_results = 0

        # Indicates whether this simulation needs a workspace directory for samples
        self.need_workspace: bool = False

    @staticmethod
    def sample_fn(x, h):
        """
        Compute a (noisy) synthetic sample value for given distribution samples.

        :param x: Distribution sample(s) (scalar or array-like).
        :param h: Simulation step (resolution parameter). Typically small positive float.
        :return: Computed sample(s). Introduces small h-dependent perturbation:
                 x + h * sqrt(1e-4 + |x|). This can produce outliers for certain x.
        """
        return x + h * np.sqrt(1e-4 + np.abs(x))

    @staticmethod
    def sample_fn_no_error(x, h):
        """
        Compute a synthetic sample without introducing numerical error.

        :param x: Distribution sample(s) (scalar or array-like).
        :param h: Simulation step (ignored for this function).
        :return: The input sample(s) unchanged (identity mapping).
        """
        return x

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Create a LevelSimulation configured for a pair of fine/coarse level parameters.

        :param fine_level_params: List-like where the first element is the fine step size.
        :param coarse_level_params: List-like where the first element is the coarse step size.
        :return: LevelSimulation instance initialized with:
                 - config_dict containing 'fine.step', 'coarse.step', 'distr', and 'res_format'
                 - task_size estimated by n_ops_estimate(...)
        """
        config = dict()
        config["fine"] = {}
        config["coarse"] = {}
        config["fine"]["step"] = fine_level_params[0]
        config["coarse"]["step"] = coarse_level_params[0]
        config["distr"] = self.config["distr"]
        config["res_format"] = self.result_format()

        return LevelSimulation(config_dict=config, task_size=self.n_ops_estimate(fine_level_params[0]))

    @staticmethod
    def generate_random_samples(distr, seed, size):
        """
        Draw random samples from the provided scipy distribution reproducibly.

        :param distr: scipy.stats distribution object (must support .rvs()).
        :param seed: Integer seed used to construct a RandomState for reproducibility.
        :param size: Number of samples to draw.
        :return: Tuple (fine_samples, coarse_samples). For this synthetic sim both are identical.
                 May return [np.nan] to simulate a failed sample according to nan_fraction.
        """
        SynthSimulation.len_results += 1
        distr.random_state = np.random.RandomState(seed)
        y = distr.rvs(size=size)

        # Inject NaN failures up to configured fraction
        if SynthSimulation.n_nans / (1e-10 + SynthSimulation.len_results) < SynthSimulation.nan_fraction:
            SynthSimulation.n_nans += 1
            y = [np.nan]

        return y, y

    @staticmethod
    def calculate(config, seed):
        """
        Calculate fine and coarse samples and convert them to the expected result format.

        :param config: Dictionary containing simulation configuration (must include 'res_format',
                       'fine.step' and 'coarse.step' keys).
        :param seed: Integer RNG seed for reproducibility.
        :return: Tuple (fine_flat, coarse_flat) where both are 1D numpy arrays produced by
                 flattening the per-quantity/time/location arrays constructed below.
        :raises: Exception if any resulting sample contains NaN.
        """
        quantity_format = config["res_format"]

        # Generate base random values for fine and coarse (identical in this toy sim)
        fine_random, coarse_random = SynthSimulation.generate_random_samples(
            config["distr"],
            seed,
            np.prod(quantity_format[0].shape)
        )

        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        # Compute sample values for fine and coarse levels
        fine_result = SynthSimulation.sample_fn(fine_random, fine_step)

        if coarse_step == 0:
            coarse_result = np.zeros(len(fine_result))  # coarse = zero baseline if step==0
        else:
            coarse_result = SynthSimulation.sample_fn(coarse_random, coarse_step)

        # Fail hard if NaNs are present
        if np.any(np.isnan(fine_result)) or np.any(np.isnan(coarse_result)):
            raise Exception("result is nan")

        # Convert results into list-of-quantities × times × locations arrays and then flatten
        results = []
        for result in [fine_result, coarse_result]:
            quantities = []
            for quantity in quantity_format:
                if coarse_step == 0:
                    # replicate the same result for each location (coarse step 0 special case)
                    locations = np.array([result for _ in range(len(quantity.locations))])
                else:
                    # create simple distinct location-dependent arrays for demonstration
                    locations = np.array([result + i for i in range(len(quantity.locations))])
                # repeat across times
                times = np.array([locations for _ in range(len(quantity.times))])
                quantities.append(times)

            results.append(np.array(quantities))

        return results[0].flatten(), results[1].flatten()

    def n_ops_estimate(self, step):
        """
        Estimate number of operations (cost) for a sample at given step size.

        :param step: Level step size (h).
        :return: Estimated operation count (float). Uses configured complexity exponent.
        """
        return (1 / step) ** self.config['complexity'] * np.log(max(1 / step, 2.0))

    def result_format(self) -> List[QuantitySpec]:
        """
        Define the synthetic simulation's result format.

        :return: List[QuantitySpec] describing the shape, units, times and locations
                 for each reported quantity. This informs how `calculate` arranges
                 and flattens outputs.
        """
        spec1 = QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1, 2, 3], locations=['10', '20'])
        spec2 = QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=['30', '40'])
        # Alternative examples with numeric locations (commented out)
        # spec1 = QuantitySpec(name="length", unit="m", shape=(2, 1), times=[1, 2, 3],
        #                      locations=[(1, 2, 3), (4, 5, 6)])
        # spec2 = QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3],
        #                      locations=[(7, 8, 9), (10, 11, 12)])
        return [spec1, spec2]



class SynthSimulationWorkspace(SynthSimulation):
    """
    Synthetic simulation variant that requires a workspace (reads config from YAML).

    This subclass behaves like `SynthSimulation` but:
      - Reads distribution and nan_fraction from a YAML configuration file.
      - Declares `need_workspace = True` so sample files are written to/read from disk.
      - Supplies `common_files` (the YAML) to LevelSimulation so workspaces get that file.
    """

    n_nans = 0
    nan_fraction = 0
    len_results = 0
    result_dict = {}

    CONFIG_FILE = 'synth_sim_config.yaml'

    # Artificial simulation. Just random parameter + numerical error."""
    def __init__(self, config):
        """
        Initialize the workspace-capable synthetic simulation.

        :param config: Dict with at least:
            - "config_yaml": path to YAML configuration file (relative or absolute)
            Optionally may contain 'nan_fraction' as a fallback.
        """
        self.config_yaml = config["config_yaml"]

        # Reset static counters
        SynthSimulationWorkspace.n_nans = 0
        SynthSimulationWorkspace.nan_fraction = config.get('nan_fraction', 0.0)
        SynthSimulationWorkspace.len_results = 0

        # This simulation requires a workspace directory for sample execution
        self.need_workspace: bool = True

    @staticmethod
    def sample_fn(x, h):
        """
        Compute a (noisy) synthetic sample value for given distribution samples.

        :param x: Distribution sample(s) (scalar or array-like).
        :param h: Simulation step (resolution parameter).
        :return: Computed sample(s): x + h * sqrt(1e-4 + |x|).
        """
        return x + h * np.sqrt(1e-4 + np.abs(x))

    @staticmethod
    def sample_fn_no_error(x, h):
        """
        Identity sampling function (no added numerical error).

        :param x: Distribution sample(s).
        :param h: Simulation step (ignored).
        :return: x (unchanged).
        """
        return x

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Produce a LevelSimulation configured to use the YAML config as a common file.

        :param fine_level_params: list-like where first element is fine step size.
        :param coarse_level_params: list-like where first element is coarse step size.
        :return: LevelSimulation configured with:
                 - config_dict: containing 'fine.step', 'coarse.step', 'res_format'
                 - common_files: list containing the YAML path (so worker/workspace has it)
                 - task_size: small constant (1/job_weight) to simulate job weighting
                 - need_sample_workspace: True (this class requires workspace)
        """
        config = dict()
        config["fine"] = {}
        config["coarse"] = {}

        config["fine"]["step"] = fine_level_params[0]
        config["coarse"]["step"] = coarse_level_params[0]
        config["res_format"] = self.result_format()

        # Use a fixed job weight to keep task_size small (simulating many small jobs)
        job_weight = 20000

        return LevelSimulation(
            config_dict=config,
            common_files=[self.config_yaml],
            task_size=1.0 / job_weight,
            need_sample_workspace=self.need_workspace
        )

    @staticmethod
    def generate_random_samples(distr, seed, size):
        """
        Draw random samples based on YAML-specified distribution names.

        This implementation currently supports only the string "norm" which
        maps to scipy.stats.norm(loc=1, scale=2). A NotImplementedError is raised
        for other distribution identifiers.

        :param distr: Either a string identifier (e.g. "norm") or a scipy distribution.
        :param seed: Integer RNG seed used to create a RandomState for reproducibility.
        :param size: Integer number of samples to draw.
        :return: Tuple (fine_samples, coarse_samples) — identical arrays for this toy sim.
                 May return [np.nan] to simulate a failed sample according to nan_fraction.
        """
        SynthSimulationWorkspace.len_results += 1

        if distr == "norm":
            distr = stats.norm(loc=1, scale=2)
        else:
            raise NotImplementedError("Other distributions are not implemented yet")

        distr.random_state = np.random.RandomState(seed)
        y = distr.rvs(size=size)

        # Inject NaN failure if configured fraction not yet reached
        if SynthSimulationWorkspace.n_nans / (1e-10 + SynthSimulationWorkspace.len_results) < SynthSimulationWorkspace.nan_fraction:
            SynthSimulationWorkspace.n_nans += 1
            y = [np.nan]

        return y, y

    @staticmethod
    def calculate(config, seed):
        """
        Calculate fine and coarse samples (using values from the YAML config file).

        Workflow:
          1. Read YAML configuration (via _read_config) to get distribution and nan_fraction.
          2. Generate base random numbers (fine_random, coarse_random).
          3. Compute fine_result and coarse_result via sample functions.
          4. Assemble results into arrays shaped by res_format and flatten them.

        :param config: LevelSimulation.config_dict (must include 'res_format', 'fine.step', 'coarse.step').
        :param seed: Integer RNG seed.
        :return: Tuple (fine_flat, coarse_flat) — 1D numpy arrays produced by flattening quantities × times × locations.
        :raises: Exception if any computed result contains NaN.
        """
        # Load runtime YAML config (distribution name and nan_fraction)
        config_file = SynthSimulationWorkspace._read_config()
        SynthSimulationWorkspace.nan_fraction = config_file["nan_fraction"]

        quantity_format = config["res_format"]

        fine_random, coarse_random = SynthSimulationWorkspace.generate_random_samples(
            config_file["distr"], seed, np.prod(quantity_format[0].shape)
        )

        fine_step = config["fine"]["step"]
        coarse_step = config["coarse"]["step"]

        fine_result = SynthSimulation.sample_fn(fine_random, fine_step)

        if coarse_step == 0:
            coarse_result = np.zeros(len(fine_result))
        else:
            coarse_result = SynthSimulation.sample_fn(coarse_random, coarse_step)

        if np.any(np.isnan(fine_result)) or np.any(np.isnan(coarse_result)):
            raise Exception("result is nan")

        results = []
        for result in [fine_result, coarse_result]:
            quantities = []
            for quantity in quantity_format:
                if coarse_step == 0:
                    locations = np.array([result for _ in range(len(quantity.locations))])
                else:
                    locations = np.array([result + i for i in range(len(quantity.locations))])
                times = np.array([locations for _ in range(len(quantity.times))])
                quantities.append(times)
            results.append(np.array(quantities))

        return results[0].flatten(), results[1].flatten()

    def n_ops_estimate(self, step):
        """
        Estimate a synthetic operation count for the workspace-enabled simulation.
        :param step: Level step size.
        :return: Estimated operation cost (float). Uses a fixed exponent of 2 here.
        """
        return (1 / step) ** 2 * np.log(max(1 / step, 2.0))

    @staticmethod
    def _read_config():
        """
        Read the YAML configuration file (CONFIG_FILE) from the current working directory.

        The YAML is parsed using ruamel.yaml and should contain keys expected by this class
        (e.g. "distr" and "nan_fraction").

        :return: Parsed configuration dictionary.
        :raises: IOError / FileNotFoundError if the YAML file is missing.
        """
        with open(os.path.join(os.getcwd(), SynthSimulationWorkspace.CONFIG_FILE)) as file:
            yaml = ruyaml.YAML(typ='rt')
            config = yaml.load(file)

        return config
