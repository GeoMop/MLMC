import numpy as np
import mlmc.random.correlated_field as cf
import gstools
from typing import List
from mlmc.sim.simulation import Simulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.level_simulation import LevelSimulation


def create_corr_field(model='gauss', corr_length=0.1, dim=1, log=True, sigma=1, mode_no=1000):
    """
    Create a correlated random field wrapper (GSTools model -> mlmc.random.correlated_field.Field).

    :param model: str, covariance model name. Supported strings: 'gauss' (default), 'exp',
                  'TPLgauss', 'TPLexp', 'TPLStable'.
    :param corr_length: float, correlation length (len_scale) supplied to the GSTools model.
    :param dim: int, spatial dimension of the covariance model.
    :param log: bool, if True the field will be considered in log-space when sampling.
    :param sigma: float, multiplicative sigma parameter for the SRF generator.
    :param mode_no: int, number of Fourier modes used by GSTools SRF.
    :return: cf.Field — an mlmc.random.correlated_field.Field wrapping a GSTools SRF.
    """
    if model == 'exp':
        model = gstools.Exponential(dim=dim, len_scale=corr_length)
    elif model == 'TPLgauss':
        model = gstools.TPLGaussian(dim=dim, len_scale=corr_length)
    elif model == 'TPLexp':
        model = gstools.TPLExponential(dim=dim, len_scale=corr_length)
    elif model == 'TPLStable':
        model = gstools.TPLStable(dim=dim, len_scale=corr_length)
    else:
        model = gstools.Gaussian(dim=dim, len_scale=corr_length)

    return cf.Field('conductivity', cf.GSToolsSpatialCorrelatedField(model, log=log, sigma=sigma, mode_no=mode_no))


class ShootingSimulation2D(Simulation):
    """
    Example 2D shooting simulation used with MLMC sampler.

    Implements:
      - level_instance(...) to create per-level LevelSimulation config,
      - calculate(...) static method used to produce a (fine, coarse) sample pair,
      - result_format() describing output QuantitySpec.
    """

    def __init__(self, config):
        """
        Initialize the simulation factory.

        :param config: dict containing base simulation configuration. Expected keys:
                       - 'start_position': np.ndarray shape (2,)
                       - 'start_velocity': np.ndarray shape (2,)
                       - 'area_borders': np.ndarray [xmin, xmax, ymin, ymax]
                       - 'max_time': float
                       - 'complexity': float used by n_ops_estimate
                       - 'fields_params': dict passed to create_corr_field()
        """
        super().__init__()
        self.config = config
        # If True, a sample workspace will be created (not required here)
        self.need_workspace: bool = False

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Construct LevelSimulation for given fine/coarse parameters.

        This method mutates a copy of self.config into a level-specific config and
        returns a LevelSimulation that will call ShootingSimulation2D.calculate to
        generate samples for this level.

        :param fine_level_params: list-like, typically [fine_step]
        :param coarse_level_params: list-like, typically [coarse_step]
        :return: LevelSimulation configured for this level
        """
        # Update level-specific fields in self.config
        self.config["fine"] = {}
        self.config["coarse"] = {}
        self.config["fine"]["step"] = fine_level_params[0]
        self.config["coarse"]["step"] = coarse_level_params[0]
        self.config["res_format"] = self.result_format()

        # compute number of elements per level from complexity and step sizes
        self.config["fine"]["n_elements"] = int(self.config["complexity"] / self.config["fine"]["step"])
        if self.config["coarse"]["step"] > 0:
            self.config["coarse"]["n_elements"] = int(self.config["complexity"] / self.config["coarse"]["step"])
        else:
            self.config["coarse"]["n_elements"] = 0

        return LevelSimulation(config_dict=self.config, calculate=ShootingSimulation2D.calculate,
                               task_size=self.n_ops_estimate(fine_level_params[0]))

    @staticmethod
    def calculate(config, seed):
        """
        Build correlated fields and produce paired fine/coarse simulation results.

        This static function is intended to be executed by the LevelSimulation worker.
        It:
          - creates two independent correlated fields (x and y components),
          - sets evaluation points,
          - samples the fields and splits the realization into fine and coarse parts,
          - simulates the projectile dynamics for fine/coarse inputs.

        :param config: dict, level-specific configuration produced by level_instance().
        :param seed: int, RNG seed (currently not used directly; GSTools RNG is internal).
        :return: tuple (fine_result, coarse_result)
                 - fine_result: np.ndarray or scalar describing the fine-level output
                 - coarse_result: np.ndarray or scalar describing the coarse-level output
        """
        # Create independent correlated random fields for X and Y forcing
        field_x = create_corr_field(**config['fields_params'])
        field_y = create_corr_field(**config['fields_params'])

        # create concatenated points and get number of fine points
        points, n_fine_points = ShootingSimulation2D.create_points(config)
        field_x.set_points(points)
        field_y.set_points(points)

        # sample and split into fine and coarse inputs
        fine_input_sample, coarse_input_sample = ShootingSimulation2D.generate_random_sample(
            field_x, field_y, coarse_step=config["coarse"]["step"], n_fine_elements=n_fine_points
        )

        # run dynamics for fine and coarse inputs
        fine_res = ShootingSimulation2D._run_sample(config, fine_input_sample)
        coarse_res = ShootingSimulation2D._run_sample(config, coarse_input_sample)

        return fine_res, coarse_res

    @staticmethod
    def _run_sample(config, rnd_input_samples):
        """
        Run the explicit Euler integrator for the 2D shooting model.

        Dynamics:
            X_{k+1} = X_k + dt * V_k
            V_{k+1} = V_k + dt * F_k

        The method returns the final position vector X (or [nan, nan] if the projectile exits area_borders).

        :param config: dict, level-specific simulation configuration.
        :param rnd_input_samples: array-like with shape (n_elements, 2) or (n_elements,) depending on caller.
        :return: np.ndarray shape (2,) representing final [x, y] or [nan, nan] if out-of-bounds.
        """
        n_elements = len(rnd_input_samples)
        # Use copies so we don't mutate config objects
        X = np.array(config["start_position"], dtype=float).copy()
        V = np.array(config['start_velocity'], dtype=float).copy()

        # compute time step if there are elements
        if n_elements != 0:
            dt = config['max_time'] / n_elements

        # integrate step-by-step
        for i in range(n_elements):
            # update position
            X = X + dt * V

            # update velocity: rnd_input_samples[i] must be 2-vector (Fx, Fy)
            V = V + dt * rnd_input_samples[i]

            x = X[0]
            y = X[1]

            # if projectile leaves the bounding box, return NaNs
            if x > config['area_borders'][1] or x < config['area_borders'][0] or \
               y > config['area_borders'][3] or y < config['area_borders'][2]:
                X = np.array([np.nan, np.nan])
                break

            time = dt * (i + 1)
            if time >= config['max_time']:
                break

        return X

    @staticmethod
    def create_points(config):
        """
        Build concatenated evaluation points for the correlated fields (fine first, then coarse).

        The points are 1D coordinates used for the GSTools SRF evaluation; the first
        n_fine rows correspond to the fine mesh, the remainder to the coarse mesh.

        :param config: dict with 'fine' and 'coarse' sub-dicts containing 'n_elements'.
        :return: tuple (points, n_fine_elements)
                 - points: np.ndarray shape (n_fine + n_coarse, 1)
                 - n_fine_elements: int number of fine-level points
        """
        n_fine_elements = config["fine"]["n_elements"]
        n_coarse_elements = config["coarse"]["n_elements"]

        assert n_fine_elements > n_coarse_elements
        points = np.empty((n_fine_elements + n_coarse_elements, 1))
        points[:, 0] = np.concatenate((
            np.linspace(0, config["start_velocity"][0] * config["max_time"], n_fine_elements),
            np.linspace(0, config["start_velocity"][0] * config["max_time"], n_coarse_elements)
        ))
        return points, n_fine_elements

    @staticmethod
    def generate_random_sample(field_x, field_y, coarse_step, n_fine_elements):
        """
        Sample two correlated fields (x and y components) and split the realization into fine and coarse parts.

        :param field_x: cf.Field for x-component (already had set_points called).
        :param field_y: cf.Field for y-component (already had set_points called).
        :param coarse_step: float, coarse-level step size (if 0 there is no coarse part).
        :param n_fine_elements: int, how many samples belong to the fine-level portion.
        :return: tuple (fine_input_sample, coarse_input_sample)
                 - fine_input_sample: np.ndarray shape (n_fine_elements, 2) with x and y forcing
                 - coarse_input_sample: np.ndarray shape (n_coarse_elements, 2) or empty array if coarse_step==0
        """
        field_sample_x = field_x.sample()
        field_sample_y = field_y.sample()

        fine_input_sample = np.empty((n_fine_elements, 2))
        fine_input_sample[:, 0] = field_sample_x[:n_fine_elements]
        fine_input_sample[:, 1] = field_sample_y[:n_fine_elements]

        coarse_input_sample = np.empty((len(field_sample_x) - n_fine_elements, 2))
        if coarse_step != 0:
            coarse_input_sample[:, 0] = field_sample_x[n_fine_elements:]
            coarse_input_sample[:, 1] = field_sample_y[n_fine_elements:]
        return fine_input_sample, coarse_input_sample

    def n_ops_estimate(self, step):
        """
        Heuristic estimate of computational expense for a single fine-level sample.

        :param step: float, fine-level step size.
        :return: float, cost estimate used by MLMC scheduler.
        """
        return (1 / step) ** self.config['complexity'] * np.log(max(1 / step, 2.0))

    def result_format(self) -> List[QuantitySpec]:
        """
        Describe the result format (QuantitySpec) produced by calculate().

        :return: list of QuantitySpec objects describing outputs. Here a single 2D target vector.
        """
        spec1 = QuantitySpec(name="target", unit="m", shape=(2,), times=[10], locations=['0'])
        return [spec1]
