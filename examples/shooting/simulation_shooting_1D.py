import copy
import gstools
import numpy as np
import mlmc.random.correlated_field as cf
from typing import List
from mlmc.sim.simulation import Simulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.level_simulation import LevelSimulation


def create_corr_field(model='gauss', corr_length=0.1, dim=1, log=True, sigma=1, mode_no=1000):
    """
    Create a correlated random field wrapper used by the simulations.

    :param model: str, covariance model name. Supported values:
                  'gauss' (default), 'exp', 'TPLgauss', 'TPLexp', 'TPLStable'.
    :param corr_length: float, correlation length (len_scale) for the covariance model.
    :param dim: int, spatial dimension of the covariance model.
    :param log: bool, whether the generated random field is treated as log-field.
    :param sigma: float, standard deviation (sigma) for the random field.
    :param mode_no: int, number of Fourier modes (used by GSTools SRF).
    :return: cf.Field instance that wraps a GSToolsSpatialCorrelatedField configured
             with the selected covariance model.
    """
    if model == 'exp':
        model = gstools.Exponential(dim=dim, len_scale=corr_length)
    elif model == 'TPLgauss':
        model = gstools.TPLGaussian(dim=dim,  len_scale=corr_length)
    elif model == 'TPLexp':
        model = gstools.TPLExponential(dim=dim,  len_scale=corr_length)
    elif model == 'TPLStable':
        model = gstools.TPLStable(dim=dim,  len_scale=corr_length)
    else:
        model = gstools.Gaussian(dim=dim,  len_scale=corr_length)

    return cf.Field('conductivity', cf.GSToolsSpatialCorrelatedField(model, log=log, sigma=sigma, mode_no=mode_no))


class ShootingSimulation1D(Simulation):
    """
    Simple 1D "shooting" simulation used as example for MLMC workflow.

    The class implements:
      - level_instance(...) to create LevelSimulation configurations,
      - calculate(...) static method to produce paired (fine, coarse) samples,
      - result_format() describing the output QuantitySpec.
    """

    def __init__(self, config):
        """
        Initialize simulation factory.

        :param config: Dict containing simulation configuration. Expected keys include:
                       - 'start_position': np.ndarray shape (2,)
                       - 'start_velocity': np.ndarray shape (2,)
                       - 'area_borders': np.ndarray [xmin, xmax, ymin, ymax]
                       - 'max_time': float
                       - 'complexity': float used in n_ops_estimate
                       - 'fields_params': dict passed to create_corr_field()
        """
        super().__init__()
        self._config = config
        # If True, a workspace for each sample will be created (not used here).
        self.need_workspace: bool = False

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Create a LevelSimulation object for given fine and coarse level parameters.

        This constructs the simulation configuration for one MLMC level (fine/coarse pair)
        and returns a LevelSimulation that knows which calculate() function to call.

        :param fine_level_params: List[float], typically a single-element list containing the fine step size.
        :param coarse_level_params: List[float], typically a single-element list containing the coarse step size.
        :return: LevelSimulation configured with the derived config dictionary and task size.
        """
        config = copy.deepcopy(self._config)
        config["fine"] = {}
        config["coarse"] = {}
        config["fine"]["step"] = fine_level_params[0]
        config["coarse"]["step"] = coarse_level_params[0]
        config["res_format"] = self.result_format()

        # compute number of elements per level from complexity and step
        config["fine"]["n_elements"] = int(config["complexity"] / config["fine"]["step"])
        if config["coarse"]["step"] > 0:
            config["coarse"]["n_elements"] = int(config["complexity"] / config["coarse"]["step"])
        else:
            config["coarse"]["n_elements"] = 0

        return LevelSimulation(config_dict=config, calculate=ShootingSimulation1D.calculate,
                               task_size=self.n_ops_estimate(fine_level_params[0]))

    @staticmethod
    def calculate(config, seed):
        """
        Build random field and produce a paired (fine, coarse) simulation result.

        This is the static worker function used by LevelSimulation to generate one sample pair.
        It creates the correlated random field, samples it on the fine+coarse point set and
        runs the deterministic integrator for fine and coarse input arrays.

        :param config: dict, level-specific configuration produced by level_instance().
                       It must include 'fields_params', and 'fine'/'coarse' with 'n_elements'
                       and 'step' keys, plus other simulation settings.
        :param seed: int, RNG seed passed by sampler (currently unused inside; RNG is handled by field implementation).
        :return: tuple (fine_result, coarse_result), each is a scalar float (y coordinate at final time or NaN).
        """
        # create random field generator (GSTools wrapper)
        field = create_corr_field(**config['fields_params'])

        # create sample points (concatenated fine + coarse) and remember fine size
        points, n_fine_points = ShootingSimulation1D.create_points(config)

        # assign points to field and generate realization
        field.set_points(points)

        # sample fine and coarse inputs from the realizations
        fine_input_sample, coarse_input_sample = ShootingSimulation1D.generate_random_sample(
            field, coarse_step=config["coarse"]["step"], n_fine_elements=n_fine_points
        )

        # run simulator for fine and coarse inputs
        fine_res = ShootingSimulation1D._run_sample(config, fine_input_sample)
        coarse_res = ShootingSimulation1D._run_sample(config, coarse_input_sample)

        return fine_res, coarse_res

    @staticmethod
    def _run_sample(config, rnd_input_samples):
        """
        Run the explicit Euler "shooting" simulation for a given input array.

        The dynamics are:
            X_{k+1} = X_k + dt * V_k
            V_{k+1} = V_k + dt * F_k
        where F_k is drawn from the correlated field sample (rnd_input_samples).

        If the simulated projectile exits `area_borders`, the function returns np.nan.

        :param config: dict, simulation configuration (same structure as in calculate()).
        :param rnd_input_samples: np.ndarray-like, shape (n_elements,) with random forcing values.
        :return: float, y-coordinate of the projectile at the end of simulation
                 (or np.nan if it left the domain).
        """
        n_elements = len(rnd_input_samples)
        X = config["start_position"].copy()  # 2-vector [x, y]
        V = config['start_velocity'].copy()  # 2-vector
        y = 0.0

        # compute time step (if n_elements == 0 dt is unused and loop is skipped)
        if n_elements != 0:
            dt = config['max_time'] / n_elements

        # step through the random forcing and integrate
        for i in range(n_elements):
            # update position and velocity (explicit Euler)
            X = X + dt * V
            V = V + dt * rnd_input_samples[i]

            x = X[0]
            y = X[1]

            # check domain boundaries; if outside, record NaN and stop
            if x > config['area_borders'][1] or x < config['area_borders'][0] or \
               y > config['area_borders'][3] or y < config['area_borders'][2]:
                y = np.nan
                break

            # current time (not used beyond stopping condition)
            time = dt * (i + 1)
            if time >= config['max_time']:
                break

        return float(y)

    @staticmethod
    def create_points(config):
        """
        Create concatenated evaluation points for the random field for fine and coarse parts.

        Points are returned as an (n_fine + n_coarse, 1) array where each row is a coordinate
        at which the GSTools SRF will be evaluated. The first n_fine rows correspond to the fine mesh,
        the remaining rows (if any) to the coarse mesh.

        :param config: dict, level configuration that contains 'fine' and 'coarse' sub-dicts
                       with 'n_elements' integer entries and other simulation parameters.
        :return: tuple (points, n_fine_elements)
                 - points: np.ndarray shape (n_fine + n_coarse, 1)
                 - n_fine_elements: int number of fine elements (split index)
        """
        n_fine_elements = config["fine"]["n_elements"]
        n_coarse_elements = config["coarse"]["n_elements"]

        assert n_fine_elements > n_coarse_elements
        # build a 1D coordinate array for GSTools evaluation; here we use distance ~ velocity * time
        points = np.empty((n_fine_elements + n_coarse_elements, 1))
        points[:, 0] = np.concatenate((
            np.linspace(0, config["start_velocity"][0] * config["max_time"], n_fine_elements),
            np.linspace(0, config["start_velocity"][0] * config["max_time"], n_coarse_elements)
        ))

        return points, n_fine_elements

    @staticmethod
    def generate_random_sample(field, coarse_step, n_fine_elements):
        """
        Sample a realization from the provided correlated field and split it into fine and coarse parts.

        :param field: cf.Field instance created by create_corr_field(...) and already `set_points()` called.
        :param coarse_step: float, coarse level step (if 0 then no coarse part is returned).
        :param n_fine_elements: int, number of entries that belong to the fine-level part (split index).
        :return: tuple (fine_input_sample, coarse_input_sample)
                 - fine_input_sample: np.ndarray shape (n_fine_elements,)
                 - coarse_input_sample: np.ndarray shape (n_coarse_elements,) or [] if coarse_step == 0
        """
        field_sample = field.sample()
        fine_input_sample = field_sample[:n_fine_elements]
        coarse_input_sample = []
        if coarse_step != 0:
            coarse_input_sample = field_sample[n_fine_elements:]
        return fine_input_sample, coarse_input_sample

    def n_ops_estimate(self, step):
        """
        Estimate computational cost (# of operations) for a sample at a given step size.

        :param step: float, simulation step size for the fine level
        :return: float, heuristic estimate of cost used by MLMC to balance work across levels
        """
        return (1 / step) ** self._config['complexity'] * np.log(max(1 / step, 2.0))

    def result_format(self) -> List[QuantitySpec]:
        """
        Describe the output format of the simulation for MLMC.

        :return: List[QuantitySpec] describing the result vector(s). Here we return a single
                 scalar quantity 'target' representing y-position at final time.
        """
        spec1 = QuantitySpec(name="target", unit="m", shape=(1,), times=[10], locations=['0'])
        return [spec1]


