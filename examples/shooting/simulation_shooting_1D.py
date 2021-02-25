import gstools
import numpy as np
import mlmc.random.correlated_field as cf
from typing import List
from mlmc.sim.simulation import Simulation
from mlmc.quantity_spec import QuantitySpec
from mlmc.level_simulation import LevelSimulation


def create_corr_field(model='gauss', corr_length=0.125, dim=1, sigma=1, log=True):
    """
    Create random fields
    :return:
    """
    len_scale = corr_length * 2 * np.pi

    if model == 'exp':
        model = gstools.Exponential(dim=dim, var=sigma**2, len_scale=len_scale)
    elif model == 'TPLgauss':
        model = gstools.TPLGaussian(dim=dim, var=sigma**2, len_scale=len_scale)
    elif model == 'TPLexp':
        model = gstools.TPLExponential(dim=dim, var=sigma**2, len_scale=len_scale)
    elif model == 'TPLStable':
        model = gstools.TPLStable(dim=dim, var=sigma**2, len_scale=len_scale)
    else:
        model = gstools.Gaussian(dim=dim, var=sigma**2, len_scale=len_scale)

    return cf.Field('conductivity', cf.GSToolsSpatialCorrelatedField(model, log=log))


class ShootingSimulation1D(Simulation):

    n_nans = 0
    nan_fraction = 0
    len_results = 0
    result_dict = {}

    # Artificial simulation. Just random parameter + numerical error."""
    def __init__(self, config):
        """
        :param config: Dict:
                distr= particular distribution,
                complexity=2,
                nan_fraction=fraction of failed samples
                sim_method=used method for calculating sample result
        """
        super().__init__()
        self.config = config
        ShootingSimulation1D.n_nans = 0
        ShootingSimulation1D.nan_fraction = config.get('nan_fraction', 0.0)
        ShootingSimulation1D.len_results = 0
        # This attribute is obligatory
        self.need_workspace: bool = False

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """

        :param fine_level_params:
        :param coarse_level_params:
        :return:
        """
        self.config["fine"] = {}
        self.config["coarse"] = {}
        self.config["fine"]["step"] = fine_level_params[0]
        self.config["coarse"]["step"] = coarse_level_params[0]
        self.config["res_format"] = self.result_format()
        self.config["fine"]["n_elements"] = int(self.config["complexity"] / self.config["fine"]["step"])
        if self.config["coarse"]["step"] > 0:
            self.config["coarse"]["n_elements"] = int(self.config["complexity"] / self.config["coarse"]["step"])
        else:
            self.config["coarse"]["n_elements"] = 0

        return LevelSimulation(config_dict=self.config, calculate=ShootingSimulation1D.calculate,
                               task_size=self.n_ops_estimate(fine_level_params[0]))

    @staticmethod
    def calculate(config, seed):
        """
        Calculate fine and coarse sample and also extract their results
        :param config: dictionary containing simulation configuration
        :param seed: random number generator seed
        :return: np.ndarray, np.ndarray
        """
        # Create random field structure
        field = create_corr_field(**config['fields_params'])
        points, n_fine_points = ShootingSimulation1D.create_points(config)
        field.set_points(points)

        fine_input_sample, coarse_input_sample = ShootingSimulation1D.generate_random_sample(field,
                                                                                           coarse_step=config["coarse"]["step"],
                                                                                           n_fine_elements=n_fine_points)

        fine_res = ShootingSimulation1D._run_sample(config, fine_input_sample, config["fine"]["n_elements"])
        coarse_res = ShootingSimulation1D._run_sample(config, fine_input_sample, config["fine"]["n_elements"])

        return fine_res, coarse_res

    @staticmethod
    def _run_sample(config, rnd_input_samples, n_elements):
        """
        Simulation of 1D shooting
        """
        x, y, time, n = 0, 0, 0, 0
        X = config["start_position"]
        V = config['start_velocity']

        # Time step
        if n_elements != 0:
            dt = config['max_time'] / n_elements

        # Loop through random array F
        for i in range(n_elements):
            # New coordinates
            X = X + dt * V
            # New vector of speed
            V = V + dt * rnd_input_samples[i]

            x = X[0]
            y = X[1]

            if x > config['area_borders'][1] or x < config['area_borders'][0] or\
                    y > config['area_borders'][3] or y < config['area_borders'][2]:
                y = np.nan
                break

            time = dt * (i + 1)

            # End simulation if time is bigger then maximum time
            if time >= config['max_time']:
                break

        return float(y)

    @staticmethod
    def create_points(config):
        n_fine_elements = config["fine"]["n_elements"]
        n_coarse_elements = config["coarse"]["n_elements"]

        assert n_fine_elements > n_coarse_elements
        points = np.empty((n_fine_elements + n_coarse_elements, 1))
        points[:, 0] = np.concatenate((np.linspace(0, config["start_velocity"][0]*config["max_time"],
                                                   n_fine_elements),
                                       np.linspace(0, config["start_velocity"][0]*config["max_time"],
                                                   n_coarse_elements)))

        return points, n_fine_elements

    @staticmethod
    def generate_random_sample(field, coarse_step, n_fine_elements):
        """
        Generate random field, both fine and coarse part.
        :return: List, List
        """
        field_sample = field.sample()
        fine_input_sample = field_sample[:n_fine_elements]
        coarse_input_sample = []
        if coarse_step != 0:
            coarse_input_sample = field_sample[n_fine_elements:]
        return fine_input_sample, coarse_input_sample

    def n_ops_estimate(self, step):
        return (1 / step) ** self.config['complexity'] * np.log(max(1 / step, 2.0))

    def result_format(self) -> List[QuantitySpec]:
        """
        Result format
        :return:
        """
        spec1 = QuantitySpec(name="target", unit="m", shape=(1,), times=[10], locations=['0'])
        return [spec1]

