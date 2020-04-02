"""
TODO:
- rename SimulationTest to SyntheticSimulation, generalize interface and move into src
- remove or integrate: _test_shooting(), variance_level, var_subsample*, anova, plot_*, impl_var_estimate

"""
import sys
import os
import random as rnd
import datetime
import numpy as np

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_path + '/../src/')
import mlmc.simulation
import mlmc.sample


class SimulationTest(mlmc.simulation.Simulation):
    # Artificial simulation. Just random parameter + numerical error."""
    def __init__(self, step, level_id, config):
        """
        :param config: Dict:
            'distr' scipy.stats.XYZ freezed distribution
            'complexity' number of FLOPS as function of step
        :param step:
        """
        super().__init__()
        self.config = config
        self.nan_fraction = config.get('nan_fraction', 0.05)
        self.n_nans = 0
        self.step = step
        self._result_dict = {}
        self._coarse_simulation = None
        self.coarse_sim_set = False

        #self.result_struct = [["value", "time"], [np.float, np.float]]
        #self.result_struct = [["value"], [np.float]]
        self.result_struct = [["value", "time", "position", "quantity", "unit"], [np.float, np.float, "S20", "S20", "S20"]]

    def _sample_fn(self, x, h):
        """
        Calculates the simulation sample
        :param x: Distribution sample
        :param h: Simluation step
        :return: sample
        """
        # This function can cause many outliers depending on chosen domain of moments function
        return x + h * np.sqrt(1e-4 + np.abs(x))

    def _sample_fn_no_error(self, x, h):
        """
        Calculates the simulation sample
        :param x: Distribution sample
        :param h: Simluation step
        :return: sample
        """
        return x

    def simulation_sample(self, tag=None, sample_id=0, time=None):
        """
        Run simulation
        :param sim_id:    Simulation id
        """
        x = self._input_sample
        h = self.step
        # Specific method is called according to pass parameters
        y = getattr(self, self.config['sim_method'])(x, h)  # self._sample_fn(x, h)

        if self.n_nans / (1e-10 + len(self._result_dict)) < self.nan_fraction:
            self.n_nans += 1
            y = np.nan

        self._result_dict[tag] = float(y)

        return mlmc.sample.Sample(sample_id=sample_id, directory=tag)

    def generate_random_sample(self):
        distr = self.config['distr']
        self._input_sample = distr.rvs(size=1)
        if self._coarse_simulation is not None:
            self._coarse_simulation._input_sample = self._input_sample

    def n_ops_estimate(self):
        return (1 / self.step) ** self.config['complexity'] * np.log(max(1 / self.step, 2.0))

    def set_coarse_sim(self, coarse_simulation=None):
        self._coarse_simulation = coarse_simulation
        self.coarse_sim_set = True

    def _extract_result(self, sample):
        """
        Extract simulation result
        :param sample: Sample instance
        :return: list of tuples
        """
        value = self._result_dict[sample.directory]
        quantities = ["quantity_1", "quantity_2", "quantity_3"]
        unit_dict = {"quantity_1": "unit_1", "quantity_2": "unit_2", "quantity_3": "unit_3"}
        result_values = []
        for i in range(3):
            time, position = self.generate_random_data()
            quantity = quantities[i]
            unit = unit_dict[quantity]
            result_values.append((value + i, time, position, quantity, unit))

        return result_values

    def generate_random_data(self):
        time = round(np.random.random(), 5)
        positions = ["frac_1", "frac_2", "frac_3", "frac_4", "frac_5", "frac_6", "frac_7", "frac_8", "frac_9"]
        position = rnd.choice(positions)
        # time = datetime.datetime.now()

        return time, position

