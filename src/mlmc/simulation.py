import numpy as np
import os, glob, shutil


class Simulation:
    """
    Parent class for simulations
    """
    def __init__(self, config=None, sim_param=0):
        """    
        :param config: Simulation configuration
        :param sim_param: Number of simulation steps
        """
        # Simulation result
        self._simulation_result = None
        self._config = config
        # Fine simulation step
        self._simulation_step = 0
        self.step = sim_param
        self._input_sample = []
        self._coarse_simulation = None

    def set_coarse_sim(self, coarse_sim):
        """
        Must be called, it is part of initialization.
        :param coarse_sim:
        :return:
        """
        pass

    def simulation_sample(self, tag):
        # Forward simulation for generated input.
        pass

    def n_ops_estimate(self):
        # complexity function
        return self.step

    def generate_random_sample(self):
        # Create new correlated random input for both fine and (related) coarse simulation
        pass

    def extract_result(self, sample_dir):
        """
        Extract simulation result
        :param sample_dir: Simulation sample directory
        :return: simulation result
        """
        try:
            result = self._extract_result(sample_dir)
            if result is np.nan:
                raise
        except:
            result = np.inf

        if result is np.inf:
            self.mv_failed_realizations(sample_dir)

        return result

    def _extract_result(self):
        return self._simulation_result

    @staticmethod
    def log_interpolation(sim_param_range, t_level):
        """
        Calculate particular simulation parameter
        :param sim_param_range: Tuple or list of two items, range of simulation parameters
        :param t_level: current level / total number of levels, it means 'precision' of current level fine simulation
        :return: int
        """
        assert 0 <= t_level <= 1
        return sim_param_range[0] ** (1 - t_level) * sim_param_range[1] ** t_level

    @classmethod
    def factory(cls, step_range, **kwargs):
        """
        Create specific simulation
        :param step_range: Simulation configuration
        :param **kwargs: Configuration of simulation
        :return: Particular simulation object
        """
        return lambda t_level, level_id, kw=kwargs: cls(Simulation.log_interpolation(step_range, t_level), level_id, **kw)

    def mv_failed_realizations(self, sample_dir):
        """
        Move failed simulation sample dir
        :param sample_dir: string
        :return: None
        """
        output_dir = os.path.abspath(sample_dir + "/../../..")
        sample_sub_dir = os.path.basename(os.path.normpath(sample_dir))
        destination = os.path.join(output_dir, "failed_realizations")

        # Make destination dir if not exists
        if not os.path.isdir(output_dir):
            os.mkdir(destination)

        if os.path.isdir(sample_dir):
            # Sample dir already exists in 'failed_realizations'
            if os.path.isdir(os.path.join(destination, sample_sub_dir)):
                similar_sample_dirs = glob.glob(os.path.join(destination, sample_sub_dir) + '_*')
                # Directory has more than one occurrence
                if len(similar_sample_dirs) > 0:
                    # Increment number of directory presents in dir name
                    sample_extension = os.path.basename(os.path.normpath(similar_sample_dirs[-1]))
                    sample_name = sample_extension.split("_")
                    sample_name[-1] = str(int(sample_name[-1]) + 1)
                    sample_extension = "_".join(sample_name)
                # Directory has just one occurrence
                else:
                    sample_extension = os.path.basename(os.path.normpath(sample_dir)) + "_1"
            else:
                sample_extension = sample_sub_dir

            # Copy sample directory to failed realizations dir
            shutil.copytree(sample_dir, destination + "/" + sample_extension)

            # Remove files in sample directory
            for file in os.listdir(sample_dir):
                file = os.path.abspath(os.path.join(sample_dir, file))
                if os.path.isdir(file):
                    shutil.rmtree(file)
                else:
                    os.remove(file)
