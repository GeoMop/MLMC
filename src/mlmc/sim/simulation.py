import numpy as np
import os, glob, shutil
from abc import ABCMeta
from abc import abstractmethod


class Simulation(metaclass=ABCMeta):
    """
    Parent class for simulations. Particular simulations always inherits from this one.
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
        # Precision of simulation
        self.step = sim_param
        # Simulation random input
        self._input_sample = []
        self._coarse_simulation = None

    @abstractmethod
    def set_coarse_sim(self, coarse_sim=None):
        """
        Set coarse simulations
        """

    @abstractmethod
    def simulation_sample(self, tag):
        """
        Forward simulation for generated input.
        """

    @abstractmethod
    def n_ops_estimate(self):
        """
        Estimate of the number of computational operations
        """

    @abstractmethod
    def generate_random_sample(self):
        """
        Create new correlated random input for both fine and (related) coarse simulation
        """

    def extract_result(self, sample):
        """
        Extract simulation result
        :param sample: Level simulation sample object
        :return: Modify sample
        """
        try:
            result, running_time = self._extract_result(sample)
            if result is np.nan:
                raise
        except:
            result = np.inf
            running_time = np.Inf

        if result is np.inf:
            Simulation._move_sample_dir(sample.directory)

        sample.result = result
        sample.running_time = running_time
        return sample

    @abstractmethod
    def _extract_result(self):
        """
        Get simulation sample result
        """

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
        :param step_range: Simulations step range
        :param **kwargs: Configuration of simulation
        :return: Particular simulation object
        """
        return lambda l_precision, l_id, kw=kwargs: cls(Simulation.log_interpolation(step_range, l_precision), l_id, **kw)

    @staticmethod
    def _move_sample_dir(sample_dir):
        """
        Move directory with failed simulation directory
        :param sample_dir: Sample directory
        :return: None
        """
        try:
            output_dir = os.path.abspath(sample_dir + "/../../..")
            sample_sub_dir = os.path.basename(os.path.normpath(sample_dir))

            target_directory = os.path.join(output_dir, "failed_realizations")

            # Make destination dir if not exists
            if not os.path.isdir(target_directory):
                os.mkdir(target_directory)

            if os.path.isdir(sample_dir):
                # Sample dir already exists in 'failed_realizations'
                if os.path.isdir(os.path.join(target_directory, sample_sub_dir)):
                    similar_sample_dirs = glob.glob(os.path.join(target_directory, sample_sub_dir) + '_*')
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
                Simulation._copy_tree(sample_dir, os.path.join(target_directory, sample_extension))

                # Remove files in sample directory
                for file in os.listdir(sample_dir):
                    file = os.path.abspath(os.path.join(sample_dir, file))
                    if os.path.isdir(file):
                        shutil.rmtree(file)
                    else:
                        os.remove(file)
        except Exception as exp:
            print(str(exp))

    @staticmethod
    def _copy_tree(source_dir, destination_dir):
        """
        Copy whole directory
        :param source_dir: absolute path to source directory
        :param destination_dir: absolute path to destination directory
        :return: None
        """
        # Top-down directory scan
        for src_dir, dirs, files in os.walk(source_dir):
            # Create destination directory if necessary
            if not os.path.exists(destination_dir):
                os.mkdir(destination_dir)
            # Copy files, use shutil.copyfile() method which doesn't need chmod permission
            for file in files:
                src_file = os.path.join(src_dir, file)
                dst_file = os.path.join(destination_dir, file)
                if not os.path.exists(dst_file):
                    shutil.copyfile(src_file, dst_file)
