import time
import numpy as np
from mlmc.mc_level import Level
from mlmc.sim.simulation import Simulation
import mlmc.tool.hdf as hdf


class MLMC:
    """
    Multilevel Monte Carlo method
    """

    def __init__(self, n_levels, sim_factory, step_range, process_options):
        """
        :param n_levels: Number of levels
        :param sim_factory: Object of simulation
        :param step_range: Simulations step range
        :param process_options: Options for processing mlmc samples
                                'output_dir' - directory with sample logs
                                'regen_failed' - bool, if True then failed simulations are generated again
                                'keep_collected' - bool, if True then dirs with finished simulations aren't removed
        """
        # Object of simulation
        self.simulation_factory = sim_factory
        # Array of level objects
        self.levels = []
        self._n_levels = n_levels
        self.step_range = step_range

        self._process_options = process_options
        # Number of simulation steps through whole mlmc
        self.target_time = None
        # Total variance
        self.target_variance = None

        # Create hdf5 file - contains metadata and samples at levels
        self._hdf_object = hdf.HDF5(file_name="mlmc_{}.hdf5".format(n_levels), work_dir=self._process_options['output_dir'])

    def load_from_file(self, file=None):
        """
        Run mlmc according to setup parameters, load setup from hdf file {n_levels, step_range} and create levels
        :return: None
        """
        if file is not None:
            self._hdf_object.file_name = file
        # Load mlmc params from file
        self._hdf_object.load_from_file()
        self._n_levels = self._hdf_object.n_levels
        self.step_range = self._hdf_object.step_range

        # Create mlmc levels
        self.create_levels()

    def create_new_execution(self):
        """
        Save mlmc main attributes {n_levels, step_range} and create levels
        :return: None
        """
        self._hdf_object.clear_groups()
        self._hdf_object.init_header(step_range=self.step_range,
                                     n_levels=self._n_levels)
        self.create_levels()

    def create_levels(self):
        """
        Create level objects, each level has own level logger object
        :return: None
        """
        for i_level in range(self._n_levels):
            previous_level = self.levels[-1] if i_level else None
            if self._n_levels == 1:
                level_param = 1
            else:
                level_param = i_level / (self._n_levels - 1)

            # Create level
            level = Level(self.simulation_factory, previous_level, level_param, i_level,
                          self._hdf_object.add_level_group(str(i_level)),
                          self._process_options['regen_failed'], self._process_options['keep_collected'])
            self.levels.append(level)

    @property
    def n_levels(self):
        """
        Number of levels
        """
        if len(self.levels) > 0:
            return len(self.levels)

    @property
    def n_samples(self):
        """
        Level samples
        """
        return np.array([l.n_samples for l in self.levels])

    @property
    def n_nan_samples(self):
        """
        Level nan samples
        """
        return np.array([len(l.nan_samples) for l in self.levels])

    @property
    def sim_steps(self):
        return np.array([Simulation.log_interpolation(self.step_range, lvl.step) for lvl in self.levels])

    def sample_range(self, n0, nL):
        """
        Geometric sequence of L elements decreasing from n0 to nL.
        Useful to set number of samples explicitly.
        :param n0: int
        :param nL: int
        :return: np.array of length L = n_levels.
        """
        return np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), self.n_levels))).astype(int)

    def set_initial_n_samples(self, n_samples=None):
        """
        Set target number of samples for each level
        :param n_samples: array of number of samples
        :return: None
        """
        if n_samples is None:
            n_samples = [100, 3]
        # Num of samples to ndarray
        n_samples = np.atleast_1d(n_samples)

        # Just maximal number of samples is set
        if len(n_samples) == 1:
            n_samples = np.array([n_samples[0], 3])

        # Create number of samples for all levels
        if len(n_samples) == 2:
            n0, nL = n_samples
            n_samples = self.sample_range(n0, nL)

        for i, level in enumerate(self.levels):
            level.set_target_n_samples(int(n_samples[i]))

    # def set_target_time(self, target_time):
    #     """
    #     For each level counts new N according to target_time
    #     :return: array
    #     TODO: Have a linear model to estimate true time per sample as a function of level step.
    #           This needs some test sampling... that is same as with variance estimates.
    #     """
    #     vars =
    #     amount = self._count_sum()
    #     # Loop through levels
    #     # Count new number of simulations for each level
    #     for level in self.levels:
    #         new_num_of_sim = np.round((target_time * np.sqrt(level.variance / level.n_ops_estimate()))
    #                                   / amount).astype(int)
    #
    #         self.num_of_simulations.append(new_num_of_sim)

    # def reset_moment_fn(self, moments_fn):
    #     for level in self.levels:
    #         level.reset_moment_fn(moments_fn)

    def process_adding_samples(self, n_estimated, pbs, sleep, add_coef=0.1):
        """
        Process adding samples
        :param n_estimated: Number of estimated samples on each level, list
        :param pbs: src.Pbs instance
        :param sleep: Sample waiting time
        :param add_coef: default value 0.1
        :return: bool, if True adding samples is complete
        """
        # Get default scheduled samples
        n_scheduled = self.l_scheduled_samples()

        # New scheduled sample will be 10 percent of difference
        # between current number of target samples and new estimated one
        # If 10 percent of estimated samples is greater than difference between estimated and scheduled samples,
        # set scheduled samples to estimated samples
        new_scheduled = np.where((n_estimated * add_coef) > (n_estimated - n_scheduled),
                                 n_estimated,
                                 n_scheduled + (n_estimated - n_scheduled) * add_coef)

        n_scheduled = np.ceil(np.where(n_estimated < n_scheduled,
                                       n_scheduled,
                                       new_scheduled))
        # Levels where estimated are greater than scheduled
        greater_items = np.where(np.greater(n_estimated, n_scheduled))[0]

        # Scheduled samples and wait until at least half of the samples are done
        self.set_scheduled_and_wait(n_scheduled, greater_items, pbs, sleep)

        return np.all(n_estimated[greater_items] == n_scheduled[greater_items])

    def set_scheduled_and_wait(self, n_scheduled, greater_items, pbs, sleep, fin_sample_coef=0.5):
        """
        Scheduled samples on each level and wait until at least half of the samples is done
        :param n_scheduled: ndarray, number of scheduled samples on each level
        :param greater_items: Items where n_estimated is greater than n_scheduled
        :param pbs: Pbs script generator object
        :param sleep: Time waiting for samples
        :param fin_sample_coef: The proportion of samples to finished for further estimate
        :return: None
        """

        # Set scheduled samples and run simulations
        self.set_level_target_n_samples(n_scheduled)
        self.refill_samples()
        # Use PBS job scheduler
        if pbs is not None:
            pbs.execute()

        # Finished level samples
        n_finished = np.array([level.get_n_finished() for level in self.levels])

        # Wait until at least half of the scheduled samples are done on each level
        while np.any(n_finished[greater_items] < fin_sample_coef * n_scheduled[greater_items]):
            # Wait a while
            time.sleep(sleep)
            n_finished = np.array([level.get_n_finished() for level in self.levels])

    def l_scheduled_samples(self):
        """
        Get all levels target number of samples
        :return: list 
        """
        return [level.target_n_samples for level in self.levels]

    def set_level_target_n_samples(self, n_samples, fraction=1.0):
        """
        Set level number of target samples
        :param n_samples: list, each level target samples
        :param fraction: Use just fraction of total samples
        :return: None
        """
        for level, n in zip(self.levels, n_samples):
            level.set_target_n_samples(int(n * fraction))

    def refill_samples(self):
        """
        For each level set fine and coarse simulations and generate samples in reverse order (from the finest sim)
        :return: None
        """
        # Set level coarse sim, it creates also fine simulations if its None
        for level in self.levels:
            level.set_coarse_sim()

        # Generate level's samples in reverse order
        for level in reversed(self.levels):
            level.fill_samples()

    def wait_for_simulations(self, sleep=0, timeout=None):
        """
        Waiting for running simulations
        :param sleep: time for doing nothing
        :param timeout: maximum time for waiting on running simulations
        :return: int, number of running simulations
        """
        if timeout is None:
            timeout = 0
        elif timeout <= 0:
            return 1
        n_running = 1
        t0 = time.clock()
        while n_running > 0:
            n_running = 0
            for level in self.levels:
                n_running += level.collect_samples()

            time.sleep(sleep)
            if 0 < timeout < (time.clock() - t0):
                break

        return n_running

    def subsample(self, sub_samples=None):
        """
        :param sub_samples: None - use all generated samples
                    array of ints, shape = n_levels; choose given number of sub samples from computed samples
        :return: None
        """
        if sub_samples is None:
            sub_samples = [None] * self.n_levels
        assert len(sub_samples) == self.n_levels, "{} != {}".format(len(sub_samples), self.n_levels)
        for ns, level in zip(sub_samples, self.levels):
            level.subsample(ns)

    def subsample_by_indices(self, sample_indices=None):
        """
        :param sample_indices: None - use all generated samples
                    array - boolean mask, shape = len(Level.sample_values)
        :return: None
        """
        for level in self.levels:
            level.subsample(size=None, sample_indices=sample_indices)

    def update_moments(self, moments_fn):
        for level in self.levels:
            level.evaluate_moments(moments_fn, force=True)

    def clean_levels(self):
        """
        Reset all levels
        :return: None
        """
        for level in self.levels:
            level.reset()

    def select_values(self, condition, selected_param=None):
        """
        Select values from sample results
        Each sample results can contains more quantities and other parameters. This method allows us to select results
        with particular parameter's values
        :param condition:
        :return: None
        """
        for level in self.levels:
            level.select(condition, selected_param)

    def clean_select(self):
        """
        Cancel param selection, so we use all collected simulation result
        :return: None
        """
        for level in self.levels:
            level.clean_select()

    def clean_subsamples(self):
        """
        Clean level subsamples
        :return: None
        """
        for level in self.levels:
            level.subsample(None)

    def get_sample_times(self):
        """
        The total average duration of one sample per level (fine + coarse together)
        :return: list
        """
        return [level.sample_time() for level in self.levels]
