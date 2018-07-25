import numpy as np
import numpy.ma as ma


class Level:
    """
    Call Simulation methods
    There are information about random variable - average, dispersion, number of simulation, ...
    TODO:
    workflow:
    - queue simulations ( need simulation object for prepare input, need pbs as executor
    - check for finished simulations (need pbs
    - estimates for collected samples ... this can be in separate class as it is independeent of simulation
    .. have to reconsider in context of Analysis
    """

    def __init__(self, level_idx, sim_factory, previous_level_sim, precision):
        """
        :param sim_factory: method that create instance of particular simulation class
        :param previous_level_sim: fine simulation on previous level
        :param moments_object: object for calculating statistical moments
        :param precision: current level number / total number of all levels
        """
        # Reference to all created simulations.
        #self.simulations = []

        self.is_zero_level = (level_idx == 0)
        self.level_idx = level_idx
        # Instance of object Simulation
        self.fine_simulation = sim_factory(precision)
        # TODO: coarse_simulation can be different to previous_level_sim if they have same mean value
        self.coarse_simulation = previous_level_sim
        self.fine_simulation.set_coarse_sim(self.coarse_simulation)

        # Target number of samples for the level
        self.target_n_samples = None
        # Last moments function
        self._last_moments_fn = None
        # Moments from coarse and fine samples
        self.last_moments_eval = None

        self.reset()

    def reset(self):
        # Currently running simulations
        self.running_simulations = []
        # Collected simulations, all results of simulations. Including Nans and None ...
        self.finished_simulations = []
        # Target number of samples.
        self.target_n_samples = 5
        # Collected samples (array may be partly filled)
        # Without any order, without Nans and inf. Only this is used for estimates.
        self._sample_values = np.empty((self.target_n_samples, 2))
        # Number of valid samples in _sample_values.
        self._n_valid_samples = 0
        # Possible subsample indices.
        self.sample_indices = None
        # Array of indices of nan samples (in fine or coarse sim)
        self.nan_samples = []
        # Cache evaluated moments.
        self._last_moments_fn = None

    def set_target_n_samples(self, n_samples):
        """
        Set target number of samples for the level.
        :param n_samples: Number of samples
        :return: None
        """
        self.target_n_samples = max(self.target_n_samples, n_samples)

    @property
    def sample_values(self):
        """
        Get valid level samples
        :return: array
        """
        return self._sample_values[:self._n_valid_samples]

    def add_sample(self, id, sample_pair):
        """
        Add samples pair to rest of samples
        :param id: sample id
        :param sample_pair: Fine and coarse result
        :return: None
        """
        fine, coarse = sample_pair
        # Samples are not finite
        if not np.isfinite(fine) or not np.isfinite(coarse):
            self.nan_samples.append(id)
            return
        # Enlarge matrix of samples
        if self._n_valid_samples == self._sample_values.shape[0]:
            self.enlarge_samples(2*self._n_valid_samples)

        # Add fine and coarse sample
        self._sample_values[self._n_valid_samples, :] = (fine, coarse)
        self._n_valid_samples += 1

    def enlarge_samples(self, size):
        """
        Enlarge matrix of samples
        :param size: New sample matrix size
        :return: None
        """
        # Enlarge sample matrix
        new_values = np.empty((size, 2))
        new_values[:self._n_valid_samples] = self._sample_values[:self._n_valid_samples]
        self._sample_values = new_values

    @property
    def n_total_samples(self):
        """
        Number of all level samples
        :return: int
        """
        return len(self.running_simulations) + len(self.finished_simulations)

    @property
    def n_samples(self):
        # Number of samples used for estimates.
        if self.sample_indices is None:
            return self._n_valid_samples
        else:
            return len(self.sample_indices)

    def _get_sample_tag(self, char):
        return "L{:02d}_{}_S{:07d}".format(self.level_idx, char, self.n_total_samples)

    def n_ops_estimate(self):
        """
        :return: fine simulation n
        """
        return self.fine_simulation.n_ops_estimate()

    def make_sample_pair(self):
        """
        Generate new random samples for fine and coarse simulation objects
        :return: (fine_sample, coarse_sample); identification tuples for the related fine and coarse sample
        """

        # All levels have fine simulation
        idx = self.n_total_samples
        self.fine_simulation.generate_random_sample()
        tag = self._get_sample_tag('F')
        fine_sample = self.fine_simulation.simulation_sample(tag)
        if self.coarse_simulation is not None:
            tag = self._get_sample_tag('C')
            coarse_sample = self.coarse_simulation.simulation_sample(tag)
        else:
            # Zero level have no coarse simulation.
            coarse_sample = None
        return [self.level_idx, idx, fine_sample, coarse_sample, None]

    def fill_samples(self, logger):
        """
        Generate samples up to target number set through 'set_target_n_samples'.
        Simulations are planed for execution, but sample values are collected in
        :param logger: FlowPbs instance
        :return: None
        """

        orig_n_running = len(self.running_simulations)
        if self.target_n_samples > self.n_total_samples:
            self.enlarge_samples(self.target_n_samples)

            # Create pair of fine and coarse simulations and add them to list of all running simulations
            while self.n_total_samples < self.target_n_samples:
                self.running_simulations.append(self.make_sample_pair())
        # log new simulation pairs
        logger.log_simulations(self.running_simulations[orig_n_running:])

    def collect_samples(self, logger):
        """
        Extract values for finished simulations.
        :return: Number of simulations to finish yet.
        """
        # Still running some simulations
        #logger.check_finished()
        # Loop through pair of running simulations
        orig_n_finised = len(self.finished_simulations)
        new_running = []
        for (level, idx, fine_sim, coarse_sim, value) in self.running_simulations:
            try:
                fine_result = self.fine_simulation.extract_result(fine_sim)
                fine_done = fine_result is not None
                
                if self.is_zero_level:
                    coarse_result = 0.0
                    coarse_done = True
                else:
                    coarse_result = self.coarse_simulation.extract_result(coarse_sim)
                    coarse_done = coarse_result is not None

                if fine_done and coarse_done:
                    if np.isnan(fine_result):
                        fine_result = np.inf
                    if np.isnan(coarse_result):
                        coarse_result = np.inf
                    # collect values
                    self.finished_simulations.append([self.level_idx, idx, fine_sim, coarse_sim, [fine_result, coarse_result]])
                    self.add_sample(idx, (fine_result, coarse_result))
                else:
                    new_running.append([level, idx, fine_sim, coarse_sim, value])

            except ExpWrongResult as e:
                print(e.message)

        self.running_simulations = new_running

        # log new collected simulation pairs
        new_finished = self.finished_simulations[orig_n_finised:]
        #new_values = self.sample_values[orig_n_finised:, :]
        #assert len(new_values) >= len(new_finished)
        logger.log_simulations(new_finished, collected=True)
        return len(self.running_simulations)

    def subsample(self, size):
        if size is None:
            self.sample_indices = None
        else:
            assert 0 < size < self._n_valid_samples, "0 < {} < {}".format(size, self._n_valid_samples)
            self.sample_indices = np.random.choice(np.arange(self._n_valid_samples, dtype=int), size=size)

    def evaluate_moments(self, moments_fn):
        """
        Evaluating moments from moments function
        :param moments_fn: Moment evaluation functions
        :return: tuple
        """

        # Current moment functions are different from last moment functions
        #if moments_fn != self._last_moments_fn:
        samples = self.sample_values

        # Moments from fine samples
        moments_fine = moments_fn(samples[:, 0])

        # For first level moments from coarse samples are zeroes
        if self.is_zero_level:
            moments_coarse = np.zeros_like(np.eye(len(moments_fine), moments_fn.size))
        else:
            moments_coarse = moments_fn(samples[:, 1])
        # Set last moments function
        self._last_moments_fn = moments_fn
        # Moments from fine and coarse samples
        self.last_moments_eval = moments_fine, moments_coarse

        # Remove outliers
        if self.last_moments_eval is not None:
            self.remove_outliers_moments()

        if self.sample_indices is None:
            return self.last_moments_eval
        else:
            m_fine, m_coarse = self.last_moments_eval
            return m_fine[self.sample_indices, :], m_coarse[self.sample_indices, :]

    def remove_outliers_moments(self):
        """
        Remove moments from outliers from fine and course moments
        :return: None
        """
        # Fine and coarse moments mask
        mask_fine = ma.masked_invalid(self.last_moments_eval[0]).mask
        mask_coarse = ma.masked_invalid(self.last_moments_eval[1]).mask

        # Common mask for coarse and fine
        mask_fine_coarse = np.logical_or(mask_fine, mask_coarse)[:, -1]

        # New moments without outliers
        self.last_moments_eval = self.last_moments_eval[0][~mask_fine_coarse], self.last_moments_eval[1][~mask_fine_coarse]

        # Remove outliers also from sample values
        self._sample_values = self._sample_values[:self._n_valid_samples][~mask_fine_coarse]

        # Set new number of valid samples
        self._n_valid_samples = len(self._sample_values)

    def estimate_diff_var(self, moments_fn):
        """
        Estimate moments variance
        :param moments_fn: Moments evaluation function
        :return: tuple (variance vector, length of moments)
        """
        # n_samples = n_dofs + 1 >= 7 leads to probability 0.9 that estimate is whithin range of 10% error from true variance
        assert self.n_samples > 1
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn)
        var_vec = np.var(mom_fine - mom_coarse, axis=0, ddof=1)
        return var_vec, len(mom_fine)

    def estimate_diff_mean(self, moments_fn):
        """
        Estimate moments mean
        :param moments_fn: 
        :return: np.array, moments mean vector
        """
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn)
        mean_vec = np.mean(mom_fine - mom_coarse, axis=0)
        return mean_vec

    def sample_range(self):
        fine_sample = self.sample_values[:, 0]
        q1, q3 = np.percentile(fine_sample, [25, 75])
        iqr = 3*(q3 - q1)
        min_sample = np.min(fine_sample)
        l = min(min_sample, q1-iqr)
        if min_sample > 0.0:    # guess that we have positive distribution
            l = min_sample
        r = max(np.max(fine_sample), q3+iqr)

        return l, r


class ExpWrongResult(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
        self.message = "Wrong simulation result"
