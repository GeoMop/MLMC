import numpy as np
import scipy as sc
import scipy.stats
import uuid
import os

class Level:
    """
    Call Simulation methods
    There are information about random variable - average, dispersion, number of simulation, ...
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

        self.reset()
        self.running_simulations = []
        self.finished_simulations = []

        # Initialization of variables
        #self._result = []
        # Default number of simulations is 10
        # that is enough for estimate variance
        self.target_n_samples=7
        # Array for collected sample pairs (fine, coarse)
        self.sample_values = np.empty( (self.target_n_samples, 2) )
        self._last_moments_fn = None

    def reset(self):
        # Currently running simulations
        self.running_simulations = []
        # Collected simulations
        self.finished_simulations = []
        # Target number of samples.
        self.target_n_samples=5
        # Collected samples (array may be partly filled)
        self.sample_values = np.empty( (self.target_n_samples, 2) )


    def set_target_n_samples(self, n_samples):
        """
        Set new target number of samples for the level.
        :param n_samples:
        :return:
        """
        self.target_n_samples = max(self.target_n_samples, n_samples)

    @property
    def n_total_samples(self):
        return len(self.running_simulations) + len(self.finished_simulations)

    @property
    def n_collected_samples(self):
        return len(self.finished_simulations)

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
        :return: (fine_sample, coarse_sample); identificaion tuples for the related fine and coarse sample
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

        return (idx, fine_sample, coarse_sample)

 
    def fill_samples(self, logger):
        """
        Generate samples up to target number set through 'set_target_n_samples'.
        Simulations are planed for execution, but sample values are collected in
        :return: None
        """

        orig_n_running = len(self.running_simulations)
        if self.target_n_samples > self.n_total_samples:
            # Enlarge array for sample values.
            new_values = np.empty( (self.target_n_samples, 2) )
            n_collected = len(self.sample_values)
            new_values[:n_collected, : ] = self.sample_values
            self.sample_values = new_values

            # Create pair of fine and coarse simulations and add them to list of all running simulations
            while self.n_total_samples < self.target_n_samples:
                self.running_simulations.append(self.make_sample_pair())
        # log new simulation pairs
        logger.log_simulations(self.level_idx, self.running_simulations[orig_n_running:])


    def collect_samples(self, logger):
        """
        Extract values for finished simulations.
        :return: Number of simulations to finish yet.
        """
        # Still running some simulations

        # Loop through pair of running simulations
        orig_n_finised = len(self.finished_simulations)
        new_running = []
        for (idx, fine_sim, coarse_sim) in self.running_simulations:
            try:
                fine_result = self.fine_simulation.extract_result(fine_sim)
                fine_done = fine_result is not None

                if self.is_zero_level:
                   coarse_result = 0.0
                   coarse_done = True
                else:
                    coarse_result = self.coarse_simulation.extract_result(coarse_sim)
                    coarse_done = fine_result is not None

                if fine_done and coarse_done:
                    # collect values
                    self.finished_simulations.append( (idx, fine_sim, coarse_sim) )
                    self.sample_values[idx, :] = (fine_result, coarse_result)
                    # TODO: mark to sample file
                else:
                    new_running.append( (idx, fine_sim, coarse_sim) )

            except ExpWrongResult as e:
                print(e.message)
        self.running_simulations = new_running

        # log new collected simulation pairs
        new_finished = self.finished_simulations[orig_n_finised:]
        #new_values = self.sample_values[orig_n_finised:, :]
        #assert len(new_values) >= len(new_finished)
        logger.log_simulations(self.level_idx,
                               new_finished,
                               values=self.sample_values)
        return len(self.running_simulations)

    def evaluate_moments(self,  moments_fn):
        if moments_fn != self._last_moments_fn:
            samples = self.sample_values[:self.n_collected_samples, :]
            moments_fine = moments_fn(samples[:, 0])
            if self.is_zero_level:
                moments_coarse = np.zeros_like(moments_fine)
            else:
                moments_coarse = moments_fn(samples[:, 1])
                self._last_moments_fn = moments_fn
            self.last_moments_eval = moments_fine, moments_coarse
        return self.last_moments_eval

    def estimate_diff_var(self, moments_fn):
        # n_samples = n_dofs + 1 >= 7 leads to probability 0.9 that estimate is whithin range of 10% error from true variance
        assert self.n_collected_samples > 4
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn)
        var_vec = np.var( mom_fine - mom_coarse, axis=0, ddof=1)
        return var_vec, len(mom_fine)


    def estimate_diff_mean(self, moments_fn):
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn)
        mean_vec = np.mean(mom_fine - mom_coarse, axis=0)
        return mean_vec

    def sample_range(self):
        fine_sample = self.sample_values[:, 0]
        q1, q3 = np.percentile(fine_sample, [25, 75])
        iqr = 3*(q3 - q1)
        min_sample = np.min(fine_sample)
        l = min( min_sample , q1-iqr )
        if min_sample > 0.0:    # guess that we have positive distribution
            l = max(0.0, l)
        r = max( np.max(fine_sample), q3+iqr)

        return l,r




class ExpWrongResult(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)
        self.message = "Wrong simulation result"
