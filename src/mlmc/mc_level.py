import numpy as np
import numpy.ma as ma
from mlmc.sample import Sample
import os
import shutil
import time as t


class Level:
    """
    Call Simulation methods
    There are information about random variable - average, dispersion, number of simulation, ...
    TODO:
    workflow:
    - queue simulations ( need simulation object for prepare input, need pbs as executor
    - check for finished simulations (need pbs
    - estimates for collected samples ... this can be in separate class as it is independent of simulation
    .. have to reconsider in context of Analysis
    """

    def __init__(self, sim_factory, previous_level, precision, level_idx, hdf_level_group, regen_failed=False,
                 keep_collected=False):
        """
        :param sim_factory: Method that create instance of particular simulation class
        :param previous_level: Previous level object
        :param precision: Current level number / total number of all levels
        :param level_idx: Level identifier
        :param hdf_level_group: hdf.LevelGroup instance, wrapper object for HDF5 group
        :param regen_failed: bool, if True then regenerate failed simulations
        :param keep_collected: bool, if True keep sample dirs otherwise remove them
        """
        # TODO: coarse_simulation can be different to previous_level_sim if they have same mean value
        # Method for creating simulations
        self._sim_factory = sim_factory
        # Level fine simulation precision
        self._precision = precision
        # LevelGroup instance - storage for level data
        self._hdf_level_group = hdf_level_group
        # Directory with level sample's jobs
        self._jobs_dir = self._hdf_level_group.job_dir
        # Level identifier
        self._level_idx = level_idx
        # Keep or remove sample directories, bool value
        self._keep_collected = keep_collected

        # Indicator of first level
        self.is_zero_level = (int(level_idx) == 0)
        # Previous level instance
        self._previous_level = previous_level
        # Fine simulation instance
        self._fine_simulation = None
        # Coarse simulation instance
        self._coarse_simulation = None
        # Estimate of operations number
        self._n_ops_estimate = None
        # Running unfinished simulations that were generated in last whole mlmc run
        self.run_failed = False
        # Target number of samples for the level
        self.target_n_samples = None
        # Last moments function
        self._last_moments_fn = None
        # Moments from coarse and fine samples
        self.last_moments_eval = None
        # Moments outliers mask
        self.mask = None
        # Currently running simulations
        self.scheduled_samples = {}
        # Collected simulations, all results of simulations. Including Nans and None ...
        self.collected_samples = []
        # Failed samples, result is np.Inf
        self.failed_samples = set()
        # Target number of samples.
        self.target_n_samples = 1
        # Number of level samples
        self._n_total_samples = None
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
        self.fine_times = []
        self.coarse_times = []
        # Load simulations from log
        self.load_samples(regen_failed)

    def reset(self):
        """
        Reset level variables for further use
        :return: None
        """
        self.scheduled_samples = {}
        self.collected_samples = []
        self.target_n_samples = 3
        self._sample_values = np.empty((self.target_n_samples, 2))
        self._n_valid_samples = 0
        self.sample_indices = None
        self.nan_samples = []
        self._last_moments_fn = None
        self.fine_times = []
        self.coarse_times = []

    @property
    def finished_samples(self):
        """
        Get collected and failed samples ids
        :return: NumPy array
        """
        return len(self.collected_samples) + len(self.failed_samples)

    @property
    def fine_simulation(self):
        """
        Fine simulation object
        :return: Simulation object
        """
        if self._fine_simulation is None:
            self._fine_simulation = self._sim_factory(self._precision, int(self._level_idx))
        return self._fine_simulation

    @property
    def coarse_simulation(self):
        """
        Coarse simulation object
        :return: Simulations object
        """
        if self._previous_level is not None and self._coarse_simulation is None:
            self._coarse_simulation = self._previous_level.fine_simulation
        return self._coarse_simulation

    def load_samples(self, regen_failed):
        """
        Load collected and scheduled samples from log
        :return: None
        """
        collected_samples = {}
        # Get logs content
        log_scheduled_samples, log_collected_samples = self._reload_samples()

        for fine_sample, coarse_sample in log_collected_samples:
            # Append collected samples
            collected_samples[fine_sample.sample_id] = (fine_sample, coarse_sample)
            # Add sample results
            self._add_sample(fine_sample.sample_id, (fine_sample.result, coarse_sample.result))
            # Get time from samples
            self.fine_times.append(fine_sample.time)
            self.coarse_times.append(coarse_sample.time)

        # Recover scheduled
        for fine_sample, coarse_sample in log_scheduled_samples:
            # Regenerate failed samples
            if regen_failed:
                # Sample that is not in collected to scheduled
                if fine_sample.sample_id not in collected_samples:
                    self.scheduled_samples[fine_sample.sample_id] = (fine_sample, coarse_sample)
            # Not collected and not failed sample to scheduled
            elif fine_sample.sample_id not in collected_samples and fine_sample.sample_id not in self._failed_sample_ids:
                    self.scheduled_samples[fine_sample.sample_id] = (fine_sample, coarse_sample)

        self.collected_samples = list(collected_samples.values())

        # @TODO check
        # Get n_ops_estimate
        if self.n_ops_estimate is None:
            self.set_coarse_sim()

        # Total number of samples
        self._n_total_samples = len(self.scheduled_samples) + len(self.collected_samples) + len(self._failed_sample_ids)

        # Regenerate failed samples and collect samples
        if regen_failed and len(self._failed_sample_ids) > 0:
            self._run_failed_samples()

        # Collect scheduled samples
        if len(self.scheduled_samples) > 0:
            self.collect_samples()

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

    def _add_sample(self, idx, sample_pair):
        """
        Add samples pair to rest of samples
        :param id: sample id
        :param sample_pair: Fine and coarse result
        :return: None
        """
        fine, coarse = sample_pair

        # Samples are not finite
        if not np.isfinite(fine) or not np.isfinite(coarse):
            self.nan_samples.append(idx)
            return
        # Enlarge matrix of samples
        if self._n_valid_samples == self._sample_values.shape[0]:
            self.enlarge_samples(2 * self._n_valid_samples)

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
    def n_samples(self):
        """
        Number of level sample
        :return: int, number of samples
        """
        # Number of samples used for estimates.
        if self.sample_indices is None:
            return self._n_valid_samples
        else:
            return len(self.sample_indices)

    def _get_sample_tag(self, char):
        """
        Create sample tag
        :param char: 'C' or 'F' depending on the type of simulation
        :return: str
        """
        return "L{:02d}_{}_S{:07d}".format(int(self._level_idx), char, self._n_total_samples)

    @property
    def n_ops_estimate(self):
        """
        :return: number of fine sim operations
        """
        if self._n_ops_estimate is None:
            self._n_ops_estimate = self._hdf_level_group.n_ops_estimate
        return self._n_ops_estimate

    @n_ops_estimate.setter
    def n_ops_estimate(self, n_ops):
        """
        Set n ops estimate
        :param n_ops: number of operations
        :return: None
        """
        self._n_ops_estimate = n_ops
        self._hdf_level_group.n_ops_estimate = n_ops

    def set_coarse_sim(self):
        """
        Set coarse sim to fine simulation
        :return: None
        """
        if not self.fine_simulation.coarse_sim_set:
            self.fine_simulation.set_coarse_sim(self.coarse_simulation)
            self.n_ops_estimate = self.fine_simulation.n_ops_estimate()

    def _make_sample_pair(self):
        """
        Generate new random samples for fine and coarse simulation objects
        :return: list
        """
        start_time = t.time()
        self.set_coarse_sim()
        # All levels have fine simulation
        idx = self._n_total_samples
        self.fine_simulation.generate_random_sample()
        tag = self._get_sample_tag('F')
        fine_sample = self.fine_simulation.simulation_sample(tag, idx, start_time)

        start_time = t.time()
        if self.coarse_simulation is not None:
            tag = self._get_sample_tag('C')
            coarse_sample = self.coarse_simulation.simulation_sample(tag, idx, start_time)
        else:
            # Zero level have no coarse simulation.
            coarse_sample = Sample(sample_id=idx)
            coarse_sample.result = 0.0

        self._n_total_samples += 1

        return [(idx, (fine_sample, coarse_sample))]

    def _run_failed_samples(self):
        """
        Run already generated simulations again
        :return: None
        """
        for sample_id, (fine_sim, coarse_sim) in self.scheduled_samples.items():
            if sample_id in self.failed_samples:
                self.set_coarse_sim()
                # All levels have fine simulation
                self.fine_simulation.generate_random_sample()
                if self.coarse_simulation is not None:
                    coarse_sample = self.coarse_simulation.simulation_sample(sample_id=coarse_sim.sample_id)
                else:
                    coarse_sample = coarse_sim
                    coarse_sample.result = 0.0
                fine_sample = self.fine_simulation.simulation_sample(sample_id=fine_sim.sample_id)

            self.scheduled_samples[sample_id] = (fine_sample, coarse_sample)

        # Empty failed samples set
        self.failed_samples = set()
        self.run_failed = False

        self.collect_samples()

    def fill_samples(self):
        """
        Generate samples up to target number set through 'set_target_n_samples'.
        Simulations are planed for execution, but sample values are collected in
        :return: None
        """
        if self.run_failed:
            self._run_failed_samples()

        new_scheduled_simulations = {}
        if self.target_n_samples > self._n_total_samples:
            self.enlarge_samples(self.target_n_samples)
            # Create pair of fine and coarse simulations and add them to list of all running simulations
            while self._n_total_samples < self.target_n_samples:
                new_scheduled_simulations.update(self._make_sample_pair())

            self.scheduled_samples.update(new_scheduled_simulations)
            self._log_scheduled(new_scheduled_simulations)

    def collect_samples(self):
        """
        Extract values from non queued samples. Save no data to HDF datasets.
        :return: Number of samples to finish yet.
        """
        # Samples that are not running and aren't finished
        not_queued_sample_ids = self._not_queued_sample_ids()
        orig_n_finised = len(self.collected_samples)

        for sample_id in not_queued_sample_ids:
            fine_sample, coarse_sample = self.scheduled_samples[sample_id]

            # Sample() instance
            fine_sample = self.fine_simulation.extract_result(fine_sample)
            fine_done = fine_sample.result is not None

            # For zero level don't create Sample() instance via simulations,
            # however coarse sample is created for easier processing
            if not self.is_zero_level:
                coarse_sample = self.coarse_simulation.extract_result(coarse_sample)
            coarse_done = coarse_sample.result is not None

            if fine_done and coarse_done:
                # 'Remove' from scheduled
                self.scheduled_samples[sample_id] = False
                # Failed sample
                if fine_sample.result is np.inf or coarse_sample.result is np.inf:
                    coarse_sample.result = fine_sample.result = np.inf
                    self.failed_samples.add(sample_id)
                    continue

                self.fine_times.append(fine_sample.time)
                self.coarse_times.append(coarse_sample.time)

                # collect values
                self.collected_samples.append((fine_sample, coarse_sample))
                self._add_sample(sample_id, (fine_sample.result, coarse_sample.result))

        # Still scheduled samples
        self.scheduled_samples = {sample_id: values for sample_id, values in self.scheduled_samples.items()
                                  if values is not False}

        # Log new collected samples
        self._log_collected(self.collected_samples[orig_n_finised:])
        # Log failed samples
        self._log_failed(self.failed_samples)

        return len(self.scheduled_samples)

    def _log_failed(self, samples):
        """
        Log failed samples
        :param samples: set; sample ids
        :return: None
        """
        self._hdf_level_group.save_failed(samples)

    def _log_collected(self, samples):
        """
        Log collected samples, eventually remove samples
        :param samples: list [(fine sample, coarse sample), ...], both are Sample() instances
        :return: None
        """
        if not samples:
            return
        self._hdf_level_group.append_collected(samples)

        # If not keep collected remove samples
        if not self._keep_collected:
            self._rm_samples(samples)

    def _reload_samples(self):
        """
        Read collected and scheduled samples from hdf file
        :return: tuple of generators - item is Sample()
        """
        scheduled_samples = self._hdf_level_group.scheduled()
        collected_samples = self._hdf_level_group.collected()
        return scheduled_samples, collected_samples

    @property
    def _failed_sample_ids(self):
        """
        Get failed samples from hdf file
        :return: set
        """
        return self._hdf_level_group.get_failed_ids()

    def _log_scheduled(self, samples):
        """
        Log scheduled samples
        :param samples: dict {sample_id : (fine sample, coarse sample), ...}, samples are Sample() instances 
        :return: None
        """
        if not samples:
            return
        # n_ops_estimate is already in log file
        if self.n_ops_estimate > 0:
            self._hdf_level_group.n_ops_estimate = self.n_ops_estimate

        self._hdf_level_group.append_scheduled(samples)

    def _not_queued_sample_ids(self):
        """
        Get level queued jobs and sample ids from these jobs
        :return: NumPy array
        """
        # Level jobs ids (=names)
        job_ids = self._hdf_level_group.level_jobs()
        # Ids from jobs that are not queued
        not_queued_jobs = [job_id for job_id in job_ids
                           if not os.path.exists(os.path.join(self._jobs_dir, *[job_id, 'QUEUED']))]

        # Set of sample ids that are not in queued
        not_queued_sample_ids = self._hdf_level_group.job_samples(not_queued_jobs)
        finished_sample_ids = self._hdf_level_group.get_finished_ids()

        # Return sample ids of not queued and not finished samples
        if len(not_queued_sample_ids) > 0 and len(finished_sample_ids) > 0:
                return np.setdiff1d(not_queued_sample_ids, finished_sample_ids)
        return not_queued_sample_ids

    def _rm_samples(self, samples):
        """
        Remove sample dirs
        :param samples: List of sample tuples (fine Sample(), coarse Sample())
        :return: None
        """
        for fine_sample, coarse_sample in samples:
            if os.path.isdir(coarse_sample.directory):
                shutil.rmtree(coarse_sample.directory, ignore_errors=True)
            if os.path.isdir(fine_sample.directory):
                shutil.rmtree(fine_sample.directory, ignore_errors=True)

    def subsample(self, size):
        """
        sub-selection from simulations results
        :param size: number of subsamples
        :return: None
        """
        if size is None:
            self.sample_indices = None
        else:
            assert 0 < size < self._n_valid_samples, "0 < {} < {}".format(size, self._n_valid_samples)
            self.sample_indices = np.random.choice(np.arange(self._n_valid_samples, dtype=int), size=size)

    def evaluate_moments(self, moments_fn, force=False):
        """
        Evaluating moments from moments function
        :param moments_fn: Moment evaluation functions
        :param force: Reevaluate moments
        :return: tuple
        """
        # Current moment functions are different from last moment functions
        if force or moments_fn != self._last_moments_fn:
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

            self._remove_outliers_moments()

        if self.sample_indices is None:
            return self.last_moments_eval
        else:
            m_fine, m_coarse = self.last_moments_eval
            return m_fine[self.sample_indices, :], m_coarse[self.sample_indices, :]

    def _remove_outliers_moments(self):
        """
        Remove moments from outliers from fine and course moments
        :return: None
        """
        # Fine and coarse moments mask
        mask_fine = ma.masked_invalid(self.last_moments_eval[0]).mask
        mask_coarse = ma.masked_invalid(self.last_moments_eval[1]).mask

        # Common mask for coarse and fine
        mask_fine_coarse = np.logical_or(mask_fine, mask_coarse)[:, -1]

        self.mask = mask_fine_coarse

        # New moments without outliers
        self.last_moments_eval = self.last_moments_eval[0][~mask_fine_coarse], self.last_moments_eval[1][
            ~mask_fine_coarse]

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
        assert self.n_samples > 1
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn, force=True)
        var_vec = np.var(mom_fine - mom_coarse, axis=0, ddof=1)
        return var_vec, len(mom_fine)

    def estimate_diff_mean(self, moments_fn):
        """
        Estimate moments mean
        :param moments_fn: Function for calculating moments
        :return: np.array, moments mean vector
        """
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn, force=True)
        mean_vec = np.mean(mom_fine - mom_coarse, axis=0)
        return mean_vec

    def sample_range(self):
        """
        Determine limits for outliers
        :return: tuple
        """
        fine_sample = self.sample_values[:, 0]
        quantile_1, quantile_3 = np.percentile(fine_sample, [25, 75])

        iqr = quantile_3 - quantile_1
        min_sample = np.min(fine_sample)

        left = max(min_sample, quantile_1 - 1.5 * iqr)
        if min_sample > 0.0:  # guess that we have positive distribution
            left = min_sample
        right = min(np.max(fine_sample), quantile_3 + 1.5 * iqr)

        return left, right

    def get_n_finished(self):
        """
        Number of finished simulations
        :return: int
        """
        self.collect_samples()
        return len(self.collected_samples) + len(self.failed_samples)

    def sample_time(self):
        """
        Get average sample time
        :return: float
        """
        times = np.array(self.fine_times) + np.array(self.coarse_times)
        # Remove error times - temporary solution
        times = times[(times < 1e5)]

        return np.mean(times)
