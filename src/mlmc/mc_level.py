import os
import shutil
import copy
import time as t
import numpy as np
from mlmc.sample import Sample


class Level:
    """
    Call Simulation methods
    There are information about random variable - average, dispersion, number of simulation, ...
    TODO:
    - have HDF level either permanently (do not copy values), or use just for load and save
    - fix consistency for: n_ops, n_ops_estimate, _n_ops_estimate
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
        # Number of collected samples in _sample_values.
        self._n_collected_samples = 0
        # Possible subsample indices.
        self.sample_indices = None
        # Array of indices of nan samples (in fine or coarse sim)
        self.nan_samples = []
        # Cache evaluated moments.
        self._last_moments_fn = None
        self.fine_times = []
        self.coarse_times = []
        self._select_condition = None
        # Load simulations from log
        self.load_samples(regen_failed)

    def reset(self):
        """
        Reset level variables for further use
        :return: None
        """
        self.scheduled_samples = {}
        self.target_n_samples = 3
        self._sample_values = np.empty((self.target_n_samples, 2))
        self._n_collected_samples = 0
        self.sample_indices = None
        self.nan_samples = []
        self._last_moments_fn = None
        self.fine_times = []
        self.coarse_times = []
        self._select_condition = None
        self.collected_samples = []

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
    def step(self):
        # TODO: modify mechanism to combine precision and step_range in simulation factory
        # in order to get true step here.
        return self._precision

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
            fine_sample.select(self._select_condition)
            coarse_sample.select(self._select_condition)
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
        self.n_ops_estimate = self._hdf_level_group.n_ops_estimate

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

    # def force_extract(self):
    #     """
    #     Call extract results for already collected samples if their directory exists.
    #     :return:
    #     """
    #     for sample_id, _ in self.scheduled_samples.items():
    #         # Run simulations again
    #         if sample_id in self._failed_sample_ids:
    #             self.scheduled_samples.update(self._make_sample_pair(sample_id))
    #
    #     sample.Sample(directory=sample_dir, sample_id=sample_id,
    #                   job_id=package_dir, prepare_time=(t.time() - start_time))
    #
    #     # Empty failed samples set
    #     self.failed_samples = set()
    #     self.run_failed = False
    #
    #     self.collect_samples()
    #
    #     samples_dir = os.path.join(self.fine_simulation.work_dir, "samples")
    #     for sdir in os.listdir(samples_dir):
    #         sid = int(sdir[-7:])
    #         sdir_path = os.path.join(samples_dir, sdir)
    #         fine_sample = Sample(directory=sdir_path, sample_id=sid,
    #                       job_id=package_dir, prepare_time=0.0)
    #
    #         # todo: see _make_sample_pair
    #         self._log_scheduled(new_scheduled_simulations)
    #
    #     new_collected = []
    #     for fine, coarse in self.collected_samples:
    #         try:
    #             fine_sample = self.fine_simulation.extract_result(fine)
    #             fine_done = not np.any(np.isnan(fine_sample.result))
    #             # For zero level don't create Sample() instance via simulations,
    #             # however coarse sample is created for easier processing
    #             if not self.is_zero_level:
    #                 coarse_sample = self.coarse_simulation.extract_result(coarse)
    #                 coarse_done = np.all(np.isnan(coarse_sample.result))
    #             else:
    #                 coarse_done = True
    #         except Exception:
    #             fine_done = False
    #
    #         # Enlarge coarse sample result to length of fine sample result
    #         if self.is_zero_level:
    #             coarse_sample.result_data = copy.deepcopy(fine_sample.result_data)
    #             coarse_sample.result = np.full((len(fine_sample.result),), 0.0)
    #
    #         # Failed sample
    #         if np.any(np.isinf(fine_sample.result)) or np.any(np.isinf(coarse_sample.result)):
    #             coarse_sample.result = fine_sample.result = np.full((len(fine_sample.result),), np.inf)
    #             self.failed_samples.add(fine_sample.sample_id)
    #             continue
    #
    #         if not fine_done or not coarse_done:
    #             (fine_sample, coarse_sample) = (fine, coarse)
    #         new_collected.append((fine_sample, coarse_sample))
    #         fine_sample.select(self._select_condition)
    #         coarse_sample.select(self._select_condition)
    #         self.collected_samples = new_collected
    #    self.select(self._select_condition)

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
        Get ALL colected level samples.
        Without filtering Nans in moments. Without subsampling.
        :return: array, shape (n_samples, 2). First column fine, second coarse.
        """
        if self.sample_indices is not None:
            bool_mask = self.sample_indices
            # Sample values are sometimes larger than sample indices (caused by enlarge_samples() method)
            if len(self.sample_indices) < len(self._sample_values):
                bool_mask = np.full(len(self._sample_values), True)
                bool_mask[:len(self.sample_indices)] = self.sample_indices

            return self._sample_values[bool_mask]
        return self._sample_values[:self._n_collected_samples]

    def _add_sample(self, idx, sample_pair):
        """
        Add samples pair to rest of samples
        :param id: sample id
        :param sample_pair: Fine and coarse result
        :return: None
        """
        fine, coarse = sample_pair

        if len(fine) == 0 or len(coarse) == 0:
            return

        if len(self._sample_values.shape) < 3:
            self._sample_values = np.empty((self.target_n_samples, 2, len(fine)))

        # Samples are not finite
        if not np.all(np.isfinite(fine)) or not np.all(np.isfinite(coarse)):
            self.nan_samples.append(idx)
            return

        # Enlarge matrix of samples
        if self._n_collected_samples == self._sample_values.shape[0]:
            self.enlarge_samples(2 * self._n_collected_samples)

        # Add fine and coarse sample
        self._sample_values[self._n_collected_samples, :] = (fine, coarse)
        self._n_collected_samples += 1

    def enlarge_samples(self, size):
        """
        Enlarge matrix of samples
        :param size: New sample matrix size
        :return: None
        """
        # Enlarge sample matrix
        new_size = list(self._sample_values.shape)
        new_size[0] = size

        new_values = np.empty(new_size)
        new_values[:self._n_collected_samples] = self._sample_values[:self._n_collected_samples]
        self._sample_values = new_values

    @property
    def n_samples(self):
        """
        Number of collected level samples.
        OR number of samples with correct fine and coarse moments
        OR number of subsampled samples.
        :return: int, number of samples
        """
        # Number of samples used for estimates.
        if self.sample_indices is None:
            if self.last_moments_eval is not None:
                return len(self.last_moments_eval[0])
            return self._n_collected_samples
        else:
            return len(self.sample_indices)

    def _get_sample_tag(self, char, sample_id):
        """
        Create sample tag
        :param char: 'C' or 'F' depending on the type of simulation
        :param sample_id: int, identifier of current sample
        :return: str
        """
        return "L{:02d}_{}_S{:07d}".format(int(self._level_idx), char, sample_id)

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
        if n_ops is not None:
            self._hdf_level_group.n_ops_estimate = n_ops

    def set_coarse_sim(self):
        """
        Set coarse sim to fine simulation
        :return: None
        """
        if not self.fine_simulation.coarse_sim_set:
            self.fine_simulation.set_coarse_sim(self.coarse_simulation)
            self.n_ops_estimate = self.fine_simulation.n_ops_estimate()

    def _make_sample_pair(self, sample_pair_id=None):
        """
        Generate new random samples for fine and coarse simulation objects
        :return: list
        """
        start_time = t.time()
        self.set_coarse_sim()
        # All levels have fine simulation
        if sample_pair_id is None:
            sample_pair_id = self._n_total_samples
        self.fine_simulation.generate_random_sample()
        tag = self._get_sample_tag('F', sample_pair_id)
        fine_sample = self.fine_simulation.simulation_sample(tag, sample_pair_id, start_time)

        start_time = t.time()
        if self.coarse_simulation is not None:
            tag = self._get_sample_tag('C', sample_pair_id)
            coarse_sample = self.coarse_simulation.simulation_sample(tag, sample_pair_id, start_time)
        else:
            # Zero level have no coarse simulation
            coarse_sample = Sample(sample_id=sample_pair_id)
            #@TODO: find more elegant workaround for multilevel usage
            coarse_sample.result_data = np.array([0.0], dtype=[("value", 'S10')])

        self._n_total_samples += 1

        return [(sample_pair_id, (fine_sample, coarse_sample))]

    def _run_failed_samples(self):
        """
        Run already generated simulations again
        :return: None
        """
        for sample_id, _ in self.scheduled_samples.items():
            # Run simulations again
            if sample_id in self._failed_sample_ids:
                self.scheduled_samples.update(self._make_sample_pair(sample_id))

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
        orig_n_finished = len(self.collected_samples)

        for sample_id in not_queued_sample_ids:
            fine_sample, coarse_sample = self.scheduled_samples[sample_id]

            # Sample() instance
            fine_sample = self.fine_simulation.extract_result(fine_sample)
            fine_done = not np.any(np.isnan(fine_sample.result))
            # For zero level don't create Sample() instance via simulations,
            # however coarse sample is created for easier processing
            if not self.is_zero_level:
                coarse_sample = self.coarse_simulation.extract_result(coarse_sample)
                coarse_done = np.all(np.isnan(coarse_sample.result))
            else:
                coarse_done = True

            if fine_done and coarse_done:
                # 'Remove' from scheduled
                self.scheduled_samples[sample_id] = False

                # Enlarge coarse sample result to length of fine sample result
                if self.is_zero_level:
                    coarse_sample.result_data = copy.deepcopy(fine_sample.result_data)
                    coarse_sample.result = np.full((len(fine_sample.result),), 0.0)

                # Failed sample
                if np.any(np.isinf(fine_sample.result)) or np.any(np.isinf(coarse_sample.result)):
                    coarse_sample.result = fine_sample.result = np.full((len(fine_sample.result), ), np.inf)
                    self.failed_samples.add(sample_id)
                    continue

                self.fine_times.append(fine_sample.time)
                self.coarse_times.append(coarse_sample.time)

                # collect values
                self.collected_samples.append((fine_sample, coarse_sample))
                fine_sample.select(self._select_condition)
                coarse_sample.select(self._select_condition)
                self._add_sample(sample_id, (fine_sample.result, coarse_sample.result))

        # Still scheduled samples
        self.scheduled_samples = {sample_id: values for sample_id, values in self.scheduled_samples.items()
                                  if values is not False}

        # Log new collected samples
        self._log_collected(self.collected_samples[orig_n_finished:])

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

    def subsample(self, size=None, sample_indices=None):
        """
        Sub-selection from samples with correct moments (dependes on last call to eval_moments).
        :param size: number of subsamples
        :param sample_indices: Sample indices, boolean mask (bool or int type)
        :return: None
        """
        self.sample_indices = sample_indices

        if size is not None and sample_indices is None:
            assert self.last_moments_eval is not None
            n_moment_samples = len(self.last_moments_eval[0])
            assert 0 < size, "0 < {}".format(size)
            self.sample_indices = np.random.choice(np.arange(n_moment_samples, dtype=int), size=size)
            self.sample_indices.sort()  # Better for caches.

    def evaluate_moments(self, moments_fn, force=False):
        """
        Evaluate level difference for all samples and given moments.
        :param moments_fn: Moment evaluation object.
        :param force: Reevaluate moments
        :return: (fine, coarse) both of shape (n_samples, n_moments)
        """
        # Current moment functions are different from last moment functions
        same_moments = moments_fn == self._last_moments_fn
        same_shapes = self.last_moments_eval is not None
        if force or not same_moments or not same_shapes:
            samples = self.sample_values
            moments_fine = moments_fn(samples[:, 0])

            # For first level moments from coarse samples are zeroes
            if self.is_zero_level:
                moments_coarse = np.zeros(moments_fine.shape)
            else:
                moments_coarse = moments_fn(samples[:, 1])

            # Set last moments function
            self._last_moments_fn = moments_fn
            # Moments from fine and coarse samples
            self.last_moments_eval = moments_fine, moments_coarse

            self._remove_outliers_moments()
            if self.sample_indices is not None:
                self.subsample(len(self.sample_indices))

        if self.sample_indices is None:
            return self.last_moments_eval
        else:
            m_fine, m_coarse = self.last_moments_eval
            return m_fine[self.sample_indices, :], m_coarse[self.sample_indices, :]

    def _remove_outliers_moments(self, ):
        """
        Remove moments from outliers from fine and course moments
        :return: None
        """
        # Fine and coarse moments mask
        ok_fine = np.all(np.isfinite(self.last_moments_eval[0]), axis=len(self.last_moments_eval[0][0].shape))
        ok_coarse = np.all(np.isfinite(self.last_moments_eval[1]), axis=len(self.last_moments_eval[1][0].shape))

        # Common mask for coarse and fine
        ok_fine_coarse = np.logical_and(ok_fine, ok_coarse)
        bool_mask = np.ones((ok_fine_coarse[0].shape))
        for bool_array in ok_fine_coarse:
            bool_mask = np.logical_and(bool_mask, bool_array)

        ok_fine_coarse = bool_mask

        # New moments without outliers
        self.last_moments_eval = self.last_moments_eval[0][:, ok_fine_coarse],\
                                 self.last_moments_eval[1][:, ok_fine_coarse]


    def estimate_level_var(self, moments_fn):
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn)
        var_fine = np.var(mom_fine, axis=0, ddof=1)
        var_coarse = np.var(mom_coarse, axis=0, ddof=1)
        return var_coarse, var_fine

    def estimate_diff_var(self, moments_fn):
        """
        Estimate moments variance
        :param moments_fn: Moments evaluation function
        :return: tuple (variance vector, length of moments)
        """
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn)

        assert len(mom_fine) == len(mom_coarse)
        assert len(mom_fine) >= 2
        var_vec = np.var(mom_fine - mom_coarse, axis=0, ddof=1)
        ns = self.n_samples
        assert ns == len(mom_fine)  # This was previous unconsistent implementation.
        return var_vec, ns

    def estimate_diff_mean(self, moments_fn):
        """
        Estimate moments mean
        :param moments_fn: Function for calculating moments
        :return: np.array, moments mean vector
        """
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn)

        assert len(mom_fine) == len(mom_coarse)
        assert len(mom_fine) >= 1

        mean_vec = np.mean(mom_fine - mom_coarse, axis=0)
        return mean_vec

    def estimate_covariance(self, moments_fn, stable=False):
        """
        Estimate covariance matrix (non central).
        :param moments_fn: Moment functions object.
        :param stable: Use alternative formula with better numerical stability.
        :return: cov covariance matrix  with shape (n_moments, n_moments)
        """
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn)
        assert len(mom_fine) == len(mom_coarse)
        assert len(mom_fine) >= 2
        assert self.n_samples == len(mom_fine)

        if stable:
            # Stable formula - however seems that we have no problem with numerical stability
            mom_diff = mom_fine - mom_coarse
            mom_sum = mom_fine + mom_coarse
            cov = 0.5 * (np.matmul(mom_diff.T, mom_sum) + np.matmul(mom_sum.T, mom_diff)) / self.n_samples
        else:
            mom_fine = np.squeeze(mom_fine)
            mom_coarse = np.squeeze(mom_coarse)

            # Direct formula
            cov_fine = np.matmul(mom_fine.T,   mom_fine)
            cov_coarse = np.matmul(mom_coarse.T, mom_coarse)
            cov = (cov_fine - cov_coarse) / self.n_samples

        return cov

    def estimate_cov_diag_err(self, moments_fn, ):
        """
        Estimate mean square error (variance) of the estimate of covariance matrix difference at this level.
        Only compute MSE for diagonal elements of the covariance matrix.
        :param moments_fn:
        :return: Vector of MSE for diagonal
        """
        mom_fine, mom_coarse = self.evaluate_moments(moments_fn)
        assert len(mom_fine) == len(mom_coarse)
        assert len(mom_fine) >= 2
        assert self.n_samples == len(mom_fine)

        mse_vec = np.var(mom_fine**2 - mom_coarse**2, axis=0, ddof=1)
        return mse_vec

    def sample_iqr(self):
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

    def sample_range(self):
        fine_sample = self.sample_values[:, 0]
        if len(fine_sample) > 0:
            return np.min(fine_sample), np.max(fine_sample)
        return 0, 0

    def sample_domain(self, quantile=None):
        if quantile is None:
            return self.sample_range()
        fine_sample = self.sample_values[:, 0]
        return np.percentile(fine_sample, [100*quantile, 100*(1-quantile)])

    def get_n_finished(self):
        """
        Number of finished simulations
        :return: int
        """
        self.collect_samples()
        return len(self.collected_samples) + len(self.failed_samples)

    def select(self, condition, selected_param=None):
        """
        Set sample select condition
        :param condition: dict, ({sample result param: (value, comparison)})
        :return: None
        """
        self._select_condition = condition
        selected_samples = []
        for f_sample, c_sample in self.collected_samples:
            f_sample.select(condition, selected_param)
            c_sample.select(condition, selected_param)
            selected_samples.append((f_sample, c_sample))

        self._reload_sample_values(selected_samples)
        return selected_samples

    def _reload_sample_values(self, samples):
        """
        Get selected samples result values
        :param samples: list of tuples [(Sample(), Sample()), ...]
        :return: None
        """
        self._sample_values = np.empty((len(samples), 2, len(samples[0][0].result)))
        for index, (fine, coarse) in enumerate(samples):
            self._sample_values[index, :] = (fine.result, coarse.result)

    def clean_select(self):
        """
        Clean sample's select condition
        :return: None
        """
        for c_sample, f_sample in self.collected_samples:
            c_sample.clean_select()
            f_sample.clean_select()

        self._reload_sample_values(self.collected_samples)
        self._select_condition = None

    def sample_time(self):
        """
        Get average sample time
        :return: float
        """
        times = np.array(self.fine_times) + np.array(self.coarse_times)
        # Remove error times - temporary solution
        times = times[(times < 1e5)]

        return np.mean(times)

    def avg_sample_running_time(self):
        """
        Get average sample simulation running time per samples
        :return: float
        """
        return np.mean([sample.running_time for sample, _ in self.collected_samples])

    def avg_sample_prepare_time(self):
        """
        Get average sample simulation running time per samples
        :return: float
        """
        return np.mean([sample.time - sample.running_time for sample, _ in self.collected_samples])

    def avg_level_running_time(self):
        """
        Get average level (fine + coarse) running time per samples
        :return: float
        """
        return np.mean([fine_sample.running_time + coarse_sample.running_time for fine_sample, coarse_sample in self.collected_samples])

    def avg_level_prepare_time(self):
        """
        Get average level (fine + coarse) prepare time per samples
        :return: float
        """
        return np.mean([fine_sample.time - coarse_sample.running_time for fine_sample, coarse_sample in self.collected_samples])
