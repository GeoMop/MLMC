import os
import os.path
import json
import shutil
import numpy as np
import mlmc.sample


class Logger:
    """
    Logging level samples
    """
    # @TODO remove all json logs dependencies
    def __init__(self, level_idx, hdf_level_group, output_dir=None, keep_collected=False):
        """
        Level logger
        :param level_idx: int, Level id
        :param hdf_level_group: HDF5 file group 
        :param keep_collected: bool, if True then directories of completed simulations are not removed
        """
        self.level_idx = str(level_idx)
        self.keep_collected = keep_collected
        self.output_dir = output_dir

        # Number of operation for fine simulations
        self._n_ops_estimate = None
        self._jobs_dir = None
        self._hdf_level_group = hdf_level_group

    @property
    def n_ops_estimate(self):
        if self._n_ops_estimate is None:
            self._n_ops_estimate = self._hdf_level_group.n_ops_estimate
        return self._n_ops_estimate

    @n_ops_estimate.setter
    def n_ops_estimate(self, n_ops):
        self._n_ops_estimate = n_ops

    def not_queued_jobs_sample_ids(self):
        """
        Get level queued jobs and sample ids from these jobs
        :return: set
        """
        if self._jobs_dir is None:
            self._jobs_dir = self._hdf_level_group.job_dir

        # Level jobs ids (=names)
        job_ids = self._hdf_level_group.level_jobs()
        # Ids from jobs that are not queued
        not_queued_jobs = [job_id for job_id in job_ids
                           if not os.path.exists(os.path.join(self._jobs_dir, *[job_id, 'QUEUED']))]

        # Set of sample ids that are not in queued
        return self._hdf_level_group.job_samples(not_queued_jobs)

    def reload_samples(self):
        """
        Read collected and scheduled samples from hdf file
        :return: tuple
        """
        scheduled_samples = self._hdf_level_group.scheduled()
        collected_samples = self._hdf_level_group.collected()

        return scheduled_samples, collected_samples

    def log_failed(self, samples):
        """
        Log failed samples
        :param samples: set; sample ids
        :return: None
        """
        self._hdf_level_group.save_failed(samples)

    def log_collected(self, samples):
        """
        Log collected samples, eventually remove samples
        :param samples: list [(fine sample, coarse sample), ...], both are Sample() instances
        :return: None
        """
        if not samples:
            return
        self._hdf_level_group.append_collected(samples)

        # If not keep collected remove samples
        if not self.keep_collected:
            self._rm_samples(samples)

    def log_scheduled(self, samples):
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

    def failed_samples(self):
        """
        Get failed samples from hdf file
        :return: set
        """
        return self._hdf_level_group.failed_samples()

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

    def json_to_hdf(self):
        """
        Auxiliary method for conversion from json log to hdf log
        :return: None
        """
        collected_log_content = []
        running_log_content = []

        # File objects
        self.collected_log_ = None
        self.running_log_ = None

        if self.output_dir is not None:
            # Get log files
            log_running_file = os.path.join(self.output_dir, "running_log_{:s}.json".format(self.level_idx))
            log_collected_file = os.path.join(self.output_dir, "collected_log_{:s}.json".format(self.level_idx))
            # Files doesn't exist
        else:
            self.log_running_file = ''
            self.log_collected_file = ''

        try:
            with open(log_collected_file, 'r') as reader:
                lines = reader.readlines()
                # File is not empty
                if len(lines) > 0:
                    for line in lines:
                        try:
                            sim = json.loads(line)
                            # Simulation list should contains 6 items if not add default time at the end
                            if len(sim) == 5:
                                sim.append([[np.inf, np.inf], [np.inf, np.inf]])
                            if len(sim) == 6:
                                collected_log_content.append(sim)
                        except:
                            continue

        except FileNotFoundError:
            collected_log_content = []
        try:
            with open(log_running_file, 'r') as reader:
                running_log_content = [json.loads(line) for line in reader.readlines()]
        except FileNotFoundError:
            running_log_content = []

        if len(collected_log_content) > 0 and len(running_log_content) > 0:
            self._hdf_samples(running_log_content, collected_log_content)

    def _hdf_samples(self, running_log_content, collected_log_content):
        """
        Convert to HDF5 format
        :return: None
        """
        n_ops_estimate = np.squeeze(running_log_content[0])

        samples = []
        scheduled_samples = {}
        failed = set()
        for sim in collected_log_content:
            _, i, fine, coarse, value, times = sim
            fine_sample = mlmc.sample.Sample(sample_id=i, directory=fine[1])
            fine_sample.result = value[0]
            fine_sample.time = times[0]

            if coarse is None:
                coarse_sample = mlmc.sample.Sample(sample_id=i)
                coarse_sample.result = 0.0
            else:
                coarse_sample = mlmc.sample.Sample(sample_id=i, directory=coarse[1])
                coarse_sample.result = value[1]
                coarse_sample.time = times[1]

            if fine_sample.result == np.Inf or coarse_sample.result == np.Inf:
                failed.add(fine_sample.sample_id)
                continue

            samples.append((fine_sample, coarse_sample))
            scheduled_samples[fine_sample.sample_id] = (fine_sample, coarse_sample)

        self._hdf_level_group.append_scheduled(scheduled_samples)
        self._hdf_level_group.append_collected(samples)
        self._hdf_level_group.save_failed(failed)
        self._hdf_level_group.n_ops_estimate = n_ops_estimate

    # def change_status(self, sample_id, status):
    #     self._hdf.change_scheduled_sample_status(self.level_idx, sample_id, status)
