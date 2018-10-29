import os
import os.path
import json
import shutil
import numpy as np
from mlmc.sample import Sample
import hdf

class Logger:
    """
    Logging running simulations and finished simulations
    """
    def __init__(self, level_idx, output_dir=None, keep_collected=False):
        """
        Level logger
        :param level_idx: int, Level id
        :param output_dir: string, Output dir for log files
        :param keep_collected: bool, if True then directories of completed simulations are not removed
        """
        # Work dir for scripts and PBS files.
        self.output_dir = output_dir
        self.level_idx = str(level_idx)
        self.keep_collected = keep_collected

        # Number of operation for fine simulations
        self.n_ops_estimate = None
        self.running_header_set = False

        self.collected_log_content = []
        self.running_log_content = []

        # File objects
        self.collected_log_ = None
        self.running_log_ = None

        self._get_hdf()
        self._level_group_path = self._hdf.create_level_group(self.level_idx)

        if output_dir is not None:
            # Get log files
            self.log_running_file = os.path.join(self.output_dir, "running_log_{:s}.json".format(self.level_idx))
            self.log_collected_file = os.path.join(self.output_dir, "collected_log_{:s}.json".format(self.level_idx))
        # Files doesn't exist
        else:
            self.log_running_file = ''
            self.log_collected_file = ''

    def _get_hdf(self):
        """
        Create object to manage HDF5 file
        :return: object
        """
        hdf_file = os.path.join(self.output_dir, "mlmc.hdf5")
        self._hdf = hdf.HDF5(hdf_file, self.output_dir)

    def _running_log(self):
        """
        Running log file object
        :return: File object
        """
        if self.running_log_ is None:
            self._open()
        return self.running_log_

    def _collected_log(self):
        """
        Collected log file object
        :return: File object
        """
        if self.collected_log_ is None:
            self._open()
        return self.collected_log_

    def reload_logs(self, log_collected_file=None):
        """
        Read collected and running simulations data from log file
        :param log_collected_file: Collected file abs path
        :return: None
        """
        self._hdf.read_level(self._level_group_path)

        self._close()
        if log_collected_file is None:
            log_collected_file = self.log_collected_file
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
                                self.collected_log_content.append(sim)
                        except:
                            continue

            # The error was detected by reading log, save correct log again
            if len(lines) != len(self.collected_log_content):
                self.rewrite_collected_log(self.collected_log_content)
        except FileNotFoundError:
            self.collected_log_content = []
        try:
            with open(self.log_running_file, 'r') as reader:
                self.running_log_content = [json.loads(line) for line in reader.readlines()]
        except FileNotFoundError:
            self.running_log_content = []

    def _open(self, flag='a'):
        self.running_log_ = open(self.log_running_file, flag)
        self.collected_log_ = open(self.log_collected_file, flag)

    def _close(self):
        if self.running_log_ is not None:
            self.running_log_.close()
        if self.collected_log_ is not None:
            self.collected_log_.close()

    def log_simulations(self, simulations, collected=False):
        """
        Log simulations, append to collected or running simulations log.
        :param simulations: array of simulations, format according to mc_levels.collect_samples
        :param collected: bool, if true then save collected simulations
        :return: None
        """
        if self.output_dir is None or not simulations:
            return
        if collected:
            log_file = self._collected_log()
            self._hdf.save_collected(self._level_group_path, simulations)
            if not self.keep_collected:
                self._rm_samples(simulations)
        else:
            log_file = self._running_log()
            # n_ops_estimate is already in log file
            if self.n_ops_estimate > 0 and not self.running_header_set:
                self._hdf.set_n_ops_estimate(self._level_group_path, self.n_ops_estimate)
                log_file.write(json.dumps([self.n_ops_estimate]))
                log_file.write("\n")
                self.running_header_set = True

            self._hdf.save_scheduled(self._level_group_path, simulations)

        for sim in simulations:
            log_file.write(json.dumps(sim, cls=ComplexEncoder))
            log_file.write("\n")
        log_file.flush()

    def rewrite_collected_log(self, simulations):
        """
        Create new collected log
        :param simulations: list of simulations
        :return: None
        """
        self.collected_log_ = open(self.log_collected_file, "w")
        self.log_simulations(simulations, True)

    def _rm_samples(self, simulations):
        """
        Remove collected samples dirs
        :param simulations: list of simulations
        :return: None
        """
        for sim in simulations:
            _, _, fine, coarse, _, _ = sim
            if coarse is not None and os.path.isdir(coarse[1]):
                shutil.rmtree(coarse[1], ignore_errors=True)
            if os.path.isdir(fine[1]):
                shutil.rmtree(fine[1], ignore_errors=True)


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Sample):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)





