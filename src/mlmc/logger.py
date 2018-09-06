import os
import os.path
import json


class Logger:
    def __init__(self, level_idx, output_dir=None, keep_collected=False):
        """
        Level logger
        :param level_idx: int, Level id
        :param output_dir: string, Output dir for log files
        :param clean: bool, remove or keep logs
        """
        # Work dir for scripts and PBS files.
        self.output_dir = output_dir
        self.level_idx = level_idx
        self.keep_collected = keep_collected

        # Number of operation for fine simulations
        self.n_ops_estimate = None
        self.running_header_set = False

        self.collected_log_content = []
        self.running_log_content = []

        self.collected_log_ = None
        self.running_log_ = None

        if output_dir is not None:
            # Get log files
            self.log_running_file = os.path.join(self.output_dir, "running_log_{:d}.json".format(self.level_idx))
            self.log_collected_file = os.path.join(self.output_dir, "collected_log_{:d}.json".format(self.level_idx))

        self.reload_logs()

    @property
    def running_log(self):
        if self.running_log_ is None:
            self.open()
        return self.running_log_

    @property
    def collected_log(self):
        if self.collected_log_ is None:
            self.open()
        return self.collected_log_

    def reload_logs(self, log_collected_file=None):
        """
        Read collected and running simulations data from log file
        :return: None
        """
        self.close()
        if log_collected_file is None:
            log_collected_file = self.log_collected_file
        try:
            with open(log_collected_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:

                    for line in lines:
                        print(line)
                        json.loads(line)
                    self.collected_log_content = [json.loads(line) for line in lines]
        except FileNotFoundError:
            self.collected_log_content = []
        try:
            with open(self.log_running_file, 'r') as f:
                self.running_log_content = [json.loads(line) for line in f.readlines()]
        except FileNotFoundError:
            self.running_log_content = []

    def open(self, flag='a'):
        self.running_log_ = open(self.log_running_file, flag)
        self.collected_log_ = open(self.log_collected_file, flag)

    def close(self):
        if self.running_log_ is not None:
            self.running_log_.close()
        if self.collected_log_ is not None:
            self.collected_log_.close()

    def log_simulations(self, simulations, collected=False):
        """
        Log simulations
        :param simulations: array of simulations
        :param collected: bool, save collected simulations
        :return: None
        """
        if self.output_dir is None or not simulations:
            return
        if collected:
            log_file = self.collected_log
            if not self.keep_collected:
                self._rm_samples(simulations)
        else:
            log_file = self.running_log
            # n_ops_estimate is already in log file
            if self.n_ops_estimate > 0 and not self.running_header_set:
                log_file.write(json.dumps([self.n_ops_estimate]))
                log_file.write("\n")
                self.running_header_set = True

        for sim in simulations:
            log_file.write(json.dumps(sim))
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
        :param simulations: 
        :return: None
        """
        for sim in simulations:
            _, _, fine, coarse, _ = sim
            if coarse is not None and os.path.isdir(coarse[1]):
                shutil.rmtree(coarse[1], ignore_errors=True)
            if os.path.isdir(fine[1]):
                shutil.rmtree(fine[1], ignore_errors=True)

