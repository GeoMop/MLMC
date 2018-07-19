import os
import os.path
import shutil
import subprocess
import yaml
import shlex

import glob
import json
import numpy as np


class FlowPbs:

    def __init__(self, work_dir=None, package_weight=200000, qsub=None, clean=False):
        """

        :param work_dir: if None, means no logging and just direct execution.
        :param package_weight:
        :param qsub: string with qsub command.
        """
        # Weight of the single PBS script (putting more small jobs into single PBS job).
        self.package_weight = package_weight
        # Current collected weight.
        self._current_package_weight = 0
        # Number of executed packages.
        self._package_count = 0
        # Work dir for scripts and PBS files.
        self.work_dir = work_dir

        # Lines to put at the beginning of the PBS script.
        self.pbs_script_heading = None
        self.pbs_script = None
        # Set q sub command or direct execution.
        self.qsub_cmd = qsub

        self.collected_log_content = None
        self.running_log_content = None

        self.collected_log_ = None
        self.running_log_ = None

        if work_dir is not None:
            # Get log files
            self.log_running_file = os.path.join(self.work_dir, "running_log.json")
            self.log_collected_file = os.path.join(self.work_dir, "collected_log.json")
            if clean:
                # Fresh work dir.
                if os.path.isdir(self.work_dir):
                    shutil.rmtree(self.work_dir)
                os.makedirs(self.work_dir, mode=0o775, exist_ok=True)
                self.open(flag='w')


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

    def reload_logs(self):
        """
        Read collected and running simulations data from log file
        :return: None
        """
        self.close()
        try:
            with open(self.log_collected_file, 'r') as f:
                self.collected_log_content = [json.loads(line) for line in f.readlines()]
                #self.collected_log_content = []
                #for line in f.readlines():                    
                #    self.collected_log_content.append(json.loads(line))
        except FileNotFoundError:
            self.collected_log_content = []
        try:
            with open(self.log_running_file, 'r') as f:
                self.running_log_content = [json.loads(line) for line in f.readlines()]
        except FileNotFoundError:
            self.running_log_content = []


    # def check_finished_jobs(self):
    #     """
    #     Pass through the list of queued simulations, group simulations by packages,
    #     check which packages are done, check if if their simulations are done.
    #
    #     TODO: write simulation tags into stdout and report end of the whole package. Need not to
    #     check sucessfuly completed packages.
    #     :return:
    #     """
    #     running_log_content = []
    #     package_jobs_idx = 0
    #     for sim in self.running_log_content:
    #         if len(sim) == 1:
    #             job_str = sim[0]
    #             # self.check_job(job_str, running_log_content[package_jobs_idx:])
    #             package_jobs_idx = len(running_log_content)
    #         else:
    #             running_log_content.append(sim)
    #     self.running_log_content = running_log_content

    # def check_job(self, job_str, jobs):
    #     """
    #     Check if the (finished) package is complete, otherwise check all its simulations, and rerun those that are not finished.
    #     :param job_str:
    #     :param jobs:
    #     :return:
    #     """
    #
    #     with open(os.path.join(self.work_dir, job_str + ".OU"), 'r') as f:
    #         content = f.readlines()
    #         if content[-1].find("SUCCESS.") > -1:
    #             return
    #         else:
    #             finished = set()
    #             for l in content:
    #                 if l.find("Finished simulation: ") > -1:
    #                     _, flow123d, sim_tag, sim_dir = shlex.split(l)
    #
    #             # check individual jobs

    def open(self, flag='a'):
        self.running_log_ = open(self.log_running_file, flag)
        self.collected_log_ = open(self.log_collected_file, flag)

    def close(self):
        if self.running_log_ is not None:
            self.running_log_.close()
        if self.collected_log_ is not None:
            self.collected_log_.close()

    def pbs_common_setting(self, **kwargs):
        """
        Values for common header of script
        :param kwargs: dict with params vales
        :return: None
        """
        kwargs['pbs_output_dir'] = self.work_dir
        # Script header
        select_flags_list = kwargs.get('select_flags', [])
        if select_flags_list:
            kwargs['select_flags'] = ":" + ":".join(select_flags_list)
        else:
            kwargs['select_flags'] = ""
            
        pbs_header_template = ["#!/bin/bash",
                               '#PBS -S /bin/bash',
                               '#PBS -l select={n_nodes}:ncpus={n_cores}:mem={mem}{select_flags}',
                               '#PBS -q {queue}',
                               '#PBS -N Flow123d',
                               '#PBS -j oe',
                               '#PBS -o {pbs_output_dir}',
                               '#PBS -e {pbs_output_dir}',
                               '']

        self.pbs_script_heading = [line.format(**kwargs) for line in pbs_header_template]
        self.clean_script()

    def add_realization(self, weight, **kwargs):
        """
        Append new flow123d realization to the existing script content
        :param weight: current simulation steps
        :param kwargs: dict with params
        :return: None
        """
        assert self.pbs_script is not None

        lines = [
            'cd {work_dir}',
            'date +%y.%m.%d_%H:%M:%S',
            'time -p {flow123d} --yaml_balance -i {output_subdir} -s {work_dir}/flow_input.yaml  -o {output_subdir} >{work_dir}/{output_subdir}/flow.out 2>&1',
            'date +%y.%m.%d_%H:%M:%S',
            'touch {output_subdir}/FINISHED',
            'echo \\"Finished simulation:\\" \\"{flow123d}\\" \\"{work_dir}\\" \\"{output_subdir}\\"',
            '']
        lines = [line.format(**kwargs) for line in lines]
        self.pbs_script.extend(lines)

        self._current_package_weight += weight
        if self._current_package_weight > self.package_weight:
            self.execute()
            self._current_package_weight = 0

    def execute(self):
        """
        Execute script
        :return: None
        """
        if self.pbs_script is None:
            return
        self.pbs_script.append("echo SUCCESS.")
        script_content = "\n".join(self.pbs_script)
        pbs_file = os.path.join(self.work_dir, "package_{:04d}.sh".format(self._package_count))
        self._package_count += 1
        with open(pbs_file, "w") as f:
            f.write(script_content)
        os.chmod(pbs_file, 0o774)  # Make exectutable to allow direct call.
        if self.qsub_cmd is None:
            subprocess.call(pbs_file)
        else:
            process = subprocess.run([self.qsub_cmd, pbs_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if process.returncode != 0:
                raise Exception(process.stderr.decode('ascii'))
            job_str = process.stdout.decode('ascii').split("\n")[0]
            line = [pbs_file, job_str]
            self.running_log.write(json.dumps(line))
            self.running_log.write("\n")

        # Clean script for other usage
        self.clean_script()

    def clean_script(self):
        """
        Clean script keep just header
        :return: None
        """
        self.pbs_script = self.pbs_script_heading.copy()

    def log_simulations(self, simulations, collected=False):
        """
        Log simulations
        :param simulations: array of simulations
        :param collected: bool, save collected simulations
        :return: None
        """

        if self.work_dir is None or not simulations:
            return
        if collected:
            log_file = self.collected_log
        else:
            log_file = self.running_log

        for sim in simulations:
            log_file.write(json.dumps(sim))
            log_file.write("\n")
        log_file.flush()

    def estimate_level_times(self, n_samples=20):
        output_dir = os.path.join(self.work_dir, "..")
        level_times = []
        for dir_entry in os.scandir(output_dir):
            sim_dir = os.path.join(dir_entry.path, "samples")
            print(sim_dir)
            if os.path.isdir(sim_dir) and dir_entry.name.startswith("sim_"):
                times = []
                sample_entries = list(os.scandir(sim_dir))
                total_samples = len(sample_entries)
                if total_samples > n_samples:
                    indices = np.random.choice(np.arange(total_samples, dtype=int), size=n_samples)
                    selected_samples = [ sample_entries[i] for i in indices]
                else:
                    selected_samples = sample_entries
                
                for sample_entry in selected_samples:
                    sample_dir = sample_entry.path
                    print("   ", sample_dir)
                    prof_files = list(glob.iglob(os.path.join(sample_dir, "profiler_*.json")))
                    #assert len(prof_files) == 1, "N: " + str(prof_files)
                    with open(prof_files[-1], 'r') as f:
                        prof_data = json.load(f)
                    total_time = float(prof_data['children'][0]['cumul-time-max'])
                    times.append(total_time)
                    if len(times) > 100:
                        break
                level_times.append(np.mean(np.array(times)))
        return level_times
