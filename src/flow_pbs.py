import os
import os.path
import shutil
import subprocess
import yaml


class FlowPbs:

    def __init__(self, work_dir=None, package_weight=200000, qsub=None, reload=False):
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

        if work_dir is not None:
            log_running_file = os.path.join(self.work_dir, "running_log.yaml")
            log_collected_file = os.path.join(self.work_dir, "collected_log.yaml")
            open_flag = 'w'
            if reload:
                with open(log_collected_file, 'r') as f:
                    self.collected_log_content = yaml.load(f)
                open_flag = 'a'
            else:
                # Fresh work dir.
                if os.path.isdir(self.work_dir):
                    shutil.rmtree(self.work_dir)
                os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

            self.running_log = open(log_running_file, open_flag)
            self.collected_log = open(log_collected_file, open_flag)

    def close(self):
        self.running_log.close()
        self.collected_log.close()


    def pbs_common_setting(self, **kwargs):
        """
        Values for common header of script
        :param kwargs: dict with params vales
        :return: None
        """
        # Script header
        pbs_header_template = ["#!/bin/bash",
                           '#PBS -S /bin/bash',
                           '#PBS -l select={n_nodes}:ncpus={n_cores}:mem={mem}',
                           '#PBS -q {queue}',
                           '#PBS -N Flow123d',
                           '#PBS -j oe',
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
            'time -p {flow123d} --yaml_balance -i {output_subdir} -s {work_dir}/flow_input.yaml  -o {output_subdir} >{work_dir}/{output_subdir}/flow.out 2>&1',
            'cd {output_subdir}',
            'touch FINISHED',
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
        script_content = "\n".join(self.pbs_script)
        pbs_file = os.path.join(self.work_dir, "package_{:04d}.sh".format(self._package_count))
        self._package_count += 1
        with open(pbs_file, "w") as f:
            f.write(script_content)
        os.chmod(pbs_file, 0o774)  # Make exectutable to allow direct call.
        if self.qsub_cmd is None:
            subprocess.call(pbs_file)
        else:
            subprocess.call(self.qsub_cmd + " " + pbs_file, shell=True)

        # Clean script for other usage
        self.clean_script()

    def clean_script(self):
        """
        Clean script keep just header
        :return: None
        """
        self.pbs_script = self.pbs_script_heading

    def log_simulations(self, level, simulations, values=None):
        if self.work_dir is None or not simulations:
            return
        if values is None:
            log_file = self.running_log
        else:
            log_file = self.collected_log
        value = None
        lines = []
        for i, fine, coarse in simulations:
            if values is not None:
                value = values[i].tolist()
            line = [ level, i, fine, coarse, value ]
            lines.append(line)
        log_file.write(yaml.safe_dump(lines))