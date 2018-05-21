import os
import os.path
import shutil
import subprocess


class FlowPbs:
    __instance = None

    @staticmethod
    def get_instance():
        """
        Getting instance, class is singleton
        """
        if FlowPbs.__instance is None:
            FlowPbs()
        return FlowPbs.__instance

    def __init__(self, work_dir="", steps_sum=200000, qsub=False, qsub_cmd="qsub"):
        # Number of simulation steps for first pbs script execution
        self.steps_sum = steps_sum
        self.current_steps_sum = self.steps_sum
        self.work_dir = work_dir
        if os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.mkdir(self.work_dir, 0o755);
        # Script header
        self.pbs_script = ["#!/bin/bash",
                           '#PBS -S /bin/bash',
                           '#PBS -l select={n_nodes}:ncpus={n_cores}:mem={mem}',
                           '#PBS -q {queue}',
                           '#PBS -N Flow123d',
                           '#PBS -j oe',
                           '']
        self.pbs_script_heading = None
        # Use -qsub
        self.pbs_qsub = qsub
        self.qsub_cmd = qsub_cmd

        if FlowPbs.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            FlowPbs.__instance = self

    def pbs_common_setting(self, **kwargs):
        """
        Values for common header of script
        :param kwargs: dict with params vales
        :return: None
        """
        lines = [line.format(**kwargs) for line in self.pbs_script]

        self.pbs_script = "\n".join(lines)
        self.pbs_script_heading = self.pbs_script

    def add_realization(self, steps, **kwargs):
        """
        Append new flow123d realization to the existing script content
        :param steps: current simulation steps
        :param kwargs: dict with params
        :return: None
        """
        lines = [
            'cd {work_dir}',
            'time -p {flow123d} --yaml_balance -i {output_subdir} -s {work_dir}/flow_input.yaml  -o {output_subdir} >{work_dir}/{output_subdir}/flow.out 2>&1',
            'cd {output_subdir}',
            'touch FINISHED',
            '']
        lines = [line.format(**kwargs) for line in lines]
        self.pbs_script = self.pbs_script + "\n".join(lines)

        self.current_steps_sum -= steps
        if self.current_steps_sum < 0:
            self.execute()
            self.current_steps_sum = self.steps_sum

    def execute(self):
        """
        Execute script 
        :return: None
        """
        pbs_file = os.path.join(self.work_dir, "pbs_script.sh")
        with open(pbs_file, "w") as f:
            f.write(self.pbs_script)
        os.chmod(pbs_file, 0o774)  # Make exectutable to allow direct call.
        if self.pbs_qsub is True:
            subprocess.call(self.qsub_cmd + " " + pbs_file, shell=True)
        else:
            subprocess.call(pbs_file)

        # Clean script for other usage
        self.clean_script()

    def clean_script(self):
        """
        Clean script keep just header
        :return: None
        """
        self.pbs_script = self.pbs_script_heading
