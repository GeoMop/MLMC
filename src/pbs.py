import os
import os.path
import shutil
import subprocess
import json
import glob
import numpy as np


class Pbs:
    """
    Create and execute scripts with simulations
    """
    def __init__(self, work_dir=None, package_weight=200000, qsub=None, clean=False):
        """
        :param work_dir: if None, means no logging and just direct execution.
        :param package_weight:
        :param qsub: string with qsub command.
        :param clean: bool, if True, create new scripts directory
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

        if work_dir is not None:
            if clean:
                # Fresh work dir.
                if os.path.isdir(self.work_dir):
                    shutil.rmtree(self.work_dir)
                os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

    def pbs_common_setting(self, flow_3=False, **kwargs):
        """
        Values for common header of script
        :param flow_3: use flow123d version 3.0.0
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
                               '#PBS -l walltime=1:00:00',
                               '#PBS -q {queue}',
                               '#PBS -N Flow123d',
                               '#PBS -j oe',
                               '#PBS -o {pbs_output_dir}',
                               '#PBS -e {pbs_output_dir}',
                               '']
        if flow_3:
            pbs_header_template.extend(('module use /storage/praha1/home/jan-hybs/modules',
                                        'module load flow123d', ''))

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
            'time -p {flow123d} --yaml_balance -i {output_subdir} -s {work_dir}/flow_input.yaml  -o {output_subdir} >{work_dir}/{output_subdir}/flow.out',
            'date +%y.%m.%d_%H:%M:%S',
            'touch {output_subdir}/FINISHED',
            'rm -f {output_subdir}/flow.out',
            'rm -f {output_subdir}/profiler_info*',
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
        Execute pbs script
        :return: None
        """
        if self.pbs_script is None:
            return
        self.pbs_script.append("echo SUCCESS.")
        script_content = "\n".join(self.pbs_script)
        pbs_file = os.path.join(self.work_dir, "package_{:04d}.sh".format(self._package_count))
        self._package_count += 1
        with open(pbs_file, "w") as file_writer:
            file_writer.write(script_content)
        os.chmod(pbs_file, 0o774)  # Make executable to allow direct call.
        if self.qsub_cmd is None:
            subprocess.call(pbs_file)
        else:
            process = subprocess.run([self.qsub_cmd, pbs_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if process.returncode != 0:
                raise Exception(process.stderr.decode('ascii'))

        # Clean script for other usage
        self.clean_script()

    def clean_script(self):
        """
        Clean script and keep header
        :return: None
        """
        self.pbs_script = self.pbs_script_heading.copy()

    def estimate_level_times(self, n_samples=20):
        """
        Estimate average simulation time for each level
        :param n_samples: Number of samples
        :return: list
        """
        output_dir = os.path.join(self.work_dir, "..")
        level_times = []
        for dir_entry in os.scandir(output_dir):
            sim_dir = os.path.join(dir_entry.path, "samples")
            if os.path.isdir(sim_dir) and dir_entry.name.startswith("sim_"):
                times = []
                sample_entries = list(os.scandir(sim_dir))
                total_samples = len(sample_entries)
                if total_samples > n_samples:
                    indices = np.random.choice(np.arange(total_samples, dtype=int), size=n_samples)
                    selected_samples = [sample_entries[i] for i in indices]
                else:
                    selected_samples = sample_entries

                for sample_entry in selected_samples:
                    sample_dir = sample_entry.path
                    prof_files = list(glob.iglob(os.path.join(sample_dir, "profiler_*.json")))
                    # assert len(prof_files) == 1, "N: " + str(prof_files)
                    with open(prof_files[-1], 'r') as file_reader:
                        prof_data = json.load(file_reader)
                    total_time = float(prof_data['children'][0]['cumul-time-max'])
                    times.append(total_time)
                    if len(times) > 100:
                        break
                level_times.append(np.mean(np.array(times)))
        return level_times
