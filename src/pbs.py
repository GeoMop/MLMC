import os
import os.path
import shutil
import subprocess


class Pbs:
    def __init__(self, work_dir=None, job_weight=200000, job_count=0, qsub=None, clean=False, home_dir="/home"):
        """
        :param work_dir: if None, means no logging and just direct execution.
        :param job_weight: Number of simulation elements per job script
        :param job_count: Number of created jobs
        :param qsub: string with qsub command.
        :param clean: bool, if True, create new scripts directory
        """
        # Weight of the single PBS script (putting more small jobs into single PBS job).
        self.job_weight = job_weight
        # Current collected weight.
        self._current_job_weight = 0
        # Number of executed jobs.
        self._job_count = job_count
        # Work dir for scripts and PBS files.
        self.work_dir = work_dir
        self.max_realizations = 10
        self._number_of_realizations = 0
        # Lines to put at the beginning of the PBS script.
        self.pbs_script_heading = None
        self.pbs_script = None
        # Set q sub command or direct execution.
        self.qsub_cmd = qsub
        self._pbs_config = None
        self._pbs_header_template = None
        self._home_dir = home_dir

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

        self._pbs_header_template = ["#!/bin/bash",
                                     '#PBS -S /bin/bash',
                                     '#PBS -l select={n_nodes}:ncpus={n_cores}:mem={mem}{select_flags}',
                                     '#PBS -l walltime=4:00:00',
                                     '#PBS -q {queue}',
                                     '#PBS -N Flow123d',
                                     '#PBS -j oe',
                                     '#PBS -o {pbs_output_dir}/{job_name}.OU',
                                     '#PBS -e {pbs_output_dir}/{job_name}.ER',
                                     '']
        if flow_3:
            self._pbs_header_template.extend(('module use /storage/praha1/home/jan-hybs/modules',
                                              'module load flow123d', ''))

        self._pbs_header_template.extend(('touch {pbs_output_dir}/RUNNING', 'rm -f {pbs_output_dir}/QUEUED'))

        self._pbs_config = kwargs
        self.clean_script()

    def add_realization(self, weight, **kwargs):
        """
        Append new flow123d realization to the existing script content
        :param weight: current simulation steps
        :param kwargs: dict with params
        :return: None
        """
        if self._number_of_realizations == 0:
            self.clean_script()

        assert self.pbs_script is not None

        lines = [
            'cd {work_dir}',
            'date +%y.%m.%d_%H:%M:%S',
            'time -p {flow123d} --yaml_balance -i {output_subdir} -s {work_dir}/flow_input.yaml  -o {output_subdir} >{work_dir}/{output_subdir}/flow.out',
            'date +%y.%m.%d_%H:%M:%S',
            'touch {output_subdir}/FINISHED',
            'echo \\"Finished simulation:\\" \\"{flow123d}\\" \\"{work_dir}\\" \\"{output_subdir}\\"',
            '']
        lines = [line.format(**kwargs) for line in lines]
        self.pbs_script.extend(lines)

        self._number_of_realizations += 1
        self._current_job_weight += weight
        if self._current_job_weight > self.job_weight or self._number_of_realizations > self.max_realizations:
            self.execute()

        return self._pbs_config['job_name']

    def execute(self):
        """
        Execute pbs script
        :return: None
        """
        if self.pbs_script is None or self._number_of_realizations == 0:
            return
        self.pbs_script.append("touch " + self._job_dir + "/FINISHED")
        self.pbs_script.append("rm -f " + self._job_dir + "/RUNNING")

        script_content = "\n".join(self.pbs_script)
        pbs_file = os.path.join(self._job_dir, "{:04d}.sh".format(self._job_count))

        pbs_file_pom = os.path.join(self._home_dir, "{:04d}.sh".format(self._job_count))

        self._job_count += 1
        with open(pbs_file_pom, "w") as file_writer:
            file_writer.write(script_content)

        os.chmod(pbs_file_pom, 0o774)  # Make executable to allow direct call.
        shutil.copyfile(pbs_file_pom, pbs_file)

        try:
            os.remove(pbs_file_pom)
        except OSError:
            pass

        if self.qsub_cmd is None:
            subprocess.call(pbs_file)
        else:
            process = subprocess.run([self.qsub_cmd, pbs_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            subprocess.call(["touch", os.path.join(self._job_dir, "QUEUED")])
            if process.returncode != 0:
                raise Exception(process.stderr.decode('ascii'))

        # Clean script for other usage
        # self.clean_script()
        self._current_job_weight = 0
        self._number_of_realizations = 0

    def clean_script(self):
        """
        Clean script and keep header
        :return: None
        """
        # Package dir path
        self._job_dir = os.path.join(self.work_dir, "{:04d}".format(self._job_count))

        # Remove dir
        if os.path.isdir(self._job_dir):
            shutil.rmtree(self._job_dir)

        os.makedirs(self._job_dir, mode=0o775, exist_ok=True)
        # Job output with similar name to job
        self._pbs_config['job_name'] = "{:04d}".format(self._job_count)
        self._pbs_config['pbs_output_dir'] = self._job_dir
        self.pbs_script = [line.format(**self._pbs_config) for line in self._pbs_header_template]
