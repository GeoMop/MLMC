import os
import shutil
import subprocess
import yaml
from sampling_pool import SamplingPool


class SamplingPoolPBS(SamplingPool):

    def __init__(self, job_weight=200000, job_count=0):
        """
        :param job_weight: Number of simulation elements per job script
        :param job_count: Number of created jobs
        """
        # Weight of the single PBS script (putting more small jobs into single PBS job).
        self.job_weight = job_weight
        # Current collected weight.
        self._current_job_weight = 0
        # Number of executed jobs.
        self._job_count = job_count
        self.max_realizations = 10
        self._number_of_realizations = 0
        # Lines to put at the beginning of the PBS script.
        self.pbs_script_heading = None
        self.pbs_script = None
        # Set q sub command or direct execution.
        # self.qsub_cmd = qsub
        self._pbs_config = None
        self._pbs_header_template = None

        self.need_workspace = True
        self.need_jobs = True

        self.workspace = None

        self.qsub_cmd = None

        self._scheduled = []

    def pbs_common_setting(self, flow_3=False, **kwargs):
        """
        Values for common header of script
        :param flow_3: use flow123d version 3.0.0
        :param kwargs: dict with params vales
        :return: None
        """
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

        self._pbs_header_template.extend(('python {pbs_process_file_dir}/pbs_process.py {files_structure} {job_name} >{pbs_output_dir}/{job_name}_STDOUT 2>&1',))
        self._pbs_config = kwargs

    def schedule_sample(self, sample_id, level_sim):
        """
        Add sample to current PBS package
        :param sample_id: unique sample id from Sampler
        :param level_sim: LevelSimulation instance
        :return: None
        """
        level_sim.config_dict["sample_id"] = sample_id

        self._scheduled.append((level_sim.level_id, sample_id))

        self._number_of_realizations += 1
        self._current_job_weight += level_sim.task_size
        if self._current_job_weight > self.job_weight or self._number_of_realizations > self.max_realizations:
            self.execute()

    def have_permanent_sample(self, sample_id):
        """
        List of permanent samples or find per call?
        """
        pass

    def get_finished(self):
        """
        1) close PBS job, run it by 'qsub'
        2) parse output and get PBS ID
        3) received PBS ID save to XYZ_pbs_id
        4) closing is also called from schedule sample in case of package limit exceeding
        :return:
        """
        self.execute()

    def _parse_qsub_output(self, output):
        print("output ", output)
        exit()

    def _create_script(self):
        # Job output with similar name to job
        self._pbs_config['job_name'] = "{:04d}".format(self._job_count)
        self._pbs_config['pbs_output_dir'] = self.workspace.jobs_dir
        self._pbs_config['files_structure'] = self.workspace.files_structure

        print("pbs header template ", self._pbs_header_template)

        self.pbs_script = [line.format(**self._pbs_config) for line in self._pbs_header_template]

        print("pbs script ", self.pbs_script)

    def execute(self):
        """
        Execute pbs script
        :return: None
        """
        job_id = "{:04d}".format(self._job_count)
        self.workspace.create_files(job_id)
        self.workspace.save_scheduled(self._scheduled)

        if self.pbs_script is None:
            self._create_script()

        if self.pbs_script is None or self._number_of_realizations == 0:
            return

        # Write pbs script
        script_content = "\n".join(self.pbs_script)
        self.workspace.write_pbs_job_file(script_content)
        self._job_count += 1

        #  @TODO: qsub command is required for PBS
        if self.qsub_cmd is None:
            process = subprocess.call(self.workspace.pbs_job_file)
        else:
            process = subprocess.run([self.qsub_cmd, self.workspace.pbs_job_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if process.returncode != 0:
                raise Exception(process.stderr.decode('ascii'))

        self._parse_qsub_output(process)

        # Clean script for other usage
        # self.clean_script()
        self._current_job_weight = 0
        self._number_of_realizations = 0
