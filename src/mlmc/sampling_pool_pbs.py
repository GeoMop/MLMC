import os
import shutil
import subprocess
import re
import pickle
import json
import glob
from mlmc.level_simulation import LevelSimulation
from mlmc.sampling_pool import SamplingPool
from mlmc.tool.pbs_process import PbsProcess


class SamplingPoolPBS(SamplingPool):

    OUTPUT_DIR = "output"
    JOBS_DIR = "jobs"
    LEVEL_SIM_CONFIG = "level_{}_simulation_config"  # Serialized level simulation
    JOB = "{}_job.sh"  # Pbs process file

    def __init__(self, work_dir, job_weight=200000, job_count=0):
        """
        :param work_dir: Path to working directory
        :param job_weight: Maximum sum of task sizes summation in single one job, if this value is exceeded then the job is executed
        :param job_count: Number of created jobs, it holds the current jobID, useful for regenerating of failed samples (jobs)
        """
        self._work_dir = work_dir
        # Working directory - other subdirectories are created in this one
        self.job_weight = job_weight
        # Weight of the single PBS script (putting more small jobs into single PBS job).
        self._current_job_weight = 0
        # Current collected weight.
        self._job_count = job_count
        # Current number of jobs - sort of jobID
        self._n_samples_in_job = 0
        # Number of samples in job
        self.pbs_script = None
        self._pbs_config = None
        # Data inserted to the pbs script
        self._pbs_header_template = None
        # Lines to put at the beginning of the PBS script.
        self._scheduled = []
        # List of scheduled samples
        self._pbs_ids = []
        # List of pbs job ids which should run

        self._output_dir = None
        self._jobs_dir = None
        self._create_output_dir()
        self._create_job_dir()

    def _create_output_dir(self):
        """
        Create output dir in working directory, remove existing one
        :return: None
        """
        self._output_dir = os.path.join(self._work_dir, SamplingPoolPBS.OUTPUT_DIR)

        if os.path.isdir(self._output_dir):
            shutil.rmtree(self._output_dir)

        os.makedirs(self._output_dir, mode=0o775, exist_ok=True)

    def _create_job_dir(self):
        """
        Create job directory - contains all necessary job files
        :return: None
        """
        self._jobs_dir = os.path.join(self._output_dir, SamplingPoolPBS.JOBS_DIR)
        os.makedirs(self._jobs_dir, mode=0o775, exist_ok=True)

    def _save_structure(self):
        """
        Save structure of files which are needed for pbs process run
        :return: None
        """
        files_structure = {"job_dir": self._jobs_dir,
                           "level_sim_config": os.path.join(self._output_dir, SamplingPoolPBS.LEVEL_SIM_CONFIG)
                           }

        with open(self._files_structure, "w") as writer:
            json.dump(files_structure, writer)

    def pbs_common_setting(self, **kwargs):
        """
        Values for common header of script
        :param kwargs: dict with params vales
        :return: None
        """
        # Script header
        select_flags_list = kwargs.get('select_flags', [])
        if select_flags_list:
            kwargs['select_flags'] = ":" + ":".join(select_flags_list)
        else:
            kwargs['select_flags'] = ""

        # Python3 by default
        if 'python' not in kwargs:
            kwargs['python'] = "python3"

        self._pbs_header_template = ["#!/bin/bash",
                                     '#PBS -S /bin/bash',
                                     '#PBS -l select={n_nodes}:ncpus={n_cores}:mem={mem}{select_flags}',
                                     '#PBS -l walltime=1:00:00',
                                     '#PBS -q {queue}',
                                     '#PBS -N MLMC_sim',
                                     '#PBS -j oe',
                                     '#PBS -o {pbs_output_dir}/{job_name}.OU',
                                     '#PBS -e {pbs_output_dir}/{job_name}.ER',
                                     '']

        self._pbs_header_template.extend(('cd {work_dir}',
                                          'source env/bin/activate',
                                          'pip3 install /storage/liberec3-tul/home/martin_spetlik/MLMC_new_design',))

        self._pbs_header_template.extend(kwargs['modules'])
        self._pbs_header_template.extend(('{python} {pbs_process_file_dir}/pbs_process.py {output_dir} {job_name} >'
                                          '{pbs_output_dir}/{job_name}_STDOUT 2>&1',))
        self._pbs_config = kwargs

    def schedule_sample(self, sample_id, level_sim):
        """
        Add sample to current PBS package
        :param sample_id: unique sample id from Sampler
        :param level_sim: LevelSimulation instance
        :return: None
        """
        self.serialize_level_sim(level_sim)

        seed = self.compute_seed(sample_id)
        self._scheduled.append((level_sim.level_id, sample_id, seed))

        self._n_samples_in_job += 1
        self._current_job_weight += level_sim.task_size
        if self._current_job_weight > self.job_weight:
            self.execute()

    def serialize_level_sim(self, level_sim: LevelSimulation):
        """
        Pickle LevelSimulation instance
        :param level_sim: LevelSimulation
        :return: None
        """
        file_path = os.path.join(self._output_dir, SamplingPoolPBS.LEVEL_SIM_CONFIG.format(level_sim.level_id))

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                pickle.dump(level_sim, f)

    def execute(self):
        """
        Execute pbs script
        :return: None
        """
        if len(self._scheduled) > 0:
            job_id = "{:04d}".format(self._job_count)
            # Create pbs job
            pbs_process = PbsProcess.create_job(self._output_dir, self._jobs_dir, job_id,
                                                                       SamplingPoolPBS.LEVEL_SIM_CONFIG)
            # Write scheduled samples to file
            pbs_process.save_scheduled(self._scheduled)

            # Format pbs script
            self._create_script()

            if self.pbs_script is None or self._n_samples_in_job == 0:
                return

            # Write pbs script
            job_file = os.path.join(self._jobs_dir, SamplingPoolPBS.JOB.format(job_id))
            script_content = "\n".join(self.pbs_script)
            self.write_script(script_content, job_file)
            self._job_count += 1

            #subprocess.call(job_file)

            process = subprocess.run(['qsub', job_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if process.returncode != 0:
                raise Exception(process.stderr.decode('ascii'))

            # Get pbs_id from qsub output
            pbs_id = process.stdout.decode("ascii").split(".")[0]
            # Store pbs id for future qstat calls
            self._pbs_ids.append(pbs_id)
            pbs_process.write_pbs_id(pbs_id)

        self._current_job_weight = 0
        self._n_samples_in_job = 0
        self._scheduled = []

    def _create_script(self):
        """
        Format pbs script
        :return: None
        """
        # Job output with similar name to job
        self._pbs_config['job_name'] = "{:04d}".format(self._job_count)
        self._pbs_config['pbs_output_dir'] = self._jobs_dir
        self._pbs_config['output_dir'] = self._output_dir
        self._pbs_config['work_dir'] = self._work_dir

        self.pbs_script = [line.format(**self._pbs_config) for line in self._pbs_header_template]

    def write_script(self, content, job_file):
        """
        Create 
        :param content: script content
        :param job_file: job file path
        :return: None
        """
        with open(job_file, "w") as f:
            f.write(content)
        os.chmod(job_file, 0o774)

    def get_finished(self):
        """
        Get results
        :return:
        """
        self.execute()
        finished_pbs_jobs, unfinished_pbs_jobs = self._qstat_pbs_job()
        return self.get_result_files(finished_pbs_jobs, unfinished_pbs_jobs)

    def _qstat_pbs_job(self):
        """
        Parse qstat output and get all unfinished job ids
        :return: finished and unfinished jobs both list of job ids (str)
        """
        finished_pbs_jobs = []
        if len(self._pbs_ids) > 0:
            # Get PBS id's status,
            # '-x' - displays status information for finished and moved jobs in addition to queued and running jobs.
            qstat_call = ["qstat", "-x"]
            qstat_call.extend(self._pbs_ids)

            # qstat call
            process = subprocess.run(qstat_call, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if process.returncode != 0:
                raise Exception(process.stderr.decode("ascii"))

            output = process.stdout.decode("ascii")
            # Find all finished jobs
            finished_pbs_jobs = re.findall(r"(\d+)\..*\d+ F", output)

        # Get unfinished as diff between planned and finished
        unfinished_pbs_jobs = []
        for pbs_id in self._pbs_ids:
            if pbs_id not in finished_pbs_jobs:
                unfinished_pbs_jobs.append(pbs_id)

        return finished_pbs_jobs, unfinished_pbs_jobs

    def get_result_files(self, finished_pbs_jobs, unfinished_pbs_jobs):
        """
        Get results from files
        :param finished_pbs_jobs: List[str], finished pbs jobs,
        :param unfinished_pbs_jobs: List[str], unfinished pbs jobs,
        :return: successful_results: Dict[level_id, List[Tuple[sample_id: str, Tuple[fine_result: np.ndarray, coarse_result: n.ndarray]]]]
                 failed_results: Dict[level_id, List[Tuple[sample_id: str, err_msg: str]]]
                 n_running: int, number of running samples
                 times:
        """
        os.chdir(self._jobs_dir)

        # Get number of running samples
        n_running = 0
        for pbs_id in unfinished_pbs_jobs:
            reg = "*_{}".format(pbs_id)
            file = glob.glob(reg)
            job_id = re.findall(r'(\d+)_\d+', file[0])[0]
            n_running += PbsProcess.get_job_n_running(job_id, self._jobs_dir)

        successful_results = {}
        failed_results = {}
        times = {}
        for pbs_id in finished_pbs_jobs:
            reg = "*_{}".format(pbs_id)  # JobID_PbsId file
            file = glob.glob(reg)

            if file:
                # Find jobID
                file = file[0]
                job_id = re.findall(r'(\d+)_\d+', file)[0]
                # Get sample results
                successful, failed, time = PbsProcess.read_results(job_id, self._jobs_dir)

                # Split results to levels
                for level_id, results in successful.items():
                    successful_results.setdefault(level_id, []).extend(results)
                for level_id, results in failed_results.items():
                    failed_results.setdefault(level_id, []).extend(results)
                for level_id, results in times.items():
                    times[level_id] = results

                # Delete pbsID file - it means job is finished
                SamplingPoolPBS.delete_pbs_id_file(file)

        # @TODO: get sample ids for failed pbs jobs and add these samples to failed ones with particular error message

        return successful_results, failed_results, n_running, times

    def have_permanent_sample(self, sample_id):
        """
        List of permanent samples or find per call?
        """

    @staticmethod
    def delete_pbs_id_file(file_path):
        """
        Delete jobId_pbsId file - it indicates finished job
        :param file_path: str
        :return: None
        """
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print("Failed to remove PBS id file, file not found")
