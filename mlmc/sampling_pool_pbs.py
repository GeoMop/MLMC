import os
import shutil
import subprocess
import re
import pickle
import json
import glob
from mlmc.level_simulation import LevelSimulation
from mlmc.sampling_pool import SamplingPool
from mlmc.tool.pbs_job import PbsJob

"""
SamplingPoolPBS description
    - this class inherits from SampleStorage, both abstract methods and other crucial ones are described

    schedule_sample(sample_id, level_sim)
        - serialize level_sim (mlmc/level_simulation.py), pickle is used
        - compute random seed from sample_id
        - add (level_sim._level_id, sample_id, seed) to job's scheduled samples 
        - add job weight, increment number of samples in job and execute if job_weight is exceeded

    execute()
        - it is call when job weight (Maximum sum of task sizes summation in single one job) is exceeded
        - methods from mlmc/pbs_job.py are called
            - PbsJob class is created and serialized (PbsJob static method does both)
            - scheduled samples are saved through PbsJob class static method
        - pbs script is written out and ready to run

    get_finished()
        - run execute()
        - get finished_pbs_jobs and unfinished_pbs_jobs from qstat output
        - call get_result_files(), it returns successful samples, failed samples and times all of that is return to Sampler

    _get_result_files()
        - set n_running - number of running samples, it is given from unfinished_pbs_jobs
        - successful samples, failed samples and run times are retrieved from PbsJob class with given job_id
        - if there are _unfinished_sample_ids ('renew' command was use) these samples are appended to previous ones


    This class cooperates with PbsJob (mlmc/pbs_job), which is used as "mediator" between master process and
    worker (job) process. Data which are necessary for worker process are passed to PbsJob from SampleStoragePbs. 
    Master process serializes PbsJob instance.
    Then PbsJob is deserialized in worker process.

"""


class SamplingPoolPBS(SamplingPool):
    """
    Sampling pool PBS (Portable batch system) runtime environment
    """

    OUTPUT_DIR = "output"
    JOBS_DIR = "jobs"
    LEVEL_SIM_CONFIG = "level_{}_simulation_config"  # Serialized level simulation
    JOB = "{}_job.sh"  # Pbs process
    QSUB_FAILED_MAX_N = 10  # Ignore 10 consecutive ''qsub' command failures
    QSTAT_FAILED_MAX_N = 10  # Ignore 10 consecutive 'qstat' command failures

    def __init__(self, work_dir, debug=False):
        """
        :param work_dir: Path to working directory
        :param debug: bool, if True keep sample directories
                      it is the strongest parameter so it overshadows 'clean' param
        """
        self._work_dir = os.path.abspath(work_dir)
        # Working directory - other subdirectories are created in this one
        self._current_job_weight = 0
        # Current job weight.
        # Job is scheduled when current job weight is above 1 (this condition replaces previous job_weight param)
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
        self._unfinished_sample_ids = set()
        # List of sample id which are not collected - collection attempts are done in the get_finished()
        self._debug = debug
        # If true then keep sample directories
        super().__init__(self._work_dir, self._debug)
        self._jobs_dir = self._create_dir(directory=SamplingPoolPBS.JOBS_DIR)
        self._job_count = self._get_job_count()
        # Current number of jobs - sort of jobID
        self._qsub_failed_n = 0
        self._qstat_failed_n = 0
        # Number of failed execution of commands qsub, qstat

    def _get_job_count(self):
        """
        Get number of created jobs
        :return:
        """
        files_pattern = os.path.join(self._jobs_dir, "*_job.sh")
        files = glob.glob(files_pattern)
        if not files:
            return 0

        job_id = re.findall(r'(\d+)_job.sh', files[-1])[0]
        return int(job_id) + 1

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
        :param kwargs: Dict[
                            env_setting: environmental setting - load modules, install packages, ...
                            n_nodes: number of used nodes,
                            n_cores: number of cores a node,
                            mem: used memory a job,
                            queue: used queue on the server,

                            optional params:
                                select_flags: other select flags, see https://wiki.metacentrum.cz/wiki/About_scheduling_system for other possible parameters
                                python: python command, default: python3
        ]
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

        if 'std_out_err' not in kwargs:
            kwargs['std_out_err'] = 'oe'  # Standard error and standard output are  merged  into standard output.

        self._pbs_header_template = ["#!/bin/bash",
                                     '#PBS -S /bin/bash',
                                     '#PBS -l select={n_nodes}:ncpus={n_cores}:mem={mem}{select_flags}',
                                     '#PBS -l walltime={walltime}',
                                     '#PBS -q {queue}',
                                     '#PBS -N {pbs_name}',
                                     '#PBS -j {std_out_err}',  # Specifies  whether and how to join the job's
                                     # standard error and standard output streams.
                                     '#PBS -o {pbs_output_dir}/{job_name}.OU',
                                     '#PBS -e {pbs_output_dir}/{job_name}.ER'
                                     ]

        self._pbs_header_template.extend(
            kwargs['optional_pbs_requests'])  # e.g. ['#PBS -m ae'] means mail is sent when the job aborts or terminates
        self._pbs_header_template.extend(('MLMC_WORKDIR=\"{}\"'.format(self._work_dir),))
        self._pbs_header_template.extend(kwargs['env_setting'])
        self._pbs_header_template.extend(('{python} -m mlmc.tool.pbs_job {output_dir} {job_name} >'
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
        self._scheduled.append((level_sim._level_id, sample_id, seed))

        self._n_samples_in_job += 1
        self._current_job_weight += level_sim.task_size
        if self._current_job_weight > 1 or self._n_samples_in_job > 1000:
            self.execute()

    def serialize_level_sim(self, level_sim: LevelSimulation):
        """
        Pickle LevelSimulation instance
        :param level_sim: LevelSimulation
        :return: None
        """
        file_path = os.path.join(self._output_dir, SamplingPoolPBS.LEVEL_SIM_CONFIG.format(level_sim._level_id))

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
            pbs_process = PbsJob.create_job(self._output_dir, self._jobs_dir, job_id,
                                            SamplingPoolPBS.LEVEL_SIM_CONFIG, self._debug)

            pbs_process.save_sample_id_job_id(job_id, self._scheduled)
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

            process = subprocess.run(['qsub', job_file], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            try:
                if process.returncode != 0:
                    raise Exception(process.stderr.decode('ascii'))
                # Find all finished jobs
                self._qsub_failed_n = 0
                # Write current job count
                self._job_count += 1

                # Get pbs_id from qsub output
                pbs_id = process.stdout.decode("ascii").split(".")[0]
                # Store pbs id for future qstat calls
                self._pbs_ids.append(pbs_id)
                pbs_process.write_pbs_id(pbs_id)

                self._current_job_weight = 0
                self._n_samples_in_job = 0
                self._scheduled = []
            except:
                self._qsub_failed_n += 1
                if self._qsub_failed_n > SamplingPoolPBS.QSUB_FAILED_MAX_N:
                    raise Exception(process.stderr.decode("ascii"))

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
        return self._get_result_files(finished_pbs_jobs, unfinished_pbs_jobs)

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
            try:
                if process.returncode != 0:
                    raise Exception(process.stderr.decode("ascii"))
                output = process.stdout.decode("ascii")
                # Find all finished jobs
                finished_pbs_jobs = re.findall(r"(\d+)\..*\d+ F", output)
                self._qstat_failed_n = 0
            except:
                self._qstat_failed_n += 1
                if self._qstat_failed_n > SamplingPoolPBS.QSTAT_FAILED_MAX_N:
                    raise Exception(process.stderr.decode("ascii"))
                finished_pbs_jobs = []

        # Get unfinished as diff between planned and finished
        unfinished_pbs_jobs = []
        for pbs_id in self._pbs_ids:
            if pbs_id not in finished_pbs_jobs:
                unfinished_pbs_jobs.append(pbs_id)
            else:
                # Remove finished ids from all saved pbs_ids
                # It prevents qstat exception: "Unknown Job Id",
                # that occurs because there is some kind of qstat 'forgetfulness' of terminated jobs
                # It is a very rare phenomenon, which is observed only during the long run (e.g. 8 hours for a job)
                # of many (e.g. 2500) simulations, here it happened after around a day and a half of running MLMC.
                self._pbs_ids.remove(pbs_id)

        return finished_pbs_jobs, unfinished_pbs_jobs

    def _get_result_files(self, finished_pbs_jobs, unfinished_pbs_jobs):
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
            if len(file) > 0:
                job_id = re.findall(r'(\d+)_\d+', file[0])[0]
                n_running += PbsJob.get_job_n_running(job_id, self._jobs_dir)

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
                successful, failed, time = PbsJob.read_results(job_id, self._jobs_dir)

                # Split results to levels
                for level_id, results in successful.items():
                    successful_results.setdefault(level_id, []).extend(results)
                for level_id, results in failed.items():
                    failed_results.setdefault(level_id, []).extend(results)
                for level_id, results in time.items():
                    if level_id in times:
                        times[level_id][0] += results[-1][0]
                        times[level_id][1] += results[-1][1]
                    else:
                        times[level_id] = list(results[-1])

                # Delete pbsID file - it means job is finished
                SamplingPoolPBS.delete_pbs_id_file(file)

        if self._unfinished_sample_ids:
            successful_results, failed_results, times = self._collect_unfinished(successful_results,
                                                                                 failed_results, times)

        return successful_results, failed_results, n_running, list(times.items())

    def _collect_unfinished(self, successful_results, failed_results, times):
        """
        Collect samples which had finished after main process crashed, append them to new collected samples
        :param successful_results: dict
        :param failed_results: dict
        :param times: dict
        :return: all input dictionaries
        """
        already_collected = set()
        for sample_id in self._unfinished_sample_ids:
            if sample_id in already_collected:
                continue

            job_id = PbsJob.job_id_from_sample_id(sample_id, self._jobs_dir)
            successful, failed, time = PbsJob.read_results(job_id, self._jobs_dir)

            # Split results to levels
            for level_id, results in successful.items():
                for res in results:
                    if res[0] in self._unfinished_sample_ids:
                        already_collected.add(res[0])
                        successful_results.setdefault(level_id, []).append(res)

            for level_id, results in failed_results.items():
                for res in results:
                    if res[0] in self._unfinished_sample_ids:
                        already_collected.add(res[0])
                        failed_results.setdefault(level_id, []).append(res)

            for level_id, results in times.items():
                for res in results:
                    if res[0] in self._unfinished_sample_ids:
                        times.setdefault(level_id, []).append(res)
                times[level_id] = results

            # Delete pbsID file - it means job is finished
            # SamplingPoolPBS.delete_pbs_id_file(file)

        self._unfinished_sample_ids = set()

        return successful_results, failed_results, times

    def have_permanent_samples(self, sample_ids):
        """
        List of unfinished sample ids, the corresponding samples are collecting in next get_finished() call .
        """
        self._unfinished_sample_ids = set(sample_ids)

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
