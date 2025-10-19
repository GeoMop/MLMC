import os
import sys
import time
import warnings
import json
import ruamel.yaml as yaml
import pickle
from mlmc.sampling_pool import SamplingPool
from ruamel.yaml.error import ReusedAnchorWarning, UnsafeLoaderWarning
warnings.simplefilter("ignore", UnsafeLoaderWarning)
warnings.simplefilter("ignore", ReusedAnchorWarning)


class PbsJob:
    SCHEDULED = "{}_scheduled.yaml"
    # Store scheduled samples as List[(level_sim._level_id, sample_id, seed)]
    SUCCESSFUL_RESULTS = "{}_successful_results.yaml"
    # Simulation results as Dict[level_id, List[Tuple[sample_id, (fine result, coarse result)]]]
    FAILED_RESULTS = "{}_failed_results.yaml"
    # Failed samples as Dict[level_id, List[Tuple[sample id, error message]]]
    TIME = "{}_times.yaml"
    # Dict[level_id, List[time, finished samples]]
    PBS_ID = "{}_"
    # File which name assign our job id to pbs jobs id 'JobID_Pbs_ID'
    CLASS_FILE = "pbs_process_serialized.txt"
    # Serialized data which are "passed" from sampling pool to pbs process
    SAMPLE_ID_JOB_ID = "sample_id_job_id.json"
    # Sample id with corresponding job id,
    # used to indicate that sample is stored in _successful_results.yaml or _failed_results.yaml

    def __init__(self, output_dir, jobs_dir, job_id, level_sim_file, debug):
        """
        Construct a PbsJob instance used both by SamplingPool (to create a job) and by PBS worker process.

        :param output_dir: str, directory where sample work dirs and outputs live
        :param jobs_dir: str, directory where scheduler/job control files are stored
        :param job_id: str, unique identifier of this job
        :param level_sim_file: str, format string for per-level serialized LevelSimulation files
        :param debug: bool; if True do not remove per-sample directories after successful runs
        """
        self._output_dir = output_dir
        self._jobs_dir = jobs_dir
        self._job_id = job_id
        self._level_sim_file = level_sim_file
        self._debug = debug

        self._level_simulations = {}
        # LevelSimulation instances deserialized on demand

    @classmethod
    def create_job(cls, output_dir, jobs_dir, job_id, level_sim_file, debug):
        """
        Create and serialize a PbsJob descriptor for a PBS process to later deserialize.

        The created descriptor (CLASS_FILE) is written under output_dir for the PBS worker.

        :param output_dir: str
        :param jobs_dir: str
        :param job_id: str
        :param level_sim_file: str, format of LevelSimulation serialization filenames
        :param debug: bool
        :return: PbsJob instance
        """
        pbs_process = cls(output_dir, jobs_dir, job_id, level_sim_file, debug)
        PbsJob._serialize_pbs_process(pbs_process)

        return pbs_process

    @classmethod
    def create_process(cls):
        """
        Create PbsJob instance inside PBS worker process.

        The worker expects command-line arguments (see command_params) and a serialized CLASS_FILE
        in output_dir describing jobs_dir and level_sim_file format.

        :return: PbsJob instance
        """
        job_id, output_dir = PbsJob.command_params()
        jobs_dir, level_sim_file_format, debug = PbsJob._deserialize_pbs_process(output_dir)

        return cls(output_dir, jobs_dir, job_id, level_sim_file_format, debug)

    @staticmethod
    def _serialize_pbs_process(pbs_process):
        """
        Persist minimal information (jobs_dir, level_sim_file format, debug) for PBS worker.

        This function writes CLASS_FILE inside the pbs_process._output_dir for later deserialization.

        :param pbs_process: PbsJob instance to serialize
        :return: None
        """
        if not os.path.exists(os.path.join(pbs_process._output_dir, PbsJob.CLASS_FILE)):
            with open(os.path.join(pbs_process._output_dir, PbsJob.CLASS_FILE), "w") as writer:
                writer.write(pbs_process._jobs_dir + ";")
                writer.write(pbs_process._level_sim_file + ";")
                writer.write(str(pbs_process._debug) + ";")

    @staticmethod
    def _deserialize_pbs_process(output_dir):
        """
        Read CLASS_FILE written by _serialize_pbs_process and return stored parameters.

        :param output_dir: str path where CLASS_FILE was written
        :return: tuple (jobs_dir: str, level_sim_file: str, debug: bool)
        """
        with open(os.path.join(output_dir, PbsJob.CLASS_FILE), "r") as reader:
            line = reader.readline().split(';')
            return line[0], line[1], True if line[2] == 'True' else False

    @staticmethod
    def command_params():
        """
        Parse PBS worker command-line parameters. Called inside worker process.

        Expects sys.argv[1] = output_dir, sys.argv[2] = job_id

        :return: tuple (job_id: str, output_dir: str)
        """
        output_dir = sys.argv[1]
        job_id = sys.argv[2]

        return job_id, output_dir

    def _get_level_sim(self, level_id):
        """
        Deserialize LevelSimulation object for a given level id and store it in self._level_simulations.

        :param level_id: int or str identifier of level (used to format self._level_sim_file)
        :return: None (LevelSimulation object is stored internally)
        """
        with open(os.path.join(self._output_dir, self._level_sim_file.format(level_id)), "rb") as reader:
            l_sim = pickle.load(reader)
            self._level_simulations[l_sim._level_id] = l_sim

    def _get_level_id_sample_id_seed(self):
        """
        Read scheduled samples list for this job.

        The scheduled YAML file contains a list of tuples (level_id, sample_id, seed).

        :return: Sorted list of tuples [(level_id, sample_id, seed), ...] sorted by level_id ascending
        """
        with open(os.path.join(self._jobs_dir, PbsJob.SCHEDULED.format(self._job_id))) as file:
            level_id_sample_id_seed = yaml.load(file, yaml.Loader)

        level_id_sample_id_seed.sort(key=lambda tup: tup[0])
        return level_id_sample_id_seed

    def calculate_samples(self):
        """
        Main worker routine: calculate each scheduled sample, move produced files, and record success/failure.

        This method:
         - reads the scheduled list,
         - deserializes LevelSimulation objects on demand,
         - calls SamplingPool.calculate_sample for each scheduled sample,
         - moves successful/failed artifacts,
         - writes partial results to YAML files (successful, failed, times).

        :return: None
        """
        self._success_file = os.path.join(self._jobs_dir, PbsJob.SUCCESSFUL_RESULTS.format(self._job_id))
        self._failed_file = os.path.join(self._jobs_dir, PbsJob.FAILED_RESULTS.format(self._job_id))
        self._times_file = os.path.join(self._jobs_dir, PbsJob.TIME.format(self._job_id))

        # List of Tuple[level id, sample id, random seed]
        level_id_sample_id_seed = self._get_level_id_sample_id_seed()

        failed = []
        success = []
        current_level = 0
        current_samples = []
        start_time = time.time()
        successful_samples_time = 0
        times = []
        n_times = 0
        successful_dest_dir = os.path.join(self._output_dir, SamplingPool.SEVERAL_SUCCESSFUL_DIR)

        for level_id, sample_id, seed in level_id_sample_id_seed:
            start_time = time.time()
            # Deserialize level simulation config if not loaded
            if level_id not in self._level_simulations:
                self._get_level_sim(level_id)

            # When level changes, reset time accounting for previous level
            if current_level != level_id:
                times.append((current_level, successful_samples_time, n_times))
                n_times = 0
                start_time = time.time()
                successful_samples_time = 0
                current_level = level_id

            level_sim = self._level_simulations[current_level]
            assert level_sim._level_id == current_level

            # Calculate sample (may create sample working dir, call external tools)
            _, res, err_msg, _ = SamplingPool.calculate_sample(sample_id, level_sim, work_dir=self._output_dir, seed=seed)

            if not err_msg:
                success.append((current_level, sample_id, (res[0], res[1])))
                # Move successful artifacts unless in debug mode
                if not self._debug:
                    SamplingPool.move_successful_rm(sample_id, level_sim,
                                                    output_dir=self._output_dir,
                                                    dest_dir=SamplingPool.SEVERAL_SUCCESSFUL_DIR)
                n_times += 1
                successful_samples_time += (time.time() - start_time)
                print("sample time ", time.time() - start_time)
            else:
                failed.append((current_level, sample_id, err_msg))
                SamplingPool.move_failed_rm(sample_id, level_sim,
                                            output_dir=self._output_dir,
                                            dest_dir=SamplingPool.FAILED_DIR)

            current_samples.append(sample_id)
            times.append((current_level, successful_samples_time, n_times))
            self._save_to_file(success, failed, times, current_samples)

            # Reset accumulators for next loop iteration
            success = []
            failed = []
            current_samples = []
            times = []

        # Final flush (in case any accumulators still have items)
        self._save_to_file(success, failed, times, current_samples)

    def _save_to_file(self, success, failed, times, current_samples):
        """
        Append success/failure/time data to corresponding YAML result files.

        :param success: list of successful sample tuples
        :param failed: list of failed sample tuples
        :param times: list of (level_id, cumulative_time, n_samples) tuples
        :param current_samples: list of current sample ids processed
        :return: None
        """
        if success:
            self._append_file(success, self._success_file)
        if failed:
            self._append_file(failed, self._failed_file)
        if times:
            self._append_file(times, self._times_file)

    def save_sample_id_job_id(self, job_id, sample_ids):
        """
        Save mapping of sample ids to this job_id so other tools can query which job handled a sample.

        :param job_id: str
        :param sample_ids: iterable of sample-identifiers (each sample_id is usually a tuple or list, code expects sample_id[1])
        :return: None
        """
        sample_id_job_id_file = os.path.join(self._jobs_dir, PbsJob.SAMPLE_ID_JOB_ID)

        job_id_list = [job_id] * len(sample_ids)
        new_ids = dict(zip([sid[1] for sid in sample_ids], job_id_list))

        saved_ids = {}
        if os.path.exists(sample_id_job_id_file):
            with open(sample_id_job_id_file, "r") as file:
                saved_ids = json.load(file)

        with open(sample_id_job_id_file, "w") as file:
            saved_ids.update(new_ids)
            json.dump(saved_ids, file)

    @staticmethod
    def job_id_from_sample_id(sample_id, jobs_dir):
        """
        Lookup job id that processed a given sample id.

        :param sample_id: str sample identifier
        :param jobs_dir: path to jobs directory where SAMPLE_ID_JOB_ID file is stored
        :return: str job id associated with sample_id
        """
        sample_id_job_id_file = os.path.join(jobs_dir, PbsJob.SAMPLE_ID_JOB_ID)
        with open(sample_id_job_id_file, "r") as file:
            saved_ids = json.load(file)
        return saved_ids[sample_id]

    def _append_file(self, data, path):
        """
        Append `data` (serializable by YAML) to a file by opening in append mode and dumping.

        :param data: Python object serializable by ruamel.yaml (list, dict, etc.)
        :param path: Path to YAML file to append to
        :return: None
        """
        with open(path, "a") as f:
            yaml.dump(data, f)

    def _handle_sim_files(self, sample_id, level_sim):
        """
        If simulation requires workspace, switch to per-sample directory and copy common files there.

        :param sample_id: str
        :param level_sim: LevelSimulation instance
        :return: None
        """
        if level_sim.need_sample_workspace:
            SamplingPool.change_to_sample_directory(self._output_dir, sample_id)
            if level_sim.common_files is not None:
                SamplingPool.copy_sim_files(level_sim.common_files, os.getcwd())

    @staticmethod
    def read_results(job_id, jobs_dir):
        """
        Read and aggregate results produced by a PBS job into dictionaries.

        The function reads SUCCESSFUL_RESULTS, FAILED_RESULTS and TIME YAML files (if present)
        and returns aggregated dicts keyed by level_id.

        :param job_id: str
        :param jobs_dir: path to directory containing job result YAML files
        :return: tuple (successful_dict, failed_dict, time_dict) where:
                 - successful_dict[level_id] = [(sample_id, result), ...]
                 - failed_dict[level_id] = [(sample_id, error_message), ...]
                 - time_dict[level_id] = [(n_samples, cumulative_time), ...]
        """
        successful = {}
        failed = {}
        time = {}

        # Load successful results
        succ_path = os.path.join(jobs_dir, PbsJob.SUCCESSFUL_RESULTS.format(job_id))
        if os.path.exists(succ_path):
            with open(succ_path, "r") as reader:
                successful_samples = yaml.load(reader)
                for level_id, sample_id, result in successful_samples:
                    successful.setdefault(level_id, []).append((sample_id, result))

        # Load failed results
        failed_path = os.path.join(jobs_dir, PbsJob.FAILED_RESULTS.format(job_id))
        if os.path.exists(failed_path):
            with open(failed_path, "r") as reader:
                failed_samples = yaml.load(reader)
                for level_id, sample_id, err_msg in failed_samples:
                    failed.setdefault(level_id, []).append((sample_id, err_msg))

        # Load times
        times_path = os.path.join(jobs_dir, PbsJob.TIME.format(job_id))
        if os.path.exists(times_path):
            with open(times_path, "r") as reader:
                times = yaml.load(reader)
                for level_id, n_samples, t in times:
                    time.setdefault(level_id, []).append((n_samples, t))

        # Mark any scheduled-but-not-recorded samples as failed ("job failed")
        level_id_sample_id_seed = PbsJob.get_scheduled_sample_ids(job_id, jobs_dir)
        for level_id, sample_id, _ in level_id_sample_id_seed:
            successfull_ids = [s[0] for s in successful.get(level_id, [])]
            failed_ids = [f[0] for f in failed.get(level_id, [])]
            if sample_id not in failed_ids and sample_id not in successfull_ids:
                failed.setdefault(level_id, []).append((sample_id, "job failed"))

        return successful, failed, time

    @staticmethod
    def get_scheduled_sample_ids(job_id, jobs_dir):
        """
        Read the scheduled YAML file and return the list of scheduled (level_id, sample_id, seed) tuples.

        :param job_id: str
        :param jobs_dir: str
        :return: list of tuples (level_id, sample_id, seed)
        """
        with open(os.path.join(jobs_dir, PbsJob.SCHEDULED.format(job_id))) as file:
            level_id_sample_id_seed = yaml.load(file, yaml.Loader)

        return level_id_sample_id_seed

    def write_pbs_id(self, pbs_job_id):
        """
        Write an empty file whose filename encodes the mapping from our internal job id to the external PBS job id.

        :param pbs_job_id: str (external PBS job identifier)
        :return: None
        """
        file_name = os.path.join(self._jobs_dir, PbsJob.PBS_ID.format(self._job_id))
        file_name += pbs_job_id
        with open(file_name, 'w') as w:
            pass

    def save_scheduled(self, scheduled):
        """
        Store scheduled samples list into the jobs folder.

        :param scheduled: list of tuples (level_id, sample_id, seed) or similar structure
        :return: None
        """
        try:
            with open(os.path.join(self._jobs_dir, PbsJob.SCHEDULED.format(self._job_id)), "w") as file:
                yaml.dump(scheduled, file)
        except FileNotFoundError:
            print("Make sure you call _create_files method previously")

    @staticmethod
    def get_job_n_running(job_id, jobs_dir):
        """
        Return number of scheduled samples for a job (length of scheduled list file).

        :param job_id: str
        :param jobs_dir: str path to jobs directory
        :return: int count of scheduled entries
        """
        with open(os.path.join(jobs_dir, PbsJob.SCHEDULED.format(job_id))) as file:
            lines = yaml.load(file, yaml.Loader)
            return len(lines)


if __name__ == "__main__":
    pbs_process = PbsJob.create_process()
    pbs_process.calculate_samples()
