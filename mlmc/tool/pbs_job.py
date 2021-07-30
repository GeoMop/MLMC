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
        Class representing both pbs job in SamplingPoolPBS and true pbs process
        :param output_dir: output directory path
        :param jobs_dir: jobs directory path
        :param job_id: unique job id
        :param level_sim_file: file name of serialized LevelSimulation instance
        :param debug: bool, if True keep sample directories
        """
        self._output_dir = output_dir
        self._jobs_dir = jobs_dir
        self._job_id = job_id
        self._level_sim_file = level_sim_file
        self._debug = debug

        self._level_simulations = {}
        # LevelSimulations instances

    @classmethod
    def create_job(cls, output_dir, jobs_dir, job_id, level_sim_file, debug):
        """
        Create PbsProcess instance from SamplingPoolPBS
        :param output_dir: str
        :param jobs_dir: str
        :param job_id: str
        :param level_sim_file: str, file name format of LevelSimulation serialization
        :param debug: bool, if True keep sample directories
        :return: PbsProcess instance
        """
        pbs_process = cls(output_dir, jobs_dir, job_id, level_sim_file, debug)
        PbsJob._serialize_pbs_process(pbs_process)

        return pbs_process

    @classmethod
    def create_process(cls):
        """
        Create PbsProcess via PBS
        :return:
        """
        job_id, output_dir = PbsJob.command_params()
        jobs_dir, level_sim_file_format, debug = PbsJob._deserialize_pbs_process(output_dir)

        return cls(output_dir, jobs_dir, job_id, level_sim_file_format, debug)

    @staticmethod
    def _serialize_pbs_process(pbs_process):
        """
        Write down files necessary for pbs process call of this class - jobs_dir and format of file with serialized
                                                                        LevelSimulation
        :param pbs_process: PbsProcess instance
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
        Get jobs_dir and level_sim_file from serialized PbsProcess
        :param output_dir: str
        :return: jobs_dir, level_sim_file
        """
        with open(os.path.join(output_dir, PbsJob.CLASS_FILE), "r") as reader:
            line = reader.readline().split(';')
            return line[0], line[1], True if line[2] == 'True' else False

    @staticmethod
    def command_params():
        """
        Read command parameters - job identifier and file with necessary files
        :return: None
        """
        output_dir = sys.argv[1]
        job_id = sys.argv[2]

        return job_id, output_dir

    def _get_level_sim(self, level_id):
        """
        Deserialize LevelSimulation object
        :return: None
        """
        with open(os.path.join(self._output_dir, self._level_sim_file.format(level_id)), "rb") as reader:
            l_sim = pickle.load(reader)
            self._level_simulations[l_sim._level_id] = l_sim

    def _get_level_id_sample_id_seed(self):
        """
        Get scheduled samples
        :return: List[Tuple[level_id: int, sample_id: str, seed: int]] sorted by level_id ASC
        """
        with open(os.path.join(self._jobs_dir, PbsJob.SCHEDULED.format(self._job_id))) as file:
            level_id_sample_id_seed = yaml.load(file, yaml.Loader)

        level_id_sample_id_seed.sort(key=lambda tup: tup[0])
        return level_id_sample_id_seed

    def calculate_samples(self):
        """
        Calculate scheduled samples
        :return:
        """
        self._success_file = os.path.join(self._jobs_dir, PbsJob.SUCCESSFUL_RESULTS.format(self._job_id))
        self._failed_file = os.path.join(self._jobs_dir, PbsJob.FAILED_RESULTS.format(self._job_id))
        self._times_file = os.path.join(self._jobs_dir, PbsJob.TIME.format(self._job_id))

        # List of Tuple[level id, sample id, random seed]
        level_id_sample_id_seed = self._get_level_id_sample_id_seed()

        failed = []
        # Failed samples - Tuple(level_id, sample_id, error_msg)
        success = []
        # Successful samples - Tuple(level_id, sample_id, (fine result, coarse result))
        current_level = 0
        current_samples = []
        # Currently saved samples
        start_time = time.time()
        times = []
        # Sample calculation time - Tuple(level_id, [n samples, cumul time for n sample])
        n_times = 0
        successful_dest_dir = os.path.join(self._output_dir, SamplingPool.SEVERAL_SUCCESSFUL_DIR)
        for level_id, sample_id, seed in level_id_sample_id_seed:
            # Deserialize level simulation config
            if level_id not in self._level_simulations:
                self._get_level_sim(level_id)

            # Start measuring time
            if current_level != level_id:
                # Save previous level times
                times.append((current_level, time.time() - start_time, n_times))
                n_times = 0
                start_time = time.time()
                current_level = level_id

            level_sim = self._level_simulations[current_level]
            assert level_sim._level_id == current_level
            # Calculate sample
            _, res, err_msg, _ = SamplingPool.calculate_sample(sample_id, level_sim, work_dir=self._output_dir, seed=seed)

            if not err_msg:
                success.append((current_level, sample_id, (res[0], res[1])))
                # Increment number of successful samples for measured time
                if not self._debug:
                    SamplingPool.move_successful_rm(sample_id, level_sim,
                                                    output_dir=self._output_dir,
                                                    dest_dir=SamplingPool.SEVERAL_SUCCESSFUL_DIR)
            else:
                failed.append((current_level, sample_id, err_msg))
                SamplingPool.move_failed_rm(sample_id, level_sim,
                                            output_dir=self._output_dir,
                                            dest_dir=SamplingPool.FAILED_DIR)

            current_samples.append(sample_id)
            n_times += 1
            times.append((current_level, time.time() - start_time, n_times))
            self._save_to_file(success, failed, times, current_samples)

            success = []
            failed = []
            current_samples = []
            times = []

        self._save_to_file(success, failed, times, current_samples)

        # self._write_end_mark(self._success_file)
        # self._write_end_mark(self._failed_file)
        # self._write_end_mark(self._times_file)

    def _save_to_file(self, success, failed, times, current_samples):
        """
        Save sample results to files, create file which indicates that sample in stored
        :param success: dict
        :param failed: dict
        :param times: dict
        :param current_samples: list
        :return: None
        """
        # Write results to files
        if success:
            self._append_file(success, self._success_file)
        if failed:
            self._append_file(failed, self._failed_file)
        if times:
            self._append_file(times, self._times_file)

    # def _write_end_mark(self, path):
    #     """
    #     Write end mark to the file
    #     :param path: str, file path
    #     :return: None
    #     """
    #     if os.path.exists(path):
    #         with open(path, "a") as f:
    #             yaml.dump("end", f)

    def save_sample_id_job_id(self, job_id, sample_ids):
        """
        Store the sample ID associated with the job ID
        :param job_id: str
        :param sample_ids: list of str
        """
        sample_id_job_id_file = os.path.join(self._jobs_dir, PbsJob.SAMPLE_ID_JOB_ID)

        job_id = [job_id] * len(sample_ids)
        new_ids = dict(zip([sid[1] for sid in sample_ids], job_id))

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
        Get job ID for given sample ID
        :param sample_id: str
        :param jobs_dir: jobs directory with results
        :return: str, job id
        """
        sample_id_job_id_file = os.path.join(jobs_dir, PbsJob.SAMPLE_ID_JOB_ID)
        with open(sample_id_job_id_file, "r") as file:
            saved_ids = json.load(file)
        return saved_ids[sample_id]

    def _append_file(self, data, path):
        """
        Append result files, it works on read - update - write basis
        :param data: Data to append
        :param path: file path
        :return: None
        """
        with open(path, "a") as f:
            yaml.dump(data, f)

    def _handle_sim_files(self, sample_id, level_sim):
        """
        Change working directory to sample dir and copy common files
        :param sample_id: str
        :param level_sim: LevelSimulation
        :return: None
        """
        if level_sim.need_sample_workspace:
            SamplingPool.change_to_sample_directory(self._output_dir, sample_id)
            if level_sim.common_files is not None:
                SamplingPool.copy_sim_files(level_sim.common_files, os.getcwd())

    @staticmethod
    def read_results(job_id, jobs_dir):
        """
        Read result file for given job id
        :param job_id: str
        :param jobs_dir: path to jobs directory
        :return: successful: Dict[level_id, List[Tuple[sample_id:str, Tuple[ndarray, ndarray]]]]
                 failed: Dict[level_id, List[Tuple[sample_id: str, error message: str]]]
                 time: Dict[level_id: int, List[total time: float, number of success samples: int]]
        """
        successful = {}
        failed = {}
        time = {}

        # Save successful results
        if os.path.exists(os.path.join(jobs_dir, PbsJob.SUCCESSFUL_RESULTS.format(job_id))):
            with open(os.path.join(jobs_dir, PbsJob.SUCCESSFUL_RESULTS.format(job_id)), "r") as reader:
                successful_samples = yaml.load(reader)
                for level_id, sample_id, result in successful_samples:
                    successful.setdefault(level_id, []).append((sample_id, result))

        # Save failed results
        if os.path.exists(os.path.join(jobs_dir, PbsJob.FAILED_RESULTS.format(job_id))):
            with open(os.path.join(jobs_dir, PbsJob.FAILED_RESULTS.format(job_id)), "r") as reader:
                failed_samples = yaml.load(reader)
                for level_id, sample_id, err_msg in failed_samples:
                    failed.setdefault(level_id, []).append((sample_id, err_msg))

        # Save time
        if os.path.exists(os.path.join(jobs_dir, PbsJob.TIME.format(job_id))):
            with open(os.path.join(jobs_dir, PbsJob.TIME.format(job_id)), "r") as reader:
                times = yaml.load(reader)
                for level_id, n_samples, t in times:
                    time.setdefault(level_id, []).append((n_samples, t))

        # Deal with not finished (failed) samples in finished job
        level_id_sample_id_seed = PbsJob.get_scheduled_sample_ids(job_id, jobs_dir)

        for level_id, sample_id, _ in level_id_sample_id_seed:
            successfull_ids = [success[0] for success in successful.get(level_id, [])]
            failed_ids = [f[0] for f in failed.get(level_id, [])]
            if sample_id not in failed_ids and sample_id not in successfull_ids:
                failed.setdefault(level_id, []).append((sample_id, "job failed"))

        # if "end" in successful:
        #     del successful["end"]
        # if "end" in failed:
        #     del failed["end"]
        # if "end" in time:
        #     del time["end"]

        return successful, failed, time

    @staticmethod
    def get_scheduled_sample_ids(job_id, jobs_dir):
        """
        Get scheduled samples
        :param job_id: str
        :param jobs_dir: str
        :return:
        """
        with open(os.path.join(jobs_dir, PbsJob.SCHEDULED.format(job_id))) as file:
            level_id_sample_id_seed = yaml.load(file, yaml.Loader)

        return level_id_sample_id_seed

    def write_pbs_id(self, pbs_job_id):
        """
        Create empty file name contains pbs jobID and our jobID
        :param pbs_job_id: str
        :return: None
        """
        file_name = os.path.join(self._jobs_dir, PbsJob.PBS_ID.format(self._job_id))
        file_name += pbs_job_id
        with open(file_name, 'w') as w:
            pass

    def save_scheduled(self, scheduled):
        """
        Save scheduled samples to yaml file
        format: List[Tuple[level_id, sample_id]]
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
        Get number of running (scheduled) samples for given unfinished jobs
        :param job_id: str
        :param jobs_dir: str, path to jobs directory
        :return: int
        """
        with open(os.path.join(jobs_dir, PbsJob.SCHEDULED.format(job_id))) as file:
            lines = yaml.load(file, yaml.Loader)
            return len(lines)


if __name__ == "__main__":
    pbs_process = PbsJob.create_process()
    pbs_process.calculate_samples()
