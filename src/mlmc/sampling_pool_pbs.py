import os
import shutil

from src.mlmc.sampling_pool import SamplingPool


class SamplingPoolPBS(SamplingPool):

    def __init__(self, work_dir=None, job_weight=200000, job_count=0):
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
        # self.qsub_cmd = qsub
        self._pbs_config = None
        self._pbs_header_template = None

        self.need_workspace = True

        self._jobs_dir = None


    def schedule_sample(self, sample_id, level_sim):
        """
        Add sample to current PBS package
        :param sample_id: unique sample id from Sampler
        :param level_sim: LevelSimulation instance
        :return: None
        """
        print("schedule sample")
        print("sample id ", sample_id)
        print("level sim ", level_sim)

        level_sim.config_dict["sample_id"] = sample_id

        # result = self._pool.apply_async(level_sim.calculate, args=(level_sim.config_dict,),
        #                                 callback=self.result_callback, error_callback=self.error_callback)
        #
        # result.get()
        # print("result get ", result.get())
        # print("res.result ", res.successful())
        #return result

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


