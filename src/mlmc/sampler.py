import os
import shutil
import numpy as np
from typing import List

from sample_storage import SampleStorage
from sampling_pool import SamplingPool
from new_simulation import Simulation
from handle_workspace import WithoutWorkspace, SimulationWorkspace, WholeWorkspace, PBSWorkspace


class Sampler:

    def __init__(self, sample_storage: SampleStorage, sampling_pool: SamplingPool, sim_factory: Simulation,
                 n_levels: int, step_range: List[int], work_dir=None):

        self._sample_storage = sample_storage
        self._sampling_pool = sampling_pool

        self._n_levels = n_levels
        self._sim_factory = sim_factory
        self._step_range = step_range

        self._n_created_samples = np.zeros(self._n_levels)

        self.n_target_samples = np.zeros(self._n_levels)
        self.n_planned_samples = np.zeros(self._n_levels)
        self.n_collected_samples = np.zeros(self._n_levels)

        self._finished_samples = []
        self._sample_ids = []

        # Set workspaces
        self._create_workspace_directories(work_dir)

    def _create_workspace_directories(self, work_dir):
        try:
            self._workspace = WithoutWorkspace()

            if work_dir:
                self._workspace = WholeWorkspace(work_dir, self._n_levels)

            try:
                if self._sampling_pool.need_jobs:
                    self._workspace = PBSWorkspace(work_dir, self._n_levels)
                    self._sampling_pool.workspace = self._workspace
            except AttributeError:
                print("PBS sampling pool must have need_jobs attribute")

            # if self._sim_factory.need_workspace:
            #     self._workspace = SimulationWorkspace(work_dir)
        except AttributeError:
            print("Attribute error, need_workspace class attribute is obligatory")

    def determine_level_n_samples(self):
        """
        Calculate number of target samples on each level
        :return:
        """
        self.n_target_samples += 5

    def _get_sample_tag(self, level_id):
        """
        Create sample tag
        :param char: 'C' or 'F' depending on the type of simulation
        :param level_id: identifier of current level
        :return: str
        """
        return "L{:02d}_S{:07d}".format(level_id, int(self._n_created_samples[level_id]))

    def create_simulations(self):
        print("self n target samples ", self.n_target_samples)

        for level_id, n_samples in enumerate(self.n_target_samples):
            for _ in range(int(n_samples)):

                sample_id = self._get_sample_tag(level_id)
                print("sample id ", sample_id)

                sample_dir = self._workspace.change_to_sample_directory(sample_id, level_id)

                #  @TODO: move to parent for loop??
                level_sim = self._sim_factory.level_instance([self._step_range[1]], [self._step_range[0]])

                self._modify_level_sim_obj(level_sim, level_id, sample_dir)

                self._workspace.serialize_level_sim(level_sim)

                self._sampling_pool.schedule_sample(sample_id, level_sim)

                # Increment number of created samples at current level
                self._n_created_samples[level_id] += 1

    def _modify_level_sim_obj(self, level_sim, level_id, sample_dir):
        # Copy simulation common files
        if level_sim.common_files:
            self._workspace.copy_sim_files(level_sim.common_files, sample_dir)

            if level_sim.sample_workspace is not None:
                raise Exception("Sample workspace must be set in Sample method")
            level_sim.sample_workspace = os.getcwd()

        print("level_sim.sample_workspace ", level_sim.sample_workspace)

        level_sim.level_id = level_id
        level_sim.calculate = self._sim_factory.calculate

        return level_sim

    def target_var_adding_samples(self, target_var, moments_fn, pbs=None, sleep=20, add_coef=0.1):
        """
        Set level target number of samples according to improving estimates.
        We assume set_initial_n_samples method was called before.
        :param target_var: float, whole mlmc target variance
        :param moments_fn: Object providing calculating moments
        :param pbs: Pbs script generator object
        :param sleep: Sample waiting time
        :param add_coef: Coefficient for adding samples
        :return: None
        """
        # New estimation according to already finished samples
        n_estimated = self.estimate_n_samples_for_target_variance(target_var, moments_fn)
        # Loop until number of estimated samples is greater than the number of scheduled samples
        while not self.mlmc.process_adding_samples(n_estimated, pbs, sleep, add_coef):
            # New estimation according to already finished samples
            n_estimated = self.estimate_n_samples_for_target_variance(target_var, moments_fn)

    def estimate_n_samples_for_target_variance(self, target_variance, moments_fn=None, prescribe_vars=None):
        """
        Estimate optimal number of samples for individual levels that should provide a target variance of
        resulting moment estimate. Number of samples are directly set to levels.
        This also set given moment functions to be used for further estimates if not specified otherwise.
        TODO: separate target_variance per moment
        :param target_variance: Constrain to achieve this variance.
        :param moments_fn: moment evaluation functions
        :param prescribe_vars: vars[ L, M] for all levels L and moments M safe the (zeroth) constant moment with zero variance.
        :return: np.array with number of optimal samples for individual levels and moments, array (LxR)
        """
        _, n_samples_estimate_safe = self.n_sample_estimate_moments(target_variance, moments_fn, prescribe_vars)
        n_samples = np.max(n_samples_estimate_safe, axis=1).astype(int)

        return n_samples

    def n_sample_estimate_moments(self, target_variance, moments_fn=None, prescribe_vars=None):
        if moments_fn is None:
            moments_fn = self.moments
        if prescribe_vars is None:
            vars = self.estimate_diff_vars_regression(moments_fn)
        else:
            vars = prescribe_vars

        n_ops = np.array([lvl.n_ops_estimate for lvl in self.levels])

        sqrt_var_n = np.sqrt(vars.T * n_ops)  # moments in rows, levels in cols
        total = np.sum(sqrt_var_n, axis=1)  # sum over levels
        n_samples_estimate = np.round((sqrt_var_n / n_ops).T * total / target_variance).astype(int)  # moments in cols

        # Limit maximal number of samples per level
        n_samples_estimate_safe = np.maximum(np.minimum(n_samples_estimate, vars * self.n_levels / target_variance), 2)

        return n_samples_estimate, n_samples_estimate_safe

    def ask_simulations_for_samples(self):
        """
        Communicates through Sample objects

        1) Loop through list of samples (instance of Sample object), check result file, extract results, change status,
           pass data to SampleStorage

        2) Given directory, go through sample directories and deserialized Sample instances from JSON objects.

        :return:
        """
        finished_samples = self._sampling_pool.get_finished()
        print("finished samples ", finished_samples)
        self._finished_samples.append(finished_samples)

    def load_from_storage(self):
        raise NotImplementedError
        self._sample_storage.load_data()

    def _store_finished_results(self):
        self._sample_storage.write_data()
