import os
import time
import numpy as np
from typing import List

from sample_storage import SampleStorage
from sampling_pool import SamplingPool
from new_simulation import Simulation
from workspace import WithoutWorkspace, SimulationWorkspace, WholeWorkspace, PBSWorkspace


class Sampler:

    def __init__(self, sample_storage: SampleStorage, sampling_pool: SamplingPool, sim_factory: Simulation,
                 n_levels: int, step_range: List[int], work_dir=None):
        """

        :param sample_storage: store scheduled samples, results and result structure
        :param sampling_pool: calculate samples
        :param sim_factory: generate samples
        :param n_levels: number of MLMC levels
        :param step_range: simulation step range
        :param work_dir: working directory
        """
        self._sample_storage = sample_storage
        self._sampling_pool = sampling_pool

        self._n_levels = n_levels
        self._sim_factory = sim_factory
        self._step_range = step_range

        # Number of created samples
        self._n_created_samples = np.zeros(self._n_levels)

        # Number of target samples
        self.n_target_samples = np.zeros(self._n_levels)
        # Number of planned samples
        self.n_planned_samples = np.zeros(self._n_levels)
        # Number of finished and collected samples
        self.n_collected_samples = np.zeros(self._n_levels)

        # Set workspaces
        self._create_workspace_directories(work_dir)

        # Save simulation result specification
        self._sample_storage.save_workspace(self._workspace)

    def _create_workspace_directories(self, work_dir):
        """
        Sampler uses workspace object for communication with files system
        :param work_dir: abs path to working directory
        :return: None
        """
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
        self.n_target_samples += 2

    def _get_sample_tag(self, level_id):
        """
        Create sample tag
        :param level_id: identifier of current level
        :return: str
        """
        return "L{:02d}_S{:07d}".format(level_id, int(self._n_created_samples[level_id]))

    def create_simulations(self):
        """
        Create simulation samples, loop through "levels" and its samples (given the number of target samples):
            1) generate sample tag (same for fine and coarse simulation)
            2) get LevelSimulation instance by simulation factory
            3) schedule sample via sampling pool
            4) store scheduled samples in sample storage, separately for each level
        :return: None
        """

        for level_id, n_samples in enumerate(self.n_target_samples):

            samples = []
            for _ in range(int(n_samples)):
                sample_id = self._get_sample_tag(level_id)

                #  @TODO: workspace to sampling pool
                sample_dir = self._workspace.change_to_sample_directory(sample_id, level_id)

                #  @TODO: move to parent for loop??
                level_sim = self._sim_factory.level_instance([self._step_range[1]], [self._step_range[0]])

                # Copy simulation files, ...
                self._modify_level_sim_obj(level_sim, level_id, sample_dir)

                # Serialized SimulationLevel object
                self._workspace.serialize_level_sim(level_sim)

                # Schedule current sample
                self._sampling_pool.schedule_sample(sample_id, level_sim)

                # Increment number of created samples at current level
                self._n_created_samples[level_id] += 1

                samples.append(sample_id)

            # Store scheduled samples
            self._sample_storage.save_scheduled_samples(samples)

    def _modify_level_sim_obj(self, level_sim, level_id, sample_dir):
        """
        Copy simulation common files to sample directory
        :param level_sim: LevelSimulation instance
        :param level_id: level identifier
        :param sample_dir: sample directory
        :return: LevelSimulation
        """
        # Copy simulation common files
        if level_sim.common_files:
            self._workspace.copy_sim_files(level_sim.common_files, sample_dir)

            if level_sim.sample_workspace is not None:
                raise Exception("Sample workspace must be set in Sample method")
            level_sim.sample_workspace = sample_dir

        level_sim.level_id = level_id
        level_sim.calculate = self._sim_factory.calculate

        return level_sim

    def ask_simulations_for_samples(self, sleep=0, timeout=None):
        """
        Waiting for running simulations
        :param sleep: time for doing nothing
        :param timeout: maximum time for waiting on running simulations
        :return: int, number of running simulations
        """
        if timeout is None:
            timeout = 0
        elif timeout <= 0:
            return 1

        n_running = 1
        t0 = time.clock()
        while n_running > 0:
            finished_samples, n_running = self._sampling_pool.get_finished()

            # Store finished samples
            if len(finished_samples) > 0:
                self._store_finished_samples(finished_samples)

            time.sleep(sleep)
            if 0 < timeout < (time.clock() - t0):
                break

        return n_running

    def load_from_storage(self):
        raise NotImplementedError
        self._sample_storage.load_data()

    def _store_finished_samples(self, finished_samples):
        self._sample_storage.save_results(finished_samples)

    def sample_storage(self):
        return self._sample_storage
