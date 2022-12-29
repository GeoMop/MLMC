import os
import copy
import shutil
import numpy as np
from typing import *

import mlmc.random.correlated_field as cf
from typing import List
from mlmc.sim.simulation import Simulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.level_simulation import LevelSimulation


class FullScaleTransportSim(Simulation):

    def __init__(self, config):
        """
        :param config: Dict, simulation configuration
        """
        #super().__init__()
        self._config = config

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Called from mlmc.Sampler, it creates single instance of LevelSimulation (mlmc.level_simulation)
        :param fine_level_params: fine simulation step at particular level
        :param coarse_level_params: coarse simulation step at particular level
        :return: mlmc.LevelSimulation object
        """
        config = copy.deepcopy(self._config)
        # Set sample specific parameters
        # config["fine"] = {}
        # config["coarse"] = {}
        # config["fine"]["n_steps"] = fine_level_params[0]
        # config["coarse"]["n_steps"] = coarse_level_params[0]
        # config["res_format"] = self.result_format()

        return LevelSimulation(config_dict=config,
                               calculate=FullScaleTransportSim.calculate,
                               task_size=config["mesh_steps"][fine_level_params[0]],  # @TODO: set size
                               need_sample_workspace=True)

    @staticmethod
    def calculate(config, seed):
        """
        Calculate fine and coarse sample and also extract their results
        :param config: general configuration
        :param seed: random number generator seed
        :return: np.ndarray, np.ndarray
        """

        from endorse.fullscale_transport import fullscale_transport

        from endorse import common
        from endorse.common import dotdict, memoize, File, call_flow, workdir, report
        from endorse.mesh_class import Mesh

        ###################
        ### fine sample ###
        ###################

        #conf_file = os.path.join(config["work_dir"], "test_data/config_homo_tsx.yaml")
        #cfg = common.load_config(conf_file)
        #cfg.flow_env["flow_executable"] = config["flow_executable"]
        #cfg["work_dir"] = config["work_dir"]

        val = fullscale_transport(config['main_cfg_file'], config['source_params'], seed)
        q10 = list(val)
        add_values = (10 - len(q10)) * [0.0]
        q10.extend(add_values) #fixed_indicators[:len(ind_time_max)] = np.array(ind_time_max)
        res_fine = np.asarray(q10)
        #fine_res = fo.hydro

        #####################
        ### coarse sample ###
        #####################
        res_coarse = np.zeros_like(res_fine)

        return res_fine, res_coarse

    def result_format(self) -> List[QuantitySpec]:
        """
        Result format
        :return:
        """
        spec1 = QuantitySpec(name="indicator_conc", unit="g/m3", shape=(10, 1), times=[1], locations=['0'])
        # spec2 = QuantitySpec(name="width", unit="mm", shape=(2, 1), times=[1, 2, 3], locations=['30', '40'])
        return [spec1]
