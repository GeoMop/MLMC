class Simulation:
    """
    Parent class for simulations

    JS TODO:
    - this can not be the base class since the averaging methods are simulation dependent
    - Make SimulationBase class containing only methods necessary from MLMC, and clearly document
       methods that should be implemented by derived classes, arguments, return values and effect on the object's data.
    - Move averaging into tests
    """

    def __init__(self, sim_param=0):
        # Simulation result
        self._simulation_result = None
        # Fine simulation step
        self._simulation_step = 0
        self.sim_param = sim_param
        self._coarse_sample = 0
        self._input_sample = []

    def cycle(self):
        pass

    @property
    def simulation_result(self):
        """
        Simulation result
        :return: number, result of simulation
        """
        return self._simulation_result

    @simulation_result.setter
    def simulation_result(self, result):
        self._simulation_result = result

    @property
    def n_sim_steps(self):
        """
        Simulation step
        :return: int
        """
        return self._simulation_step

    @n_sim_steps.setter
    def n_sim_steps(self, step):
        self._simulation_step = step

    def n_ops_estimate(self):
        return self.n_sim_steps

    def random_array(self):
        pass

    def get_random_array(self):
        pass

    def set_coarse_sim(self, coarse_sim):
        pass
