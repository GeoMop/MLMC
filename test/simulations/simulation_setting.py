import numpy as np
from test.simulation_water import SimulationWater as sim_water
from test.simulation_shooting import SimulationShooting as sim_shooting
from test.simulation_test import SimulationTest as sim_test


class SimulationSetting:
    """
    Simulation and its setting 
    """

    def __init__(self, simulation_type):
        # Type of simulation
        self.type = simulation_type
        # Simulation object
        self.sim = None

    def make_simulation(self):
        """
        Creating specific simulation
        :return: 
        """
        if self.type == 2:
            self.sim = sim_test()
            return self.sim

        if self.type == 1:
            self.sim = sim_shooting(np.array([0, 0]), np.array([10, 0]), np.array([-100, 200, -300, 400]), 10,
                                          np.array([0, -1]), 2)
        else:
            # Last argument is type of random array
            self.sim = sim_water(10, 100, 2)

        return self.sim
