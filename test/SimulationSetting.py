import numpy as np
import SimulationWater as SimW
import SimulationShooting as SimS

class SimulationSetting():
    '''
    Simulation and its setting 
    '''

    def __init__(self, type):
        self.type = type
        self.sim = None
        pass

    def make_simulation(self):


        if self.type == 1:
            self.sim = SimS.SimulationShooting(np.array([0, 0]), np.array([10, 0]), np.array([-100, 200, -300, 400]), 10,
                                          np.array([0, -1]), 2)
        else:
            # Last argument is type of random array
            self.sim = SimW.SimulationWater(10,100, 2)

        return self.sim



    def set_data(self, data):
        pass

    def get_data(self):
        pass

