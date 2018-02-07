import numpy as np
import scipy.linalg as la
from src.simulation import Simulation


class SimulationWater(Simulation):
    """
    Class for 'water' simulation 
    Inherits from Simulation
    """

    def __init__(self, step_length, volume, type_of_random_array):
        """
        :param step_length: length of step
        :param volume: volume
        """

        self.length_of_area = 10

        # vychozi objem
        self.V = volume;
        self.data = 0
        self.step_length = step_length
        self.volume = volume
        self.F = None

        super(SimulationWater, self).__init__(type_of_random_array)


    def count_h(self, n):
        self.n_sim_steps = n
        self.h = self.length_of_area / n
        self.time_step = self.h

    def cycle(self, n_fine):
        '''
        Execution of simulation 
        :param n_coarse: coarse n - number of steps
        :param F: pseudorandom array
        :param n_fine: fine n - number of steps
        :return: Last value of concentration
        '''

        if (None == self.F):
            self.random_array(self.n_sim_steps)

        if (self.n_sim_steps != n_fine):
            self.F = self.averaging(self.n_sim_steps, n_fine, self.F)

        prumer = np.average(self.F)

        self.count_h(self.n_sim_steps)

        for j in range(len(self.F)):
            self.F[j] = self.F[j] - prumer

        sum = 0
        for i in range(len(self.F)):
            sum += self.F[i]

        v = []
        v.append(2)
        for k in range(len(self.F) - 1):
            v.append(v[k] + self.F[k] * self.h)

        # Create array of concentration - prvni hodnota je jedna ostatni nula
        ct = np.zeros(len(v))
        ct[0] = 1

        matrix = self.getMatrix(v)
        # Loop through time
        for t in range(self.n_sim_steps):
            # count concentration in next time
            ct = (la.solve_banded((1, 0), matrix, ct))
        return ct[-1]

    def random_array(self):
        self.F = self.get_rnd_array(self.n_sim_steps)

    def get_random_array(self):
        return self.F

    def set_random_array(self, F):
        self.F = F

    def getMatrix(self, v):
        '''
        Count matrix
        :param v: array of speed
        :return: matrix
        '''

        a = []
        b = []
        alfa = self.time_step / self.h


        # first concentration is 1
        a.append(1)

        # Count values
        for i in range(1, len(v)):
            v_max = v[i] < v[i - 1] and v[i - 1] or v[i]

            # values on diagonal
            a.append(1 + alfa * v_max)

            # values under the diagonal
            b.append(-alfa * v[i - 1])

        # Add values zero to values under the diagonal
        b.append(0)

        return [a, b]
