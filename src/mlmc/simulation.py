import numpy as np
import random as rn


class Simulation:
    """
    Parent class for simulations

    JS TODO:
    - this can not be the base class since the averaging methods are simulation dependent
    - Make SimulationBase class containing only methods necessary from MLMC, and clearly document
       methods that should be implemented by derived classes, arguments, return values and effect on the object's data.
    - Move averaging into tests
    """

    def __init__(self, r):
        # Type of random array
        self.type_of_random_array = r
        # Simulation result
        self._simulation_result = 0
        # Fine simulation step
        self._simulation_step = 0

    def cycle(self):
        pass

    @property
    def simulation_result(self):
        """
        Simulation result
        """
        return self._simulation_result

    @simulation_result.setter
    def simulation_result(self, result):
        self._simulation_result = result

    @property
    def n_sim_steps(self):
        """
        Simulation step
        """
        return self._simulation_step

    @n_sim_steps.setter
    def n_sim_steps(self, step):
        self._simulation_step = step

    def averaging(self, n_coarse, n_fine, F):
        """
        Array interpolation
        :param n_coarse:    Steps on previous level
        :param n_fine:      Steps on current level
        :param F:           Random array
        :return: array      G array
        """
        G = []
        for i in range(n_coarse):
            jbegin = np.floor((i / n_coarse) * n_fine).astype(int)
            jend = np.floor(((i + 1) / n_coarse) * n_fine).astype(int)
            v = 0
            v = v - ((i / n_coarse) - (jbegin / n_fine)) * F[jbegin]

            if jend < n_fine:
                for k in range(jbegin, jend + 1):
                    v = v + (1 / n_fine) * F[k]
            else:
                for k in range(jbegin, jend):
                    v = v + (1 / n_fine) * F[k]

            if jend < n_fine:
                v = v - (((jend + 1) / n_fine) - ((i + 1) / n_coarse)) * F[jend]
            v = v / (1 / n_coarse)

            G.append(v)
        return G

    def get_rnd_array(self, length):
        """
        :param length: length of array
        :return: array
        """
        if self.type_of_random_array == 1:
            return self.Z_array(length)
        if self.type_of_random_array == 2:
            return self.F_array(length)
        if self.type_of_random_array == 3:
            return self.normal(length)

    def normal(self, length):
        array = np.random.normal(2, 1)
        return array

    def Z_array(self, length):
        """
        :param length: int, length of array
        :return: array
        """
        Z = []
        for i in range(length):
            Z.append(0.2 * (2 * rn.random() - 1))
        return Z

    def F_array(self, length):
        """
        :param length: int length of array
        :return: array
        """
        # -1 for shooting simulation
        # 0 for water simulation
        F_average = -1
        F_deviation = 0.5
        F = [F_average] * 2
        F[0] = F[0] + (rn.random() - 0.5) * 2 * F_deviation
        F[1] = F[1] + (rn.random() - 0.5) * 2 * F_deviation
        scale = F_deviation
        fraction = 0.2  # 0-1; 0 means perfect correlation
        new_F = []
        if length > 2:
            while len(F) < length:
                new_F = []
                scale *= fraction
                for i in range(len(F) - 1):
                    shift = scale * 2 * (rn.random() - 0.5)
                    new_F.append(F[i])
                    new_F.append((F[i] + F[i + 1]) / 2 + shift)
                new_F.append(F[-1])
                F = new_F

            del new_F[length:]  # drop remaining items
        else:
            new_F = F
        return new_F

    def n_ops_estimate(self):
        return self.n_sim_steps
