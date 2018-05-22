from src.mlmc.simulation import Simulation
import random as rn
import numpy as np

class SimulationShooting(Simulation):
    """
    Class for 'shooting' simulation 
    Inherits from Simulation
    """

    def __init__(self, coord, v, extremes, time, F0, sim_param):
        """
        :param coord:       starting position
        :param v:           starting speed
        :param extremes:    borders of area
        :param time:        maximum time
        :param F0:          starting power
        """
        self.X = coord
        self.V = v
        self._input_sample = None
        self.extremes = extremes
        self.time = time
        self.sim_param = sim_param

        super(SimulationShooting, self).__init__()

    def cycle(self, sim_id):
        """
        Simulation of 2D shooting 
        :param sim_id:    simulation id

        """
        x, y, time, n = 0, 0, 0, 0
        X = self.X
        V = self.V

        # Time step
        if self.sim_param != 0:
            self.dt = 10 / self.sim_param

        # Loop through random array F
        for i in range(self.sim_param):

            # New coordinates
            X = X + self.dt * V

            # New vector of speed
            V = V + self.dt * self._input_sample[i]

            x = X[0]
            y = X[1]

            if x > self.extremes[1]:
                print("x is too big")
                break
            if x < self.extremes[0]:
                print("x is too small")
                break
            if y > self.extremes[3]:
                print("y is too big")
                break
            if y < self.extremes[2]:
                print("y is too small")
                break

            time = self.dt * (i + 1)

            # End simulation if time is bigger then maximum time
            if time >= self.time:
                break;

        # Set simulation data
        self.simulation_result = y
        return y

    def random_array(self):
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
        if self.sim_param > 2:
            while len(F) < self.sim_param:
                new_F = []
                scale *= fraction
                for i in range(len(F) - 1):
                    shift = scale * 2 * (rn.random() - 0.5)
                    new_F.append(F[i])
                    new_F.append((F[i] + F[i + 1]) / 2 + shift)
                new_F.append(F[-1])
                F = new_F

            del new_F[self.sim_param:]  # drop remaining items
        else:
            new_F = F
        self._input_sample = new_F

    def get_random_array(self):
        return self._coarse_sample

    def set_random_array(self, F):
        self._input_sample = F

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
        self._coarse_sample = G

    def set_coarse_sim(self, coarse_simulation):
        self.averaging(coarse_simulation.sim_param, self.sim_param, self._input_sample)
