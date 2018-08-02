import src.mlmc.simulation as simulation
import random as rn
import numpy as np


class SimulationShooting(simulation.Simulation):
    """
    Class for 'shooting' simulation 
    Inherits from Simulation
    """

    def __init__(self, step, config):
        """
        :param coord:       starting position
        :param v:           starting speed
        :param extremes:    borders of area
        :param time:        maximum time
        :param F0:          starting power
        """
        self.X = config['coord']
        self.V = config['speed']
        self._input_sample = None
        self.extremes = config['extremes']
        self.time = config['time']
        self._fields = config['fields']
        self.sim_param = int(2/step)
        self.step = step
        self._coarse_simulation = None
        self._result_dict = {}

    def simulation_sample(self, sim_id):
        """
        Simulation of 2D shooting
        :param sim_id:    simulation id
        """
        x, y, time, n = 0, 0, 0, 0
        X = self.X
        V = self.V

        # Time step
        if self.sim_param != 0:
            self.dt = self.time / self.sim_param

        # Loop through random array F
        for i in range(self.sim_param):
            # New coordinates
            X = X + self.dt * V

            # New vector of speed
            V = V + self.dt * self._input_sample[i]

            x = X[0]
            y = X[1]

            if x > self.extremes[1] or x < self.extremes[0] or y > self.extremes[3] or y < self.extremes[2]:
                y = np.nan
                break

            time = self.dt * (i + 1)

            # End simulation if time is bigger then maximum time
            if time >= self.time:
                break

        # Set simulation data
        self._result_dict[sim_id] = y

        return sim_id

    def generate_rnd_sample(self):
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
        return new_F

    def create_points(self):
        if self._coarse_simulation is None:
            self.points = np.empty((self.sim_param, 1))
            self.points[:, 0] = np.linspace(0, self.V[0]*self.time, self.sim_param) #np.arange(self.sim_param)#

        else:
            self.points = np.empty((self.sim_param + self._coarse_simulation.sim_param, 1))
            self.points[:, 0] = np.concatenate((np.linspace(0, self.V[0]*self.time, self.sim_param),
                                                   np.linspace(0, self.V[0]*self.time, self._coarse_simulation.sim_param)))

    def _make_fields(self):
        self.create_points()
        self._fields.set_points(self.points)

    def generate_random_sample(self):
        """
        Generate random field, both fine and coarse part.
        Store them separeted.
        :return:
        """
        # assert self._is_fine_sim
        self._make_fields()
        fields_sample = self._fields.sample()

        self._input_sample = fields_sample[:self.sim_param]
        #self._input_sample = self.generate_rnd_sample()
        if self._coarse_simulation is not None:
            self._coarse_simulation._input_sample = avg = fields_sample[self.sim_param:]
            #self._coarse_simulation._input_sample = avg = self.averaging(self.sim_param, self._coarse_simulation.sim_param, self._input_sample)

    def extract_result(self, sim_id):
        return self._result_dict[sim_id]

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

    def set_coarse_sim(self, coarse_simulation):
        """
        Set coarse simulations
        :param coarse_simulation: Simulation object
        :return: None
        """
        self._coarse_simulation = coarse_simulation
