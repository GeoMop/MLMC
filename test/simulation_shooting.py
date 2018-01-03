from src.simulation import Simulation


class SimulationShooting(Simulation):
    """
    Class for 'shooting' simulation 
    Inherits from Simulation
    """

    def __init__(self, coord, v, extremes, time, F0, type_of_random_array):
        """
        :param coord:       starting position
        :param v:           starting speed
        :param extremes:    borders of area
        :param time:        maximum time
        :param F0:          starting power
        """
        self.X = coord
        self.V = v
        self.F = None
        self.extremes = extremes
        self.time = time

        super(SimulationShooting, self).__init__(type_of_random_array)

    def cycle(self, n_fine):
        """
        Simulation of 2D shooting 
        :param n_coarse:    number of steps in previous level
        :param F:           pseudorandom array
        :param n_fine:      number of steps in this level
        :return: flaot      end y coordinate
        """
        x, y, time, n = 0, 0, 0, 0
        X = self.X
        V = self.V

        if self.F is None:
            self.random_array()

        if self.simulation_step != n_fine:
            self.F = self.averaging(self.simulation_step, n_fine, self.F)

        # Time step
        self.dt = 10 / len(self.F)

        # Loop through random array F
        for i in range(len(self.F)):

            # New coordinates
            X = X + self.dt * V

            # New vector of speed
            V = V + self.dt * self.F[i]

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
        self.F = self.get_rnd_array(self.simulation_step)

    def get_random_array(self):
        return self.F

    def set_random_array(self, F):
        self.F = F
