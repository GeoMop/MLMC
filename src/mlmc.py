import numpy as np
import Level as Level


class MLMC:

    def __init__(self, L, n_fine, n_coarse, sim):
        '''
        :param L:           Number of levels
        :param n_fine:      Maximum steps of one simulation
        :param n_coarse:    Minimum steps of one simulation
        :param sim:         Instance of object Simulation
        '''

        self.simulation = sim;

        # Initialization of variables
        self.levels = []
        self.target_time = None;
        self.target_variance = None

        self.number_of_levels = L

        self.n_fine = n_fine
        self.n_coarse = n_coarse

        # Random variable
        self.Y = []


    def monte_carlo(self, type, value):
        '''
        Implements complete multilevel Monte Carlo Method
        Calls other necessary methods
        :param type: int        1 for fix variance and other for fix time
        :param value: float     value of variance or time according to type
        '''

        self.i = 0

        if type == 1:
            self.target_variance = value
        else:
            self.target_time = value

        if 1 < self.number_of_levels:
            for k in range(self.number_of_levels):
                self.create_level()
        else:
            l = Level.Level([0, self.n_fine], self.simulation)
            self.levels.append(l)

        # Count new number of simulations according to variance of time
        if self.target_variance is not None:
            N = self.count_N_variance()
        elif self.target_time is not None:
            N = self.count_N_time()

        self.count_new_N(N)

        self.result()

        self.formatting_result()


    def result(self):
        '''
        Counts result from all levels
        '''
        C = 0
        for k in range(self.number_of_levels):
            Level = self.levels[k]

            C = C + Level.get_average()

        self.Y = C


    def get_arrays(self):
        arrays = []
        for k in range(self.number_of_levels):
            Level = self.levels[k]
            arrays.append(Level.get_Y())

        return arrays

    def get_data(self):
        data = []
        for k in range(self.number_of_levels):
            Level = self.levels[k]
            data.append(Level.get_data())

        return data

    def count_new_N(self, N):
        '''
        For each level counts further number of simulations by appropriate N
        :param N: array      new N for each level
        '''

        for k in range(self.number_of_levels):
            Level = self.levels[k]

            if Level.get_number_of_sim() < N[k]:
                Level.set_number_of_sim(N[k] - Level.get_number_of_sim())

                # Launch further simulations
                Level.level()

                Level.set_number_of_sim(N[k])



    def create_level(self):
        '''
        Create new level add its to the array of levels
        Call method for counting number of simulation steps
        Pass instance of Simulation to Level
        '''

        l = Level.Level(self.count_small_n(), self.simulation)

        self.levels.append(l)




    def count_small_n(self):
        '''
        Count number of steps for level
        :return: array  [n_coarse, n_fine]
        '''

        n2 = np.power(np.power((self.n_fine/self.n_coarse),(1/(self.number_of_levels-1))),self.i)*self.n_coarse


        # Coarse number of simulation steps from previouse level
        if 0 < len(self.levels):
            Level = self.levels[self.i - 1]
            n1 = Level.n_ops_estimate()

        else:
            n1 = 0

        self.i = self.i + 1

        return [np.round(n1).astype(int), np.round(n2).astype(int)]


    def count_N_time(self):
        '''
        For each level counts new N according to target_time
        :return: array
        '''
        N = []
        sum = self.count_sum()

        # Loop through levels
        # Count new number of simulations for each level
        for i in range(self.number_of_levels):
            Level = self.levels[i]
            # print(self.target_time)
            # print(Level.get_variance_Y())
            # print(Level.n_ops_estimate())

            vysledek = np.round((self.target_time * np.sqrt(Level.get_variance_Y() / Level.n_ops_estimate())) / sum).astype(int)

            N.append(vysledek)

        return N


    def count_N_variance(self):
        '''
        For each level counts new N according to target_variance
        :return: array
        '''
        N = []
        sum = self.count_sum()

        # Loop through levels
        # Count new number of simulations for each level
        for i in range(self.number_of_levels):
            Level = self.levels[i]
            vysledek = np.round((sum * np.sqrt(Level.get_variance_Y() / Level.n_ops_estimate())) / self.target_variance).astype(int)
            N.append(vysledek)

        return N


    def count_sum(self):
        '''
        Loop through levels and count sum of varinace * simulation n
        :return: float  sum
        '''
        sum = 0
        for j in range(len(self.levels)):
            Level = self.levels[j]
            sum = sum + np.sqrt(Level.get_variance_Y() * Level.n_ops_estimate())

        return sum



    def get_level_data(self):
        '''
        Get level data
        :return: array
        '''
        data = []
        for i in range(len(self.levels)):
            Level = self.levels[i]
            data.append(Level.get_data())

        return data


    def formatting_result(self):
        '''
        Formatting calculated data
        :return: 
        '''
        n = []
        V = []
        N= []
        data = []

        for j in range(len(self.levels)):
            Level = self.levels[j]
            n.append(Level.n_ops_estimate())

            data.append(Level.get_data())
            #print("DATA", Level.get_data())


            V.append(Level.get_variance_Y())
            N.append(Level.get_number_of_sim())

        return [n, V, N, data]


    def get_Y(self):
        '''
        Return result
        :return: self.Y
        '''
        return self.Y


    def average(self, array):
        '''
        :param array: array
        :return: average from array
        '''
        return np.mean(array)

    def dispersion(self, array):
        '''
        :param array: array
        :return: variance of array
        '''
        return np.var(array)


