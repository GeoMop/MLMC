import numpy as np
from moments import Moments


class Result():
    '''
    Multilevel Monte Carlo result
    '''
    def __init__(self, moments_number):
        self.n = []
        self.variance = []
        self.N = []
        self.time = []
        self.average = []
        self.arrays = []
        self.data = []
        self.data_for_distribution = []
        self.moments = []
        self.moments_number = moments_number


    def add_average(self, average):
        self.average.append(average)


    def add_arrays(self, arrays):
        self.arrays = arrays


    def set_variance(self, variance):
        self.variance = variance


    def set_levels(self, levels):
        self.levels = levels


    def set_levels_number(self, number):
        self.levels_number = number


    def set_execution_number(self, number):
        self.execution_number = number

    def append_result(self, result):
        self.n.append(result[0])
        self.variance.append(result[1])
        self.N.append(result[2])
        self.data.append(result[3])

    def add_time(self, time):
        self.time.append(time)

    def prepare_data(self):
        """
        Count result each item in each level, it means fine step result - coarse step result
        :return: 
        """
        levels =[]
        for d in self.data:
            for j in d:
                level = []
                for i in j:
                    level.append(i[0] - i[1])
                levels.append(level)
        self.data = levels


        # Sum of each mlmc level result
        all_data = self.data
        self.data_for_distribution = all_data[0]
        for level_data in all_data[1:]:
            for index, d in enumerate(level_data):
                self.data_for_distribution[index] = self.data_for_distribution[index] + d


    def result(self):
        N_final = []
        V_final = []

        mo = Moments(self.execution_number)
        self.prepare_data()
        mo.set_data(self.data)
        self.moments = mo.counting_moment(self.moments_number)

        for j in range(self.levels_number):
            n_final = self.n[0]

            sum_N = 0
            sum_V = 0
            for k in range(self.execution_number):
                sum_N = sum_N + self.N[k][j]
                sum_V = sum_V + self.variance[k][j]

            N_final.append(np.round(sum_N / self.execution_number).astype(int))
            V_final.append((sum_V / self.execution_number))

        cas_L = 0
        for i in range(self.execution_number):
            cas_L = cas_L + self.time[i]

        cas_final = (cas_L / self.execution_number)

        print("n = ", n_final)
        print("N = ", N_final)
        print("V = ", V_final)
        print("vektor Y", self.average)
        print("Y prumer = ", np.average(self.average))
        print("Rozptyl hodnot Y = ", np.var(self.average))
        print("Celkovy cas = ", cas_final)
        #print("Centrální momenty", self.moments)

        return self.moments, self.data_for_distribution







