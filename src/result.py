import numpy as np
import Moments


class Result():
    '''
    Multilevel Monte Carlo result
    '''
    def __init__(self):
        self.n = []
        self.variance = []
        self.N = []
        self.time = []
        self.average = []
        self.arrays = []
        self.data = []
        self.moments = []


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

    def result(self):
        N_final = []
        V_final = []

        mo = Moments.Moments(self.execution_number)
        mo.set_data(self.data)
        self.moments = mo.counting_moment(5)

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
        print("Centrální momenty", self.moments)







