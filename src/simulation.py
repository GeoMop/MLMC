import numpy as np
import random as rn


class Simulation():
    '''
    Parent class for simulations
    '''

    def __init__(self, r):
        self.type_of_random_array = r
        pass

    def cycle(self):
        pass

    def set_data(self, data):
        pass

    def get_data(self):
        pass


    def averaging(self, n_coarse, n_fine, F):
        '''
         Array interpolation 
        :param n_coarse:    Steps on previous level
        :param n_fine:      Steps on current level
        :param F:           Random array
        :return: array      G array 
        '''

        G = []
        for i in range(n_coarse):
            jbegin = np.floor((i / n_coarse) * n_fine).astype(int)
            jend = np.floor(((i + 1) / n_coarse) * n_fine).astype(int)

            v = 0
            v = v - ((i / n_coarse) - (jbegin / n_fine)) * F[jbegin]

            if (jend < n_fine):

                for k in range(jbegin, jend + 1):
                    v = v + (1 / n_fine) * F[k]

            else:
                for k in range(jbegin, jend):
                    v = v + (1 / n_fine) * F[k]

            if (jend < n_fine):
                v = v - (((jend + 1) / n_fine) - ((i + 1) / n_coarse)) * F[jend]
            v = v / (1 / n_coarse)

            G.append(v)
        #print('averaging')
        return G

    def get_rnd_array(self, n):

        if(1 == self.type_of_random_array):
            return self.Z_array(n)
        if(2 == self.type_of_random_array):
            return self.F_array(n)


    def Z_array(self, n):

        Z = []
        for i in range(n):
            Z.append(0.2 * (2 * rn.random() - 1))


        return Z


    def F_array(self, n):

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
        if (n > 2):
            while len(F) < n:
                new_F = []
                scale *= fraction
                for i in range(len(F) - 1):
                    shift = scale * 2 * (rn.random() - 0.5)
                    new_F.append(F[i])
                    new_F.append((F[i] + F[i + 1]) / 2 + shift)
                new_F.append(F[-1])
                F = new_F

            del new_F[n:]  # drop remaining items
        else:
            new_F = F

        #print("F", np.mean(new_F))
        return new_F