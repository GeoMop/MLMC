import random as rn
import numpy as np


class Level:
    '''
    Call Simulation methods
    There are informations about random variable - average, dispersion, number of simulation, ...
    '''

    def __init__(self, simulation_size, sim):
        '''
        :param simulation_size: number of simulation steps 
        :param sim: instance of object Simulation
        '''

        # Number of simulation steps in previous level
        self.n_coarse = simulation_size[0]

        # Number of simulation steps in this level
        self.n_fine = simulation_size[1]

        # Instance of object Simulation
        self.fine_simulation = sim.make_simulation()
        self.fine_simulation.set_step(self.n_fine)
        self.coarse_simulation = sim.make_simulation()
        self.coarse_simulation.set_step(self.n_coarse)

        # Initialization of variables
        self.data = []
        self.variance = 0;

        # Random variable
        self.Y = []

        # Default number of simulations is 10
        # that is enough for estimate variance
        self.number_of_simulations = 10


        self.Y = self.level()

       # print('level')

        self.set_variance_Y(self.dispersion(self.Y))

      #  print('set variance')



    def set_data(self, data):
        '''
        :param data:   simulation data
        '''
        self.data.append(data)


    def get_data(self):
        ''' 
        :return:array   simulation data
        '''
        return self.data


    def get_Y(self):
        '''    
        :return:array   self.Y
        '''
        return self.Y


    def get_number_of_sim(self):
        '''
        :return: int    number of simulations
        '''
        return self.number_of_simulations

    def set_number_of_sim(self, N):
        '''
        Set number of simulations
        :param N:
        '''
        self.number_of_simulations = N


    def get_average(self):
        ''' 
        :return: average of array of 
        '''
        return self.average(self.Y)


    def set_variance_Y(self, var):
        '''
        Set variance of self.Y
        :param var: variance
        '''
        self.variance = var


    def get_variance_Y(self):
        '''
        :return: float     variance of self.Y
        '''
        return self.variance


    def n_ops_estimate(self):
        '''
        :return: fine simulation n
        '''
        return self.fine_simulation.get_step()


    def level(self):
        '''
        Implements level of MLMC 
        Call Simulation methods
        Set simulation data
        :return: array      self.Y
        '''

        Y = []
        for i in range(self.number_of_simulations):
            self.fine_simulation.random_array()
            fine_step_result = self.fine_simulation.cycle(self.n_fine)
            self.coarse_simulation.set_random_array(self.fine_simulation.get_random_array())

            if(self.n_coarse != 0):

                coarse_step_result = self.coarse_simulation.cycle(self.n_fine)

                Y.append(coarse_step_result)
                self.Y.append(fine_step_result - coarse_step_result)
            else:
                self.Y.append(fine_step_result)

            # Save simulation data
            self.set_data([self.fine_simulation.get_data(), self.coarse_simulation.get_data()])

        #print(self.Y)
        return self.Y


    def average(self, array):
        '''
        :param array:    input array
        :return: float   average of array
        '''
        return np.mean(array)


    def dispersion(self, array):
        '''
        :param array:   input array
        :return:float   variance of array
        '''
        return np.var(array)
