import numpy as np
import scipy.stats

class Moments():

    def __init__(self, execution_number = 0):
        self.data = []
        self.execution_number = execution_number

    def set_data(self, data):
        ''' 
        :param data: array of result of simulation on each level, there are coarse and fine results
        :return: 
        '''
        self.data = data
       # self.prepare_data()

    def prepare_data(self):
        levels =[]
        for d in self.data:
            for j in d:
                level = []
                for i in j:
                    level.append(i[0] - i[1])
                levels.append(level)

        self.data = levels

    def get_data(self):
        return self.data

    def counting_moment(self, degrees):
        '''
        Calculating average moment from all execution of MLMC
        :param degree: degree of moment
        :return: array of moments
        '''

        all_moments = []
        for degree in range(degrees-1):
            degree += 1
            # Moment in each level and in each execution
            level_moments = []
            for d in self.data:
                d = np.array(d)
                level_moments.append(scipy.stats.moment(d, degree))

            # Separate moments to each execution
            execution_moments = np.split(np.array(level_moments), self.execution_number)

            print("execution_moments", execution_moments)

            # Average moment on each level from all execution
            moments = []
            for k in range(len(execution_moments[0])):
                moments.append(np.mean([item[k] for item in execution_moments]))

            all_moments.append(moments)

        result_moments = []
        for moment in all_moments:
            result_moments.append(sum(moment))

        return result_moments


