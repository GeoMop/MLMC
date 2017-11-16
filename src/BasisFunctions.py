import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import Moments



class BasisFunctions():



    def get_moments(self, number):

        s = np.random.lognormal(0, 0.1, 10000000)
        moments = []
        for k in range (number):
            moments.append(scipy.stats.moment(s, k));

        return (moments, s)

        #print("moments", moments)


        '''
        count, bins, ignored = plt.hist(s, 100,
        normed=True, align='mid')

        x = np.linspace(min(bins), max(bins), 10000)
        pdf = (np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2)) / (x * sigma * np.sqrt(2 * np.pi)))



        plt.plot(x, pdf, linewidth=2, color='r')
        #plt.axis('tight')
        #plt.show()
        s = 1
        mean, var, skew, kurt = lognorm.stats(s, moments='mvsk')
        print("mean", mean)
        print("var", var)
        print("skew", skew)
        print("kurt", kurt)

        moments = [mean,var,skew, kurt]
        '''



