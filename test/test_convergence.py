# TESTS, selected
import numpy as np
import numpy.linalg as la
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import importlib
import os
import sys
#libdir = os.path.join(os.path.split(
#         os.path.dirname(os.path.realpath(__file__)))[0],"C:\\Users\\Clara\\Documents\\Intec\\MLMC_Python\\src\\mlmc")
#sys.path.insert(1,libdir)

#import correlated_field
from mlmc.correlated_field import SpatialCorrelatedField
import texttable as tt
import scipy.interpolate as sc_inter
import scipy.stats as sc_stat

#import importlib
#importlib.import_module('correlated_field')

def make_points_grid(bound, size):
    """
    Creates a regular grid within given bounds and number of cells
    bound: array of size 3 with spatial max dimensions of the grid, initial point at [0,0,0] by default
    size   : array of size 3 with number of cells in each dimensions
    """

    assert len(bound) == len(size)
    dim = len(bound)
    grids = []
    for d in range(dim):
        grid = np.linspace(0, bound[d], num = size[d])
        grids.append(grid)

    grids = [ g.ravel() for g in np.meshgrid(*grids)]
    return np.stack(grids, axis=1)


def make_points_random(bound, size):
    """
    The same as above, but with a random placing of points
    bound : max coordinate in each direction
    size  : number of random points
    """
    dim = len(bound)
    grids = []
    for d in range(dim):
        grid = np.random.uniform(high=bound[0], size=size)
        grids.append(grid)
    return np.stack(grids, axis=1)
    
class Cumul:
    """
    Auxiliary class for evaluation convergence of a MC approximation of EX for
    a random field of given shape.
    """
    def __init__(self, shape):
        self.cumul = np.zeros(shape)
        self.n_iter = 0
        self.log_cum=[]
        self.log_limit = 16

    def __iadd__(self, other):
        self.cumul += other
        self.n_iter += 1
        if self.n_iter > self.log_limit:
            self.log_cum.append( (self.n_iter, self.cumul.copy()))
            self.log_limit *= 2
        return self

    def finalize(self):
        self.log_cum.append((self.n_iter, self.cumul))

    def avg_array(self):
        return np.array([cumul/n for n, cumul in self.log_cum])

    def n_array(self):
        return np.array([n for n,cumul in self.log_cum])    
        
class TestSpatialCorrelatedField:

    def plot_grid_field_2d(self, ncels, sample, title):
        # imgshow plot X axis verticaly, need to swap
        grid = sample.reshape( (ncels[1], ncels[0]) )
        imgplot = plt.imshow(grid)
        plt.colorbar()
        plt.title(title,size = 8)
        plt.show()

    def plot_mc(self, n_samples, data, title):
        """
        :param n_samples: np.array with number of samples for L different sizes
        :param data: LxN array, we make scatter + box plot from N size vectors on individual stages
        :return:
        """
        means = np.mean(data, axis=1)
        devs = np.std(data, axis=1)
        plt.errorbar(n_samples, means, yerr=devs, fmt='-o', capsize=3, ecolor="lightblue")
        m1, m0 = np.polyfit(np.log(n_samples), np.log(means), 1)

        legend = "rate: {}".format(m1)
        plt.plot(n_samples, np.exp(m1 * np.log(n_samples) + m0), '--k', label=legend)


        plt.xscale('log',size = 10)
        plt.yscale('log',size = 10)
        plt.legend()
        plt.title(title, size = 8)
        plt.xlabel("log_2 N", size = 10)
        plt.show()


    def perform_test_on_point_set(self, ncells, points, n_terms_range,typecorr):
        """
        A full covariance matrix for all the grid points and another one for only a
        subset of grid points is created (N fixed). A full SVD decomposiiton and  a truncated
        one are performed to derive L (full C) and Lr (reduced C) matrix.
        For the same subset of points multiple realizations are produced with the same
        field characteristics (mu, sigma, corr).
        From the full field only the subset point values are extracted and statistically
        compared against the reduced Cr, Lr realizations.

        :param points: Point set
        :return:
        """
        n = len(points)
        corr_length = 14.6
        mu = 6.14
        sigma = 2.3
        field = SpatialCorrelatedField(typecorr, dim=points.shape[1], corr_length = corr_length, aniso_correlation = None,)
        field.set_points(points, mu, sigma)
        L,ev = field.svd_dcmp(precision=0.01, n_terms_range=n_terms_range)

        # Estimate statistcs by Monte Carlo
        n_samples = 2400

        cum_mean = Cumul(n)
        cum_sigma = Cumul(n)

        # =========== Testing ===========================================================
        for j in range(n_samples):
            sample = field.sample()
            cum_mean += sample
            centered = sample - mu
            cum_sigma += centered * centered
           
        # ========== Mean, check convergence ========================================
        mu_err = np.abs(cum_mean.avg_array() - mu)
        means  = np.mean(mu_err, axis=1)
        m1, m0 = np.polyfit(np.log(cum_mean.n_array()), np.log(means), 1)
        #assert -m1 > 0.3    # convergence rate close to 0.5 (optimal for MC)
        #assert m0 < 0.1     # small absolute error

        # ========= Sigma, check convergence ======================================
        sigma_err = np.abs( cum_sigma.avg_array() - sigma )
        means     = np.mean(sigma_err, axis=1)
        s1, s0    = np.polyfit(np.log(cum_sigma.n_array()), np.log(means), 1)
        #assert -s1 > 0.4    # convergence rate close to 0.5 (optimal for MC)
        #assert s0 < 0.05     # small absolute error
        
        # ========== Plots and visualization =====================================
        fig = plt.figure(figsize=plt.figaspect(1))
        ax  = fig.add_subplot(2,2,1)
        self.plot_mc(cum_mean.n_array(), mu_err,"Convergence of 'mu' with number of samples N")   # convergence plot
        ax  = fig.add_subplot(2,2,2)
        self.plot_mc(cum_sigma.n_array(), sigma_err,"Convergence of 'sigma' with number of samples N")   # convergence plot
        ax  = fig.add_subplot(2,2,3)
        self.plot_grid_field_2d(ncells, cum_sigma.avg_array()[-1, :], "Last sigma conv")
        ax  = fig.add_subplot(2,2,4)
        self.plot_grid_field_2d(ncells, sigma_err[-1, :], "Error in 'sigma' estimate, N={}.".format(n_samples))  # error distribution
        
        # =========== Table output =============================================
        tab = tt.Texttable()
        headings = ['convergence','m1','m0','mu','s1','s0','sig2']
        tab.header(headings)
        tab.add_row([str(typecorr),m1,m0,cum_mean.avg_array().mean(),s1,s0,cum_sigma.avg_array().mean()])
        s = tab.draw()
        print(s)

        
        return mu_err, sigma_err        