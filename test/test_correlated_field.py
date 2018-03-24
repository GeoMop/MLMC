# TEST OF CONSISTENCY in the field values generated

import pytest
import numpy as np
import numpy.linalg as la
import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

from mlmc.correlated_field import SpatialCorrelatedField
#import scipy.interpolate as sc_inter
#import scipy.stats as sc_stat


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

    def __iadd__(self, other):      # Overridding the += method
        self.cumul += other
        self.n_iter += 1
        if self.n_iter > self.log_limit:
            self.log_cum.append( (self.n_iter, self.cumul.copy()))
            self.log_limit *= 2
        return self

    def finalize(self):
        """
        Add final cumulation point to the table.
        :return:
        """
        self.log_cum.append((self.n_iter, self.cumul))

    def avg_array(self):
        """
        :return: Array L x Shape, L averages for increasing number of samples.
        """
        return np.array([cumul/n for n, cumul in self.log_cum])    # Vysvetlit, why long arrays   

    def n_array(self):
        """
        :return: Array L, number of samples on individual levels.
        """
        return np.array([n for n,cumul in self.log_cum])


class CorrError():
    """
    Auxiliary class for evaluating error in correlation function of the field.
    """
    def __init__(self, corr_fn, radius, n_samples=100):
        """
        :param corr_fn: Scalar correlation function for an isotropic stationary field.
        :param radius: Maximal distance of two points.
        :param n_samples:
        """
        self.corr_fn = corr_fn
        self.n_samples = n_samples
        self.radius = radius
        #self.cum_array = [Cumul(1) for i in range(n_samples)]
        #self.dl = self.radius / n_samples
        self.samples=[]

    def add_samples(self, points, field):
        """
        :param:Points where the random field is evaluated.
        :param field: Field values with mean == 0.0
        :return:
        """
        for i in range(self.n_samples):
            k = np.random.random_integers(0, len(field)-1, 2)
            l = la.norm(points[k[0]] - points[k[1]], 2)
            if l >= self.radius:
                continue
            #corr_fn = self.corr_fn(l)
            corr_sample = field[k[0]] * field[k[1]]
            #il = int(l / self.dl)
            #self.cum_array[il] += corr_fn - corr_sample
            self.samples.append( (l, corr_sample))

    def plot(self):
        pass


class PointSet:
    def __init__(self, bounds, n_points):
        """
        :param bounds: Bounding box of points. Tuple (min_point, max_point).
        :param n_points: Either int for random points or list [nx, ny, ...] for regular grid.
        """
        self.min_pt = np.array(bounds[0], dtype=float)
        self.max_pt = np.array(bounds[1], dtype=float)
        self.n_points = n_points
        assert len(self.min_pt) == len(self.max_pt)
        dim = len(self.min_pt)
        if type(n_points) == int:
            # random points
            self.regular_grid = False
            self.points = np.random.rand(n_points, dim)*(self.max_pt - self.min_pt) + self.min_pt
        else:
            # grid
            assert dim == len(n_points)
            self.regular_grid = True
            grids = []
            for d in range(dim):
                grid = np.linspace(self.min_pt[d], self.max_pt[d], num = n_points[d])
                grids.append(grid)
            grids = [ g.ravel() for g in np.meshgrid(*grids)]
            self.points = np.stack(grids, axis=1)

        self.dim = dim
        self.size = self.points.shape[0]

    def plot_field_2d(self, values, title):
        assert self.dim == 2
        if self.regular_grid:
            # imgshow plot X axis verticaly, need to swap
            grid = values.reshape((self.n_points[1], self.n_points[0]))
            imgplot = plt.imshow(grid)
            plt.colorbar()
        else:
            pass
        plt.title(title)
        plt.show()




def plot_mc(n_samples, data):
    """
    :param n_samples: np.array with number of samples for L different sizes
    :param data: LxN array, we make scatter + box plot from N size vectors on individual stages
    :return:
    """
    means = np.mean(data, axis=1)
    devs = np.std(data, axis=1)
    for n, field_avg in zip(n_samples, data):
        X = n + 0.01*np.random.rand(len(field_avg))
        plt.scatter(X, field_avg)
    plt.errorbar(n_samples, means, yerr=devs, fmt='-o', capsize=3, ecolor="lightblue")
    m1, m0 = np.polyfit(np.log(n_samples), np.log(means), 1)

    legend = "rate: {}".format(m1)
    plt.plot(n_samples, np.exp(m1 * np.log(n_samples) + m0), '--k', label=legend)


    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title("Convergence of 'mu'/'sigma' with number of samples N")
    plt.xlabel("log_2 N")
    plt.show()





def perform_test_on_point_set(corr_exp, points, n_terms_range):
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
    n_pt = points.size
    corr_length = 10
    mu = 3.14
    sigma = 1.5
    field = SpatialCorrelatedField(corr_exp, dim=points.dim, corr_length = corr_length)
    field.set_points(points.points, mu, sigma)
    field.svd_dcmp(precision=0.01, n_terms_range=n_terms_range)


    # # plot single sample
    # self.plot_grid_field_2d(ncells, field.sample())

    # Estimate statistcs by Monte Carlo
    n_samples = 2300

    cum_mean = Cumul(n_pt)
    cum_sigma = Cumul(n_pt)

    # correlation funcion - stationary, isotropic
    corr_fn = lambda dist: (sigma**2)*np.exp( (-1.0/corr_exp)*(dist / corr_length)**corr_exp)

    radius = la.norm(np.max(points.points, axis=0) - np.min(points.points, axis=0))
    corr_error = CorrError(corr_fn, int(radius), n_samples=100)

    # =========== Testing ===========================================================
    for j in range(n_samples):
        sample = field.sample()
        cum_mean += sample
        centered = sample - mu
        cum_sigma += centered * centered
        #corr_error.add_samples(points, centered)

    #### Mean plots and tests
    mu_err = np.abs(cum_mean.avg_array() - mu)  # cum_mean.avg%array has size [log 2 N * n] but its one value along n axis
    #plot_mc(cum_mean.n_array(), mu_err)   # convergence plot
    #points.plot_field_2d(mu_err[-1, :], "Error in 'mu' estimate, N={}.".format(n_samples))  # error distribution  , the last averaged error?

    # check convergence
    means = np.mean(mu_err, axis=1)
    m1, m0 = np.polyfit(np.log(cum_mean.n_array()), np.log(means), 1)
    log_mean = np.average(np.log(means))
    assert -m1 > 0.4    # convergence rate close to 0.5 (optimal for MC)
    print("Mean fit: {} {} {}".format( m1, log_mean, np.exp(log_mean)))
    assert np.exp(log_mean) < 0.2     # should be about 0.1


    ### Sigma plot and test

    # Estimate sigma of field average
    # i-th sample X_i,j = mu_j + sigma_j * L_j,k * P_i,k, where P_i,k ~ N(0,1)
    # avg(X_j)_i = avg(mu) + avg(sigma_j * L_j)_k * P_i,k
    # avg(X_i,j) = avg(mu) + avg(sigma_j * L_j)_k * avg(P_i)_k ; ovsem avg(P_i) ~ Student ~ N(0,1/n_i)
    # avg(X_i,j) = avg(mu) + beta_k * Y_k ; Y_k ~ N(0, 1/N_i)
    # D(avg(X_i,j))  =  sum(beta_k^2) / N_i / N_k^2
    #var_field_avg = (sigma * np.sum(field._cov_l_factor) / n_pt)**2 / n_samples
    #var_field = sigma**2 * np.sum(field._cov_l_factor**2, axis=1) / n_samples
    #print(var_field_avg)
    #print(var_field)
    #X = np.full(n_pt, 1.0)
    #plt.plot(X, var_field)
    #plt.plot([0.0, 1.0], np.full(2, var_field_avg))
    #plt.show()

    #self.plot_grid_field_2d(ncells, np.sqrt(cum_sigma.avg_array()[0,:]), "sigma conv")
    #self.plot_grid_field_2d(ncells, np.sqrt(cum_sigma.avg_array()[-1, :]), "sigma conv")
    sigma_err = np.abs( np.sqrt(cum_sigma.avg_array()) - sigma )
    #plot_mc(cum_sigma.n_array(), sigma_err)   # convergence plot
    #self.plot_grid_field_2d(ncells, sigma_err[-1, :], "Error in 'sigma' estimate, N={}.".format(n_samples))  # error distribution

    means = np.mean(sigma_err, axis=1)
    s1, s0 = np.polyfit(np.log(cum_sigma.n_array()), np.log(means), 1)

    assert -s1 > 0.38    # convergence rate close to 0.5 (optimal for MC)
    #assert s0 < 0.05     # small absolute error
    log_mean = np.average(np.log(means))
    print("Mean fit: {} {} {}".format( s1, log_mean, np.exp(log_mean)))
    assert np.exp(log_mean) < 0.1     # should be about 0.7


    #### Correlation plots and tests
    #means = [ cumm.avg_array()[-1] for cumm in corr_error.cum_array]
    #plt.plot(means)
    #plt.show()
    #corr_samples = np.array(corr_error.samples)
    #sorted_idx = corr_samples[:, 0].argsort()
    #corr_sorted = corr_samples[ sorted_idx, :]

    # group same 'l' values
    # l_last=-1
    # corr_group=[]
    # l_values = []
    # means=[]
    # devs=[]
    # for l, corr in zip(corr_sorted[:, 0], corr_sorted[:, 1]):
    #     if l != l_last:
    #         if corr_group:
    #             array = np.array(corr_group)
    #             means.append(np.mean(array))
    #             devs.append(np.std(array))
    #             l_values.append(l_last)
    #             corr_group=[]
    #         l_last = l
    #     corr_group.append(corr)

    #corr_fn_samples = [corr_fn(l) for l in corr_sorted[:, 0]]

    # plot samples and theoretical fn
    # plt.scatter(corr_sorted[:,0], corr_sorted[:, 1], c='b', s=2.0 )
    # plt.plot(corr_sorted[:, 0], corr_fn_samples, c='r')
    #
    # ## plot errors
    # plt.scatter(corr_sorted[:, 0], np.sqrt( (corr_sorted[:, 1] - corr_fn_samples)**2 ) , c='b', s=2.0)
    #
    # knots=[0,0, 0, 1, 10, 30, 45,45,45]
    # x = corr_sorted[:, 0].copy()
    # x[1:] += 1e-14 * np.cumsum(x[1:] == x[:-1])
    #
    #
    # f = sc_inter.make_lsq_spline(x, corr_sorted[:, 1], knots, k=2 )
    # # z = np.polyfit(corr_sorted[:,0], corr_sorted[:, 1], 5)
    # # f = np.poly1d(z)
    # x_new = np.linspace(corr_sorted[0, 0], corr_sorted[-1, 0], 100)
    #
    # plt.plot(x_new, f(x_new), c='g')
    #
    #
    #
    # #plt.errorbar(l_values, means, yerr=devs, fmt='-o', capsize=3, ecolor="lightblue")
    # plt.ylim([0, 2])
    # plt.show()
    #
    # ## Chi square test
    # chi_sq = np.sum( (corr_sorted[:, 1] - corr_fn_samples) ** 2 )
    # p_val = 1 - sc_stat.chi2.cdf(chi_sq, df = len(corr_sorted[:,1])  )
    # print("H0: match of covarianvce function, p-val: ", p_val)



    #plt.plot(np.arange(0,n), mu_err[-1, :])
    #plt.show()

    return mu_err, None


    # ============================ Comparison ======================================
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax  = fig.add_subplot(1, 2, 1)
    ax.set_title('The average value',size = 11)
    plt.plot(fp_ave, label = 'From the field on fine grid')
    plt.plot(fr_ave, label = 'With reduced covariance matrix')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    ax  = fig.add_subplot(1, 2, 2)
    ax.set_title('The variance',size = 11)
    plt.plot(fp_var, label = 'From the field on fine grid')
    plt.plot(fr_var, label = 'With reduced covariance matrix')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.show()
    """


def _test_single_field():
    # Test properties of a single field

    ncells = [50, 40]
    points = make_points_grid(ncells, ncells)

    n = len(points)
    mu = 0

    sigma = 0.5
    corr_exp = 1.5
    corr_length = 5
    field = SpatialCorrelatedField(corr_exp = corr_exp, dim=2, corr_length = corr_length)
    field.set_points(points, mu, sigma)
    field.svd_dcmp(precision=0.01, n_terms_range=(60, 1000))

    sample = field.sample()
    self.plot_grid_field_2d(ncells, sample)

    # test correlation function
    n_samples=1000
    cum_array = np.zeros(int(np.sqrt(ncells[0]**2 + ncells[1]**2)))
    for i in range(n_samples):
        i = np.random.random_integers(0, ncells[0]-1, 2)
        j = np.random.random_integers(0, ncells[1]-1, 2)
        di = np.abs(i[1] - i[0])
        dj = np.abs(j[1] - j[0])
        l= int(np.sqrt(di ** 2 + dj ** 2))
        exponent  = -1.0/corr_exp * (di**2 + dj**2)**corr_exp
        corr_fn = sigma**2 * np.exp(exponent)
        n = i + ncells[0] * j
        corr_sample = (sample[n[0]] - mu) * (sample[n[1]] - mu)
        cum_array[l] += (corr_fn - corr_sample)
    plt.plot(cum_array / n_samples)
    plt.show()

@pytest.mark.parametrize('seed', [2,3,4,5,6])
def test_field_values_consistency(seed):
    np.random.seed(seed)
    np.random.rand(1000)
    # ===========  A structured grid of points: =====================================
    bounds = ([0,0], [32, 32])
    grid_size = [20, 30]
    points = PointSet(bounds, grid_size)
    corr_exp = 1.0
    n_terms = (np.inf, np.inf)  # Use full expansion to avoid error in approximation.
    full_mean_err, full_cov_err = perform_test_on_point_set(corr_exp, points, n_terms_range = n_terms)
    corr_exp = 2.0
    full_mean_err, full_cov_err = perform_test_on_point_set(corr_exp, points, n_terms_range=n_terms)
    #assert la.norm(full_mean_err) < 1
    #assert la.norm(full_cov_err) < 1

    #red_mean_err, red_cov_err = self.perform_test_on_point_set(points, n_terms_range=(1, np.inf))
    #assert la.norm(red_mean_err) < 1
    #assert la.norm(red_cov_err) < 1

    ## Random point set
    #bound = [32,32,32]
    #size = 9 * 17
    #points  = make_points_random(bound, size)

    #self.perform_test_on_point_set(points)
