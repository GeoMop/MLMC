# TEST OF CONSISTENCY in the field values generated


import pytest
import numpy as np
import numpy.linalg as la

from mlmc.correlated_field import SpatialCorrelatedField

# Only for debugging
#import statprof
#import matplotlib
#matplotlib.use("agg")
#import matplotlib.pyplot as plt
#plt.switch_backend('agg')

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
            # Uniform
            self.points = np.random.rand(n_points, dim)*(self.max_pt - self.min_pt) + self.min_pt
            # Normal
            #self.points = np.random.randn(n_points, dim) * la.norm(self.max_pt - self.min_pt, 2)/10 + (self.min_pt + self.max_pt)/2
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
        assert len(values) == self.size
        if self.regular_grid:
            # imgshow plot X axis verticaly, need to swap
            grid = values.reshape((self.n_points[1], self.n_points[0]))
            imgplot = plt.imshow(grid)
            plt.colorbar()
        else:
            plt.scatter(self.points[:,0], self.points[:,1], c=values)
        plt.title(title)
        plt.show()




def plot_mc(n_samples, data, title=""):
    """
    :param n_samples: np.array with number of samples for L different sizes
    :param data: LxN array, we make box plot from N size vectors on individual stages
    :return:
    """
    means = np.mean(data, axis=1)
    devs = np.std(data, axis=1)
    for n, field_avg in zip(n_samples, data):
        X = n + 0.01*np.random.rand(len(field_avg))
        #plt.scatter(X, field_avg)
    plt.errorbar(n_samples, means, yerr=devs, fmt='-o', capsize=3, ecolor="lightblue")
    m1, m0 = np.polyfit(np.log(n_samples), np.log(means), 1)

    legend = "rate: {}".format(m1)
    plt.plot(n_samples, np.exp(m1 * np.log(n_samples) + m0), '--k', label=legend)


    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title(title)
    plt.xlabel("log_2 N")
    plt.show()





def impl_test_mu_sigma(corr_exp, points, n_terms_range):
    """
    Test rate of convergence for global mu and sigma.
    :param corr_exp: Covarianc exponent,
    :param points: Point set
    :param n_terms_range: Limits for size of approximation
    :return:
    """
    n_pt = points.size
    corr_length = 2
    mu = 3.14
    sigma = 1.5
    field = SpatialCorrelatedField(corr_exp, dim=points.dim, corr_length = corr_length)
    field.set_points(points.points, mu, sigma)
    field.svd_dcmp(precision=0.01, n_terms_range=n_terms_range)


    # # plot single sample
    #points.plot_field_2d(field.sample(), "Single sample exp: {}".format(corr_exp))

    # Estimate mu and sigma  by Monte Carlo
    n_samples = 2300
    cum_mean = Cumul(n_pt)
    cum_sigma = Cumul(n_pt)
    for j in range(n_samples):
        sample = field.sample()
        cum_mean += sample
        centered = sample - mu
        cum_sigma += centered * centered

    #### Mean plots and tests
    mu_err = np.abs(cum_mean.avg_array() - mu)  # cum_mean.avg%array has size [log 2 N * n] but its one value along n axis
    #plot_mc(cum_mean.n_array(), mu_err, "Error of 'mu' estimate as func of samples.")   # convergence plot
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


    sigma_err = np.abs( np.sqrt(cum_sigma.avg_array()) - sigma )
    #plot_mc(cum_sigma.n_array(), sigma_err, "Error of 'sigma' estimate as func of samples.")   # convergence plot
    #self.plot_grid_field_2d(ncells, sigma_err[-1, :], "Error in 'sigma' estimate, N={}.".format(n_samples))  # error distribution

    sigmas = np.mean(sigma_err, axis=1)
    s1, s0 = np.polyfit(np.log(cum_sigma.n_array()), np.log(sigmas), 1)

    assert -s1 > 0.38    # convergence rate close to 0.5 (optimal for MC)
    #assert s0 < 0.05     # small absolute error
    log_sigma = np.average(np.log(sigmas))
    print("Sigma fit: {} {} {}".format( s1, log_sigma, np.exp(log_sigma)))
    assert np.exp(log_sigma) < 0.1     # should be about 0.7








@pytest.mark.parametrize('seed', [2,3,4,5,6])
#@pytest.mark.parametrize('seed', [3])
def test_field_mean_std_convergence(seed):
    np.random.seed(seed)
    np.random.rand(1000)
    # ===========  A structured grid of points: =====================================
    bounds = ([13, 3], [40, 32])
    grid_size = [10, 15]
    grid_points = PointSet(bounds, grid_size)
    random_points = PointSet(bounds, grid_size[0] * grid_size[1])
    exponential = 1.0
    gauss = 2.0
    n_terms = (np.inf, np.inf)  # Use full expansion to avoid error in approximation.


    #statprof.start()
    try:
        print("Test exponential, grid points.")
        impl_test_mu_sigma(exponential, grid_points, n_terms_range = n_terms)
        print("Test Gauss, grid points.")
        impl_test_mu_sigma(gauss, grid_points, n_terms_range=n_terms)
        print("Test exponential, random points.")
        impl_test_mu_sigma(exponential, random_points, n_terms_range=n_terms)
        print("Test Gauss, random points.")
        impl_test_mu_sigma(gauss, random_points, n_terms_range=n_terms)
    finally:
        #statprof.stop()
        #statprof.display()
        pass




def impl_test_cov_func(corr_exp, points, n_terms_range):
    """

    :param points: Point set
    :return:
    """

    n_pt = points.size
    corr_length = 10.0
    field = SpatialCorrelatedField(corr_exp, dim=points.dim, corr_length=corr_length)
    field.set_points(points.points)
    field.svd_dcmp(precision=0.01, n_terms_range=n_terms_range)

    # # plot single sample
    #points.plot_field_2d(field.sample(), "Single sample exp: {}".format(corr_exp))

    # Select pairs to sample various point distances


    radius = 0.5 * la.norm(points.max_pt - points.min_pt, 2)
    n_cells = 20
    i_cell = lambda l: ( n_cells * np.minimum(l/radius, 1.0) ).astype(int)
    n_fn_samples = 20000
    samples_per_cell = 10

    pairs = np.random.choice(points.size, (n_cells*n_fn_samples, 2))
    cells = i_cell( la.norm(points.points[pairs[:,0]] - points.points[pairs[:,1]], axis=1) )
    cell_pairs = [set() for i in range(n_cells+1)]
    for ij, i_cell in zip(pairs, cells):
        cell_pairs[i_cell].add(tuple(ij))

    # Merge cells with few samples
    cell_lists = [[]]
    for c in cell_pairs:
        if len(cell_lists[-1]) < samples_per_cell:
            cell_lists[-1] += list(c)
        else:
            cell_lists[-1] = cell_lists[-1][:samples_per_cell]
            cell_lists.append(list(c))
    cell_lists.pop(-1)

    print("Lens: ", [len(cp) for cp in cell_lists])
    pairs_array = np.transpose(np.array(cell_lists), axes=(1,0,2))
    # Estimate statistcs by Monte Carlo


    # correlation funcion - stationary, isotropic
    corr_fn = lambda dist: np.exp((-1.0 / corr_exp) * (dist / corr_length) ** corr_exp)

    #cumul = np.zeros(n_cells+1)
    #n_corr_samples = np.zeros(n_cells+1)

    errors = Cumul(len(cell_lists))
    lengths = Cumul(len(cell_lists))
    n_samples = 1000
    for j in range(n_samples):
        sample = field.sample()
        for pa in pairs_array:
            dist = la.norm(points.points[pa[:,0]] - points.points[pa[:,1]], axis=1)
            err = sample[pa[:,0]] * sample[pa[:,1]] - corr_fn(dist)
            errors += err
            lengths += dist

    #cumul.pop(-1)

    #avg_err = [cumul.avg_array() for cumul in cumul_table]
    #n_samples = [cumul.n_array() for cumul in cumul_table]
    #n_levels = max([len(cell_avg) for cell_avg in avg_err])

    #X = lengths.avg_array()[-1, :]
    Y = np.std(errors.avg_array(), axis=1)
    X = errors.n_array()

    #    norm = matplotlib.colors.Normalize(0, n_levels)
    #    color = matplotlib.cm.hot(1.0 - norm(l))
    #    plt.plot(X, Y, c=color, label=str(l))
    #plot_mc(errors.n_array(), avg_errors*avg_errors, "Error of covariance function estimate.")

    # TODO: generalize PlotMC to this case
    m1, m0 = np.polyfit(np.log(X), np.log(Y), 1)
    legend = "rate: {}".format(m1)

    # plt.plot(X, Y)
    # plt.plot(X, np.exp(m1 * np.log(X) + m0), '--k', label=legend)
    # plt.legend()
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()

    log_mean = np.average(np.log(Y))
    assert -m1 > 0.38    # convergence rate close to 0.5 (optimal for MC)
    print("Mean fit: {} {} {}".format( m1, log_mean, np.exp(log_mean)))
    assert np.exp(log_mean) < 0.1     # should be about 0.05



@pytest.mark.parametrize('seed', [5, 7])
def test_cov_func_convergence(seed):
    # TODO:
    # Seems that we have systematic error in covariance function.
    # Error seems to grow with distance. About l**0.1 for corr_exp==1,
    # faster grow for corr_exp == 2.
    # Not clear if it is a selection effect however there is cleare convergence
    # of the error to some smooth limit function.
    # Using (sum of squares) of absolute error gives systemetic error
    # between 0.5 - 4.0, seems to grow a bit with level !! but no grow with distance.
    # No influence of uniform vs. normal points. Just more cosy for larger distances
    # as there is smaller sample set.
    np.random.seed()
    np.random.rand(100)
    # ===========  A structured grid of points: =====================================
    bounds = ([0, 0], [40, 30])
    random_points = PointSet(bounds, 1000)
    exponential = 1.0
    gauss = 2.0
    n_terms = (np.inf, np.inf)  # Use full expansion to avoid error in approximation.
    impl_test_cov_func(exponential, random_points, n_terms_range=n_terms)
    impl_test_cov_func(gauss, random_points, n_terms_range=n_terms)
    #impl_test_cov_func(gauss, random_points, n_terms_range=n_terms)

