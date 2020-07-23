# TEST OF CONSISTENCY in the field values generated
import pytest
import numpy as np
import numpy.linalg as la

from mlmc.random.correlated_field import SpatialCorrelatedField
from mlmc.random.correlated_field import GSToolsSpatialCorrelatedField

# Only for debugging
#import statprof
import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import gstools


class Cumul:
    """
    Auxiliary class for evaluation convergence of a MC approximation of EX for
    a random field of given shape.
    """
    def __init__(self, shape):
        self.cumul = np.zeros(shape)
        self.n_iter = 0
        self.log_cum = []
        self.log_limit = 16

    def __iadd__(self, other):      # Overridding the += method
        self.cumul += other
        self.n_iter += 1
        if self.n_iter > self.log_limit:
            self.log_cum.append((self.n_iter, self.cumul.copy()))
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
        return np.array([cumul/n for n, cumul in self.log_cum])

    def n_array(self):
        """
        :return: Array L, number of samples on individual levels.
        """
        return np.array([n for n, cumul in self.log_cum])


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
            #self.points = np.random.randn(n_points, dim) * la.norm(self.max_pt - self.min_pt, 2)/10 +
            #  (self.min_pt + self.max_pt)/2
        else:
            # grid
            assert dim == len(n_points)
            self.regular_grid = True
            grids = []
            for d in range(dim):
                grid = np.linspace(self.min_pt[d], self.max_pt[d], num=n_points[d])
                grids.append(grid)
            grids = [g.ravel() for g in np.meshgrid(*grids)]
            self.points = np.stack(grids, axis=1)

        self.dim = dim
        self.size = self.points.shape[0]

    def plot_field_2d(self, values, title):
        """
        Plot 2d field
        :param values: Values to plot
        :param title: Plot title
        :return: None
        """
        assert self.dim == 2
        assert len(values) == self.size

        if self.regular_grid:
            # imgshow plot X axis verticaly, need to swap
            grid = values.reshape((self.n_points[1], self.n_points[0]))
            imgplot = plt.imshow(grid)
            plt.colorbar()
        else:
            plt.scatter(self.points[:, 0], self.points[:, 1], c=values)
        plt.title(title)
        plt.show()


def plot_mc(n_samples, data, title=""):
    """
    Plot Monte Carlo data
    :param n_samples: np.array with number of samples for L different sizes
    :param data: LxN array, we make box plot from N size vectors on individual stages
    :param title: Plot title
    :return: None
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


def impl_test_mu_sigma(field_impl, corr_exp, points, n_terms_range, corr_length=2):
    """
    Test rate of convergence for global mu and sigma.
    :param corr_exp: Correlation exponent,
    :param points: Point set
    :param n_terms_range: Limits for size of approximation
    :param corr_length: correlation length
    :return: None
    """
    n_pt = points.size
    mu = 3.14
    sigma = 1.5

    if isinstance(field_impl, GSToolsSpatialCorrelatedField):
        field = field_impl
    else:
        field = field_impl(corr_exp, dim=points.dim, corr_length=corr_length)

    field.set_points(points.points, mu, sigma)
    if isinstance(field, SpatialCorrelatedField):
        field.svd_dcmp(precision=0.01, n_terms_range=n_terms_range)

    # # plot single sample
    #points.plot_field_2d(field.sample(), "Single sample exp: {}".format(corr_exp))

    # Estimate mu and sigma by Monte Carlo
    n_samples = 5000

    cum_mean = Cumul(n_pt)
    cum_sigma = Cumul(n_pt)

    for _ in range(n_samples):
        if isinstance(field_impl, GSToolsSpatialCorrelatedField):
            field.change_srf(seed=np.random.randint(0, 1e5))
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

    m1, _ = np.polyfit(np.log(cum_mean.n_array()), np.log(means), 1)
    log_mean = np.average(np.log(means))

    assert -m1 > 0.3    # convergence rate close to 0.5 (optimal for MC)
    print("Mean fit: {} {} {}".format(m1, log_mean, np.exp(log_mean)))
    assert np.exp(log_mean) < 0.2     # should be about 0.1

    ### Sigma plot and test

    # Estimate sigma of field average
    # i-th sample X_i,j = mu_j + sigma_j * L_j,k * P_i,k, where P_i,k ~ N(0,1)
    # avg(X_j)_i = avg(mu) + avg(sigma_j * L_j)_k * P_i,k
    # avg(X_i,j) = avg(mu) + avg(sigma_j * L_j)_k * avg(P_i)_k ; ovsem avg(P_i) ~ Student ~ N(0,1/n_i)
    # avg(X_i,j) = avg(mu) + beta_k * Y_k ; Y_k ~ N(0, 1/N_i)
    # D(avg(X_i,j))  =  sum(beta_k^2) / N_i / N_k^2

        # var_field_avg = (sigma * np.sum(field._cov_l_factor) / n_pt) ** 2 / n_samples
        # var_field = sigma ** 2 * np.sum(field._cov_l_factor ** 2, axis=1) / n_samples
        # X = np.full(n_pt, 1.0)
        # plt.plot(X, var_field)
        # plt.plot([0.0, 1.0], np.full(2, var_field_avg))
        # plt.show()


    sigma_err = np.abs(np.sqrt(cum_sigma.avg_array()) - sigma)
    #plot_mc(cum_sigma.n_array(), sigma_err, "Error of 'sigma' estimate as func of samples.")   # convergence plot
    #plot_grid_field_2d(ncells, sigma_err[-1, :], "Error in 'sigma' estimate, N={}.".format(n_samples))  # error distribution

    sigmas = np.mean(sigma_err, axis=1)
    s1, s0 = np.polyfit(np.log(cum_sigma.n_array()), np.log(sigmas), 1)
    # print("s1 ", s1)
    # print("s0 ", s0)

    assert -s1 > 0.38    # convergence rate close to 0.5 (optimal for MC)
    #assert s0 < 0.05     # small absolute error
    log_sigma = np.average(np.log(sigmas))
    print("Sigma fit: {} {} {}".format(s1, log_sigma, np.exp(log_sigma)))
    assert np.exp(log_sigma) < 0.1     # should be about 0.7

#@pytest.mark.skip
@pytest.mark.parametrize('seed', [2, 5])
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
    corr_length = 2
    n_terms = (np.inf, np.inf)  # Use full expansion to avoid error in approximation.

    impl = SpatialCorrelatedField
    #print("Test exponential, grid points.")
    impl_test_mu_sigma(impl, exponential, grid_points, n_terms_range=n_terms)
    #print("Test Gauss, grid points.")
    impl_test_mu_sigma(impl, gauss, grid_points, n_terms_range=n_terms)
    #print("Test exponential, random points.")
    impl_test_mu_sigma(impl, exponential, random_points, n_terms_range=n_terms)
    #print("Test Gauss, random points.")
    impl_test_mu_sigma(impl, gauss, random_points, n_terms_range=n_terms)

    len_scale = corr_length * 2 * np.pi

    # Random points gauss model
    gauss_model = gstools.Gaussian(dim=random_points.dim, len_scale=len_scale)
    impl = GSToolsSpatialCorrelatedField(gauss_model)
    impl_test_mu_sigma(impl, gauss, random_points, n_terms_range=n_terms, corr_length=2)
    # Random points exp model
    exp_model = gstools.Exponential(dim=random_points.dim, len_scale=len_scale)
    impl = GSToolsSpatialCorrelatedField(exp_model)
    impl_test_mu_sigma(impl, exponential, random_points, n_terms_range=n_terms, corr_length=2)

    # Grid points gauss model
    gauss_model = gstools.Gaussian(dim=grid_points.dim, len_scale=len_scale)
    impl = GSToolsSpatialCorrelatedField(gauss_model)
    impl_test_mu_sigma(impl, gauss_model, grid_points, n_terms_range=n_terms, corr_length=2)

    # Grid points exp model
    exp_model = gstools.Exponential(dim=grid_points.dim, len_scale=len_scale)
    impl = GSToolsSpatialCorrelatedField(exp_model)
    impl_test_mu_sigma(impl, exp_model, grid_points, n_terms_range=n_terms, corr_length=2)


def impl_test_cov_func(field_impl, corr_exp, points, n_terms_range, corr_length=10):
    """
    Test if random field covariance match given covariance function
    :param field_impl: class for random field generation
    :param corr_exp: Correlation exponent, currently: 1 - exponential distr, 2 - gauss distr
    :param points: PointSet instance
    :param n_terms_range: (min, max), number of terms in KL expansion to use
    :param corr_length: correlation length, default 10
    :return: None
    """
    if isinstance(field_impl, GSToolsSpatialCorrelatedField):
        field = field_impl
    else:
        field = field_impl(corr_exp, dim=points.dim, corr_length=corr_length)

    field.set_points(points.points, mu=0, sigma=1)
    if isinstance(field, SpatialCorrelatedField):
        field.svd_dcmp(precision=0.01, n_terms_range=n_terms_range)
    # # plot single sample
    #points.plot_field_2d(field.sample(), "Single sample exp: {}".format(corr_exp))

    # Select pairs to sample various point distances
    radius = 0.5 * la.norm(points.max_pt - points.min_pt, 2)
    n_cells = 20
    n_fn_samples = 20

    # random pairs of points
    pairs = np.random.choice(points.size, (n_cells*n_fn_samples, 2))
    pair_dists = la.norm(points.points[pairs[:, 0]] - points.points[pairs[:, 1]], axis=1)
    indices = np.argsort(pair_dists)
    pair_dists = pair_dists[indices].reshape(n_cells, n_fn_samples)
    cell_lists = np.transpose(pairs[indices, :].reshape(n_cells, n_fn_samples, 2), axes=(1, 0, 2))
    lengths = np.mean(pair_dists, axis=1)

    # Estimate statistics by Monte Carlo
    # correlation function - stationary, isotropic
    if isinstance(field, SpatialCorrelatedField):
        corr_fn = lambda dist: np.exp(-(dist / corr_length) ** corr_exp)
    else:
        corr_fn = field.model.correlation

    errors = Cumul(n_cells)
    #lengths = Cumul(n_cells)
    field_diffs = Cumul(n_cells)

    n_samples = 2000
    for _ in range(n_samples):
        if isinstance(field_impl, GSToolsSpatialCorrelatedField):
            field.change_srf(seed=np.random.randint(0, 1e5))
        sample = field.sample()
        for i_pt, pa in enumerate(cell_lists):
            dist = pair_dists[:, i_pt]
            err = sample[pa[:, 0]] * sample[pa[:, 1]] - corr_fn(dist)
            #field_diff = 0.5*(sample[pa[:, 0]] - sample[pa[:, 1]])**2
            field_diff = sample[pa[:, 0]] * sample[pa[:, 1]]
            field_diffs += field_diff
            errors += err
            #lengths += dist
    errors.finalize()
    field_diffs.finalize()

    def plot_error_fn():
        X = lengths.avg_array()[-1]
        Y = errors.avg_array()[-1]
        plt.plot(X, Y)
        plt.show()

    def plot_variogram():
        # For sigma == 1 variogram is 1-correlation function
        # Plot mean of every cell.
        variogram = field_diffs.avg_array()[-1,:]
        plt.plot(lengths, variogram)
        plt.plot(lengths, corr_fn(lengths), c='red')
        plt.show()
    #plot_variogram()

    # TODO: generalize PlotMC to this case
    Y = np.std(errors.avg_array(), axis=1)
    X = errors.n_array()
    m1, m0 = np.polyfit(np.log(X), np.log(Y), 1)
    log_mean = np.average(np.log(Y))

    def plot_fit():
        legend = "rate: {}".format(m1)
        for c in range(n_cells):
            Y = errors.avg_array()[:, c]
            col = plt.cm.viridis(plt.Normalize(0,n_cells)(c))
            plt.plot(X, Y, c=col)

        plt.plot(X, np.exp(m1 * np.log(X) + m0), '--k', label=legend)
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
    #plot_fit()

    # @TODO determine the parameters more precisely
    assert -m1 > 0.3    # convergence rate close to 0.5 (optimal for MC)
    print("Mean fit: {} {} {}".format(m1, log_mean, np.exp(log_mean)))
    assert np.exp(log_mean) < 0.08

#@pytest.mark.skip
@pytest.mark.parametrize('seed', [10, 8])
def test_cov_func_convergence(seed):
    # TODO:
    # Seems that we have systematic error in covariance function.
    # Error seems to grow with distance. About l**0.1 for corr_exp==1,
    # faster grow for corr_exp == 2.
    # Not clear if it is a selection effect however there is cleare convergence
    # of the error to some smooth limit function.
    # Using (sum of squares) of absolute error gives systemetic error
    # between 0.5 - 4.0, seems o grow a bit with level !! but no grow with distance.
    # No influence of uniform vs. normal points. Just more cosy for larger distances
    # as there is smaller sample set.
    np.random.seed(seed)
    np.random.rand(100)
    # ===========  A structured grid of points: =====================================
    bounds = ([0, 0], [40, 30])
    random_points = PointSet(bounds, 100)
    exponential = 1.0
    gauss = 2.0
    corr_length = 2
    n_terms = (np.inf, np.inf)  # Use full expansion to avoid error in approximation.

    impl_test_cov_func(SpatialCorrelatedField, gauss, random_points, n_terms_range=n_terms, corr_length=corr_length)
    impl_test_cov_func(SpatialCorrelatedField, exponential, random_points, n_terms_range=n_terms, corr_length=corr_length)

    len_scale = corr_length * 2 * np.pi
    gauss_model = gstools.Gaussian(dim=random_points.dim, len_scale=len_scale)
    impl = GSToolsSpatialCorrelatedField(gauss_model)
    impl_test_cov_func(impl, gauss, random_points, n_terms_range=n_terms)
    # Random points exp model
    exp_model = gstools.Exponential(dim=random_points.dim, len_scale=len_scale)
    impl = GSToolsSpatialCorrelatedField(exp_model)
    impl_test_cov_func(impl, exponential, random_points, n_terms_range=n_terms)


if __name__ == "__main__":
    test_field_mean_std_convergence(2)
    #test_cov_func_convergence(2)
