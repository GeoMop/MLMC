import numpy as np
import numpy.linalg as la
import scipy as sp
from sklearn.utils.extmath import randomized_svd


class FieldSet:
    """
    A set of cross correlated named correlated fields.
    Currently just a wrapper of a single named field.
    TODO, questions:
    - mu, and sigma for individual fields can not be set via set_points method
      as we need the field set to be shared for different meshes. We have to introduce a kind of continuous
      field that can be evaluated at any point.
    - we can not make hard wired model for any field set so this should be a base class that
      defines an interface and common methods but evaluation of the feild set is case dependent

    """

    def __init__(self, name, field):
        self._back_fields = [field]
        # List of backend fields that are independent. We always have to
        # decompose resulting fields as functions of the independent fields.

        self.field_names = [name]

    def names(self):
        """
        Generator for field names.
        :return:
        """
        return self.field_names

    def set_points(self, points):
        for field in self._back_fields:
            field.set_points(points)

    def sample(self):
        """
        Return dictionary of sampled fields.
        :return: { 'field_name': sample, ...}
        """
        result = dict()
        result[self.field_names[0]] = self._back_fields[0].sample()
        return result


class SpatialCorrelatedField:
    """
    Generating realizations of a spatially correlated random field F for a fixed set of points at X.
    E[F(x)]       = mu(x) 
    Cov_ij = Cov[x_i,x_j]  = E[(F(x_i) - mu(x))(F(x_j) - mu(x))]

    We assume stationary random field with covariance matrix Cov_ij:
        Cov_i,j = c(x_i - x_j)
    where c(X) is the "stationary covariance" function. We assume:
          c(X) = sigma^2 exp( -|X^t K X|^(alpha/2) )
    for spatially heterogeneous sigma(X) we consider particular non-stationary generalization:\
          Cov_i,i = sigma(x_i)*sigma(x_j) exp( -|X^t K X|^(alpha/2) ); X = x_i - x_j

    where:
        - sigma(X) is the standard deviance of the single uncorrelated value
        - K is a positive definite tensor with eigen vectors corresponding to
          main directions and eigen values equal to (1/l_i)^2, where l_i is correlation
          length in singel main direction.
        - alpha is =1 for "exponential" and =2 for "Gauss" correlation

    SVD decomposition:
        Considering first m vectors, such that lam(m)/lam(0) <0.1

    Example:
    ```
        field = SpatialCorrelatedField(corr_exp='exp', corr_length=1.5)
        X, Y = np.mgrid[0:1:10j, 0:1:10j]
        points = np.vstack([X.ravel(), Y.ravel()])
        field.set_points(points)
        sample = field.sample()

    ```
    """

    def __init__(self, corr_exp='gauss', dim=2, corr_length=1.0,
                 aniso_correlation=None, mu=0.0, sigma=1.0, log=False):
        """
        :param corr_exp: 'gauss', 'exp' or a float (should be >= 1)
        :param dim: dimension of the domain (size of point coords)
        :param corr_length: scalar, correlation length L > machine epsilon; tensor K = (1/L)^2
        :param aniso_correlation: 3x3 array; K tensor, overrides correlation length
        :param mu - mu field (currently just a constant), can override by set_points parameters
        :param sigma - sigma field (currently just a constant), can override by set_points parameters

        TODO: use kwargs and move set_points into constructor
        """
        self.dim = dim
        self.log = log

        if corr_exp == 'gauss':
            self.correlation_exponent = 2.0
        elif corr_exp == 'exp':
            self.correlation_exponent = 1.0
        else:
            self.correlation_exponent = float(corr_exp)

        if aniso_correlation is None:
            assert corr_length > np.finfo(float).eps
            self.correlation_tensor = np.eye(dim, dim) * (1 / (corr_length ** 2))
            self._max_corr_length = corr_length
        else:
            self.correlation_tensor = aniso_correlation
            self._max_corr_length = la.norm(aniso_correlation, ord=2)  # largest eigen value

        #### Attributes set through `set_points`.
        self.points = None
        # Evaluation points of the field.
        self.mu = mu
        # Mean in points. Or scalar.
        self.sigma = sigma
        # Standard deviance in points. Or scalar.

        ### Attributes computed in precalculation.
        self.cov_mat = None
        # Covariance matrix (dense).
        self._n_approx_terms = None
        # Length of the sample vector, number of KL (Karhunen-Loe?ve) expansion terms.
        self._cov_l_factor = None
        # (Reduced) L factor of the SVD decomposition of the covariance matrix.
        self._sqrt_ev = None
        # (Reduced) square roots of singular values.

    def set_points(self, points, mu=None, sigma=None):
        """
        :param points: N x d array. Points X_i where the field will be evaluated. d is the dimension.
        :param mu: Scalar or N array. Mean value of uncorrelated field: E( F(X_i)).
        :param sigma: Scalar or N array. Standard deviance of uncorrelated field: sqrt( E ( F(X_i) - mu_i )^2 )
        :return: None
        """
        points = np.array(points, dtype=float)
        assert len(points.shape) == 2
        assert points.shape[1] == self.dim
        self.n_points, self.dimension = points.shape
        self.points = points

        if mu is not None:
            self.mu = mu
        self.mu = np.array(self.mu, dtype=float)
        assert self.mu.shape == () or self.mu.shape == (len(points),)

        if sigma is not None:
            self.sigma = sigma
        self.sigma = np.array(self.sigma, dtype=float)
        assert self.sigma.shape == () or sigma.shape == (len(points),)

        self.cov_mat = None
        self._cov_l_factor = None

    def cov_matrix(self):
        """
        Setup dense covariance matrix for given set of points.
        :return: None.
        """
        assert self.points is not None, "Points not set, call set_points."
        self._points_bbox = box = (np.min(self.points, axis=0), np.max(self.points, axis=0))
        diameter = np.max(np.abs(box[1] - box[0]))
        self._relative_corr_length = self._max_corr_length / diameter

        # sigma_sqr_mat = np.outer(self.sigma, self.sigma.T)
        self._sigma_sqr_max = np.max(self.sigma) ** 2
        n_pt = len(self.points)
        print("len self.points", n_pt)
        self.cov_mat = np.empty( (n_pt, n_pt))
        corr_exp = self.correlation_exponent / 2.0
        exp_scale = - 1.0 / self.correlation_exponent

        for i_row in range(n_pt):
            pt = self.points[i_row]
            diff_row = self.points - pt
            len_sqr_row = np.sum(diff_row.dot(self.correlation_tensor) * diff_row, axis=-1)
            self.cov_mat[i_row, :] = np.exp(exp_scale * len_sqr_row ** corr_exp)
        return self.cov_mat

    def _eigen_value_estimate(self, m):
        """
        Estimate of the m-th eigen value of the covariance matrix.
        According to paper: Schwab, Thodor: KL Approximation  of Random Fields by ...
        However for small gamma the asimtotics holds just for to big values of 'm'.
        We rather need to find a semiempricial formula.
        greater
        :param m:
        :return:
        """
        assert self.cov_mat is not None
        d = self.dimension
        alpha = self.correlation_exponent
        gamma = self._relative_corr_length
        return self._sigma_sqr_max * (1.0 / gamma) ** (m ** (1.0 / d) + alpha) / sp.special.gamma(0.5 * m ** (1 / d))

    def svd_dcmp(self, precision=0.01, n_terms_range=(1, np.inf)):
        """
        Does decomposition of covariance matrix defined by set of points
        :param precision: Desired accuracy of the KL approximation, smaller eigen values are dropped.
        :param n_terms_range: (min, max) number of terms in KL expansion to use. The number of terms estimated from
        given precision is snapped to the given interval.

        truncated SVD:
         cov_mat = U*diag(ev) * V,
         cov_l_factor = U[:,0:m]*sqrt(ev[0:m])

        Note on number of terms:
        According to: C. Schwab and R. A. Todor: KL Approximation of Random Fields by Generalized Fast Multiploe Method
        the eigen values should decay as (Proposition 2.18):
            lambda_m ~ sigma^2 * ( 1/gamma ) **( m**(1/d) + alpha ) / Gamma(0.5 * m**(1/d) )
        where gamma = correlation length / domain diameter
        ans alpha is the correlation exponent. Gamma is the gamma function.
        ... should be checked experimantaly and generalized for sigma(X)

        :return:
        """
        if self.cov_mat is None:
            self.cov_matrix()

        if n_terms_range[0] >= self.n_points:
            U, ev, VT = np.linalg.svd(self.cov_mat)
            m = self.n_points
        else:
            range = list(n_terms_range)
            range[0] = max(1, range[0])
            range[1] = min(self.n_points, range[1])

            prec_range = (self._eigen_value_estimate(range[0]), self._eigen_value_estimate(range[1]))
            if precision < prec_range[0]:
                m = range[0]
            elif precision > prec_range[1]:
                m = range[1]
            else:
                f = lambda m: self._eigen_value_estimate(m) - precision
                m = sp.optmize.bisect(f, range[0], range[1], xtol=0.5, )

            m = max(m, range[0])
            threshold = 2 * precision
            # TODO: Test if we should cut eigen values by relative (like now) or absolute value
            while threshold > precision and m < range[1]:
                print("t: ", threshold, "m: ", m, precision, range[1])
                U, ev, VT = randomized_svd(self.cov_mat, n_components=m, n_iter=3, random_state=None)
                threshold = ev[-1] / ev[0]
                m = int(np.ceil(1.5 * m))

            m = len(ev)
            m = min(m, range[1])
        self.n_approx_terms = m
        self._sqrt_ev = np.sqrt(ev[0:m])
        self._cov_l_factor = U[:, 0:m].dot(sp.diag(self._sqrt_ev))
        return self._cov_l_factor, ev[0:m]

    def sample(self, uncorelated=None):
        """
        :param uncorelated: Random samples from standard normal distribution.
        :return: Random field evaluated in points given by 'set_points'.
        """

        if self._cov_l_factor is None:
            self.svd_dcmp()
        if uncorelated is None:
            uncorelated = np.random.normal(0, 1, self.n_approx_terms)
        else:
            assert uncorelated.shape == (self.n_approx_terms,)

        field = (self.sigma * self._cov_l_factor.dot(uncorelated)) + self.mu
        if self.log:
            return np.exp(field)
        else:
            return field


# =====================================================================
# Example:
"""
"""


