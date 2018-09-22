import numpy as np
import numpy.linalg as la
import scipy as sp
import copy
from sklearn.utils.extmath import randomized_svd



def kozeny_carman(porosity, m, factor, viscosity):
    """
    Kozeny-Carman law. Empirical relationship between porosity and conductivity.
    :param porosity: Porosity value.
    :param m: Power. Suitable values are 1 < m < 4
    :param factor: [m^2]
        E.g. 1e-7 ,   m = 3.48;  juta fibers
             2.2e-8 ,     1.46;  glass fibers
             1.8e-13,     2.89;  erruptive material
             1e-12        2.76;  erruptive material
             1.8e-12      1.99;  basalt
    :param viscosity: [Pa . s], water: 8.90e-4
    :return:
    """
    assert np.all(viscosity > 1e-10)
    porosity = np.minimum(porosity, 1-1e-10)
    porosity = np.maximum(porosity, 1e-10)
    cond = factor * porosity ** (2 + m) / (1 - porosity) ** 2 / viscosity
    cond = np.maximum(cond, 1e-15)
    return cond

def positive_to_range(exp, a, b):
    """
    Mapping a positive parameter 'exp' from the interval <0, \infty) to the interval <a,b).
    Suitable e.g. to generate meaningful porosity from a variable with lognormal distribution.
    :param exp: A positive parameter. (LogNormal distribution.)
    :param a, b: Range interval.
    """
    return  b * (1 - (b - a) / (b + (b - a) * exp))

class Field:
    def __init__(self, name, field=None, param_fields=[], regions=[]):
        """
        :param name: Name of the field.
        :param field: scalar (const field), or instance of SpatialCorrelatedField, or a callable
               for evaluation of the field from its param_fields.
        :param regions: Domain where field is sampled.
        :param param_fields: List of names of parameter fields, dependees.
        TODO: consider three different derived classes for: const, random and func fields.
        """
        self.correlated_field = None
        self.const = None
        self._func = field
        self.is_outer = True

        if type(regions) is str:
            regions = [regions]
        self.name = name
        if type(field) in [float, int]:
            self.const = field
            assert len(param_fields) == 0
        elif type(field) is SpatialCorrelatedField:
            self.correlated_field = field
            assert len(param_fields) == 0
        else:
            assert len(param_fields) > 0, field

            # check callable
            try:
                params = [ np.ones(2) for i in range(len(param_fields))]
                field(*params)
            except:
                raise Exception("Invalid field function for field: {}".format(name))
            self._func = field

        self.regions = regions
        self.param_fields = param_fields

    def set_points(self, points):
        """
        Internal method to set evaluation points. See Fields.set_points.
        """
        if self.const is not None:
            self._sample = self.const * np.ones(len(points))
        elif self.correlated_field is not None:
            #print("Set crr field, ", self.name, self.regions)
            self.correlated_field.set_points(points)
            self.correlated_field.svd_dcmp(n_terms_range=(10, 100))
        else:
            pass

    def sample(self):
        """
        Internal method to generate/compute new sample.
        :return:
        """
        if self.const is not None:
            return self._sample
        elif self.correlated_field is not None:
            self._sample = self.correlated_field.sample()
        else:
            params = [ pf._sample for pf in self.param_fields]
            self._sample = self._func(*params)
        return self._sample


class Fields:

    def __init__(self, fields):
        """
        Creates a new set of cross dependent random fields.
        Currently no support for cross-correlated random fields.
        A set of independent basic random fields must exist
        other fields can be dependent in deterministic way.

        :param fields: A list of dependent fields.

        Example:
        rf = SpatialCorrelatedField(log=True)
        Fields([
            Field('por_top', rf, regions='ground_0'),
            Field('porosity_top', positive_to_range, ['por_top', 0.02, 0.1], regions='ground_0'),
            Field('por_bot', rf, regions='ground_1'),
            Field('porosity_bot', positive_to_range, ['por_bot', 0.01, 0.05], regions='ground_1'),
            Field('conductivity_top', cf.kozeny_carman, ['porosity_top', 1, 1e-8, water_viscosity], regions='ground_0'),
            Field('conductivity_bot', cf.kozeny_carman, ['porosity_bot', 1, 1e-10, water_viscosity],regions='ground_1')
            ])

        TODO: use topological sort to fix order of 'fields'
        TODO: syntactic sugar for calculating with fields (like with np.arrays).
        """
        self.fields_orig = fields
        self.fields_dict = {}
        self.fields = []

        # Have to make a copy of the fields since we want to generate the samples in them
        # and the given instances of Field can be used by an independent FieldSet instance.
        for field in self.fields_orig:
            new_field = copy.copy(field)
            if new_field.param_fields:
                new_field.param_fields = [self._get_field_obj(field, new_field.regions) for field in new_field.param_fields]
            self.fields_dict[new_field.name] = new_field
            self.fields.append(new_field)

    def _get_field_obj(self, field_name, regions):
        """
        Get fields by name, replace constants by constant fields for unification.
        """
        if type(field_name) in [float, int]:
            const_field = Field("const_{}".format(field_name), field_name, regions=regions)
            self.fields.insert(0, const_field)
            self.fields_dict[const_field.name] = const_field
            return const_field
        else:
            assert field_name in self.fields_dict, "name: {} dict: {}".format(field_name, self.fields_dict)
            return self.fields_dict[field_name]

    @property
    def names(self):
        return self.fields_dict.keys()

    # def iterative_dfs(self, graph, start, path=[]):
    #     q = [start]
    #     while q:
    #         v = q.pop(0)
    #         if v not in path:
    #             path = path + [v]
    #             q = graph[v] + q
    #
    #     return path

    def set_outer_fields(self, outer):
        """
        Set fields that will be in a dictionary produced by FieldSet.sample() call.
        :param outer: A list of names of fields that are sampled.
        :return:
        """
        outer_set = set(outer)
        for f in self.fields:
            if f.name in outer_set:
                f.is_outer = True
            else:
                f.is_outer = False


    def set_points(self, points, region_ids=[], region_map={}):
        """
        Set mesh related data to fields.
        - set points for sample evaluation
        - translate region names to region ids in fields
        - create maps from region constraned point sets of fields to full point set
        :param points: np array of points for field evaluation
        :param regions: regions of the points;
               empty means no points for fields restricted to regions and all points for unrestricted fields
        :return:
        """
        self.n_elements = len(points)
        assert len(points) == len(region_ids)
        reg_points = {}
        for i, reg_id in enumerate(region_ids):
            reg_list = reg_points.get(reg_id, [])
            reg_list.append(i)
            reg_points[reg_id] = reg_list

        for field in self.fields:
            point_ids = []
            if field.regions:
                for reg in field.regions:
                    reg_id = region_map[reg]
                    point_ids.extend(reg_points.get(reg_id, []))
                field.set_points(points[point_ids])
                field.full_sample_ids = point_ids
            else:
                field.set_points(points)
                field.full_sample_ids = np.arange(self.n_elements)

    def sample(self):
        """
        Return dictionary of sampled fields.
        :return: { 'field_name': sample, ...}
        """
        result = {}
        for field in self.fields:
            sample = field.sample()
            if field.is_outer:
                result[field.name] = np.zeros(self.n_elements)
                result[field.name][field.full_sample_ids] = sample
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
        :param mu - mu field (currently just a constant)
        :param sigma - sigma field (currently just a constant)

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

        assert len(points.shape) >= 1
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

        #return self.n_points

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
            while threshold >= precision and m <= range[1]:
                #print("treshold: {} m: {} precision: {} max_m: {}".format(threshold,  m, precision, range[1]))
                U, ev, VT = randomized_svd(self.cov_mat, n_components=m, n_iter=3, random_state=None)
                threshold = ev[-1] / ev[0]
                m = int(np.ceil(1.5 * m))

            m = len(ev)
            m = min(m, range[1])

        #print("KL approximation: {} for {} points.".format(m, self.n_points))
        self.n_approx_terms = m
        self._sqrt_ev = np.sqrt(ev[0:m])
        self._cov_l_factor = U[:, 0:m].dot(sp.diag(self._sqrt_ev))
        self.cov_mat = None
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

        if not self.log:
            return field
        return np.exp(field)




