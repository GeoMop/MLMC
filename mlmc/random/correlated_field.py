import copy
import numpy as np
import numpy.linalg as la
import numpy.random as rand
import scipy as sp
from sklearn.utils.extmath import randomized_svd
import warnings
warnings.simplefilter('always', DeprecationWarning)
import gstools


def kozeny_carman(porosity, m, factor, viscosity):
    """
    Kozeny-Carman law. Empirical relationship between porosity and conductivity.

    :param porosity: Porosity value.
    :param m: Power. Suitable values are 1 < m < 4
    :param factor: Factor [m^2]. Examples:
        1e-7 , m = 3.48;  juta fibers
        2.2e-8 , m = 1.46;  glass fibers
        1.8e-13, m = 2.89;  erruptive material
        1e-12 , m = 2.76;  erruptive material
        1.8e-12, m = 1.99;  basalt
    :param viscosity: Fluid viscosity [Pa.s], e.g., water: 8.90e-4
    :return: Conductivity
    """
    assert np.all(viscosity > 1e-10)
    porosity = np.minimum(porosity, 1-1e-10)
    porosity = np.maximum(porosity, 1e-10)
    cond = factor * porosity ** (2 + m) / (1 - porosity) ** 2 / viscosity
    cond = np.maximum(cond, 1e-15)
    return cond


def positive_to_range(exp, a, b):
    """
    Map a positive parameter 'exp' from <0, ∞) to <a, b>.

    :param exp: Positive parameter (e.g., lognormal variable)
    :param a: Lower bound of target interval
    :param b: Upper bound of target interval
    :return: Mapped value in [a, b)
    """
    return b * (1 - (b - a) / (b + (b - a) * exp))


class Field:
    def __init__(self, name, field=None, param_fields=[], regions=[]):
        """
        Initialize a Field object.

        :param name: Name of the field
        :param field: Scalar (const), RandomFieldBase, or callable function
        :param param_fields: List of dependent parameter fields
        :param regions: List of region names where the field is defined
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
        elif isinstance(field, RandomFieldBase):
            self.correlated_field = field
            assert len(param_fields) == 0
        else:
            assert len(param_fields) > 0, field
            try:
                params = [np.ones(2) for i in range(len(param_fields))]
                field(*params)
            except:
                raise Exception("Invalid field function for field: {}".format(name))
            self._func = field

        self.regions = regions
        self.param_fields = param_fields

    def set_points(self, points):
        """
        Set points for field evaluation.

        :param points: Array of points where the field will be evaluated
        """
        if self.const is not None:
            self._sample = self.const * np.ones(len(points))
        elif self.correlated_field is not None:
            self.correlated_field.set_points(points)
            if type(self.correlated_field) is SpatialCorrelatedField:
                self.correlated_field.svd_dcmp(n_terms_range=(10, 100))
        else:
            pass

    def sample(self):
        """
        Generate or compute a new sample of the field.

        :return: Sample values of the field
        """
        if self.const is not None:
            return self._sample
        elif self.correlated_field is not None:
            self._sample = self.correlated_field.sample()
        else:
            params = [pf._sample for pf in self.param_fields]
            self._sample = self._func(*params)
        return self._sample


class Fields:
    def __init__(self, fields):
        """
        Create a set of cross-dependent random fields.

        :param fields: List of Field objects
        """
        self.fields_orig = fields
        self.fields_dict = {}
        self.fields = []

        for field in self.fields_orig:
            new_field = copy.copy(field)
            if new_field.param_fields:
                new_field.param_fields = [self._get_field_obj(f, new_field.regions)
                                          for f in new_field.param_fields]
            self.fields_dict[new_field.name] = new_field
            self.fields.append(new_field)

    def _get_field_obj(self, field_name, regions):
        """
        Get Field object by name or create constant field.

        :param field_name: Field name or constant
        :param regions: Regions of the field
        :return: Field object
        """
        if type(field_name) in [float, int]:
            const_field = Field("const_{}".format(field_name), field_name, regions=regions)
            self.fields.insert(0, const_field)
            self.fields_dict[const_field.name] = const_field
            return const_field
        else:
            assert field_name in self.fields_dict
            return self.fields_dict[field_name]

    def set_outer_fields(self, outer):
        """
        Set fields to be included in the sampled dictionary.

        :param outer: List of outer field names
        """
        outer_set = set(outer)
        for f in self.fields:
            f.is_outer = f.name in outer_set

    def set_points(self, points, region_ids=[], region_map={}):
        """
        Assign evaluation points to each field.

        :param points: Array of points for field evaluation
        :param region_ids: Optional array of region ids for each point
        :param region_map: Mapping from region name to region id
        """
        self.n_elements = len(points)
        reg_points = {}
        for i, reg_id in enumerate(region_ids):
            reg_points.setdefault(reg_id, []).append(i)

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
        Sample all outer fields.

        :return: Dictionary with field names as keys and sampled arrays as values
        """
        result = {}
        for field in self.fields:
            sample = field.sample()
            if field.is_outer:
                shape = (self.n_elements, 3) if field.name == "cond_tn" else self.n_elements
                result[field.name] = np.zeros(shape)
                result[field.name][field.full_sample_ids] = sample
        return result


class RandomFieldBase:
    """
    Base class for generating spatially correlated random fields.

    Random field F(x) with mean E[F(x)] = mu(x) and covariance Cov[x_i,x_j].
    Stationary covariance: Cov_ij = sigma^2 * exp(-|X^T K X|^(alpha/2)),
    X = x_i - x_j.
    Supports optional non-stationary variance sigma(X).
    """

    def __init__(self, corr_exp='gauss', dim=2, corr_length=1.0,
                 aniso_correlation=None, mu=0.0, sigma=1.0, log=False, **kwargs):
        """
        Initialize a random field.

        :param corr_exp: 'gauss', 'exp', or float >=1 (correlation exponent)
        :param dim: Dimension of the domain
        :param corr_length: Scalar correlation length
        :param aniso_correlation: Optional anisotropic 3x3 correlation tensor
        :param mu: Mean (scalar or array)
        :param sigma: Standard deviation (scalar or array)
        :param log: If True, output field is exponentiated
        """
        self.dim = dim
        self.log = log

        if corr_exp == 'gauss':
            self.correlation_exponent = 2.0
        elif corr_exp == 'exp':
            self.correlation_exponent = 1.0
        else:
            self.correlation_exponent = float(corr_exp)

        self._corr_length = corr_length
        if aniso_correlation is None:
            assert corr_length > np.finfo(float).eps
            self.correlation_tensor = np.eye(dim, dim) * (1 / (corr_length ** 2))
            self._max_corr_length = corr_length
        else:
            self.correlation_tensor = aniso_correlation
            self._max_corr_length = la.norm(aniso_correlation, ord=2)

        self.points = None
        self.mu = mu
        self.sigma = sigma
        self._initialize(**kwargs)

    def _initialize(self, **kwargs):
        """Implementation-specific initialization. To be overridden in subclasses."""
        raise NotImplementedError()

    def set_points(self, points, mu=None, sigma=None):
        """
        Set points for field evaluation.

        :param points: Array of points (N x dim)
        :param mu: Optional mean at points
        :param sigma: Optional standard deviation at points
        """
        points = np.array(points, dtype=float)
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
        assert self.sigma.shape == () or self.sigma.shape == (len(points),)

    def _set_points(self):
        """Optional internal method to update points. Can be overridden."""
        pass

    def sample(self):
        """
        Generate a realization of the random field.

        :return: Array of field values at set points
        """
        field = self._sample()
        field = self.sigma * field + self.mu
        return np.exp(field) if self.log else field

    def _sample(self, uncorrelated):
        """
        Implementation-specific sample generation. To be overridden.

        :param uncorrelated: Array of uncorrelated standard normal samples
        :return: Field sample
        """
        raise NotImplementedError()


class SpatialCorrelatedField(RandomFieldBase):
    """
    Generate spatially correlated fields using covariance matrix and KL decomposition.
    """

    def _initialize(self, **kwargs):
        """Initialization specific to SVD/KL-based spatial correlation."""
        self.cov_mat = None
        self._n_approx_terms = None
        self._cov_l_factor = None
        self._sqrt_ev = None

    def _set_points(self):
        self.cov_mat = None
        self._cov_l_factor = None

    def cov_matrix(self):
        """
        Compute dense covariance matrix for current points.

        :return: Covariance matrix
        """
        assert self.points is not None, "Points not set, call set_points."
        self._points_bbox = (np.min(self.points, axis=0), np.max(self.points, axis=0))
        diameter = np.max(np.abs(self._points_bbox[1] - self._points_bbox[0]))
        self._relative_corr_length = self._max_corr_length / diameter
        self._sigma_sqr_max = np.max(self.sigma) ** 2

        n_pt = len(self.points)
        self.cov_mat = np.empty((n_pt, n_pt))
        corr_exp = self.correlation_exponent / 2.0

        for i_row in range(n_pt):
            pt = self.points[i_row]
            diff_row = self.points - pt
            len_sqr_row = np.sum(diff_row.dot(self.correlation_tensor) * diff_row, axis=-1)
            self.cov_mat[i_row, :] = np.exp(-len_sqr_row ** corr_exp)

        return self.cov_mat

    def _eigen_value_estimate(self, m):
        """
        Semi-empirical estimate of the m-th eigenvalue of covariance matrix.

        :param m: Eigenvalue index
        :return: Estimated eigenvalue
        """
        d = self.dimension
        alpha = self.correlation_exponent
        gamma = self._relative_corr_length
        return self._sigma_sqr_max * (1.0 / gamma) ** (m ** (1.0 / d) + alpha) / sp.special.gamma(0.5 * m ** (1 / d))

    def svd_dcmp(self, precision=0.01, n_terms_range=(1, np.inf)):
        """
        Perform truncated SVD for Karhunen-Loeve decomposition.

        :param precision: Desired accuracy
        :param n_terms_range: Min/max number of KL terms
        :return: (_cov_l_factor, singular values)
        """
        if self.cov_mat is None:
            self.cov_matrix()

        if n_terms_range[0] >= self.n_points:
            U, ev, VT = np.linalg.svd(self.cov_mat)
            m = self.n_points
        else:
            range_vals = [max(1, n_terms_range[0]), min(self.n_points, n_terms_range[1])]
            prec_range = (self._eigen_value_estimate(range_vals[0]), self._eigen_value_estimate(range_vals[1]))
            if precision < prec_range[0]:
                m = range_vals[0]
            elif precision > prec_range[1]:
                m = range_vals[1]
            else:
                f = lambda m: self._eigen_value_estimate(m) - precision
                m = sp.optimize.bisect(f, range_vals[0], range_vals[1], xtol=0.5)
            m = max(m, range_vals[0])
            threshold = 2 * precision
            while threshold >= precision and m <= range_vals[1]:
                U, ev, VT = randomized_svd(self.cov_mat, n_components=m, n_iter=3, random_state=None)
                threshold = ev[-1] / ev[0]
                m = int(np.ceil(1.5 * m))

            m = min(len(ev), range_vals[1])

        self.n_approx_terms = m
        self._sqrt_ev = np.sqrt(ev[:m])
        self._cov_l_factor = U[:, :m].dot(np.diag(self._sqrt_ev))
        self.cov_mat = None
        return self._cov_l_factor, ev[:m]

    def _sample(self):
        """
        Generate a field realization using KL decomposition.

        :return: Field sample array
        """
        if self._cov_l_factor is None:
            self.svd_dcmp()
        uncorrelated = np.random.normal(0, 1, self.n_approx_terms)
        return self._cov_l_factor.dot(uncorrelated)


class GSToolsSpatialCorrelatedField(RandomFieldBase):
    """
    Spatially correlated random field generator using GSTools.

    This class acts as an adapter between :mod:`gstools` and the MLMC
    random field interface (:class:`mlmc.random.random_field_base.RandomFieldBase`).
    It supports 1D, 2D, and 3D random fields with optional logarithmic transformation,
    and can generate fields on both structured and unstructured grids.
    """

    def __init__(self, model, mode_no=1000, log=False, sigma=1, seed=None, mode=None, structured=False):
        """
        Initialize a spatially correlated random field generator.

        :param model: Covariance model instance (subclass of ``gstools.covmodel.CovModel``)
            defining the spatial correlation structure.
        :param mode_no: Number of Fourier modes used in the random field generation.
            Default is 1000.
        :param log: If True, applies an exponential transformation to obtain
            a lognormal field. Default is False.
        :param sigma: Standard deviation scaling factor applied to the generated field.
            Default is 1.
        :param seed: Random seed for reproducibility. Default is None.
        :param mode: Sampling mode for GSTools SRF. Use "fft" for structured grids or
            None for unstructured. Default is None.
        :param structured: If True, assumes a structured grid for field evaluation.
            Default is False.
        """
        self.model = model
        self.mode_no = mode_no
        if mode == "fft":
            self.srf = gstools.SRF(model, mode="fft", seed=seed)
        else:
            self.srf = gstools.SRF(model, mode_no=mode_no, seed=seed)
        self.mu = self.srf.mean
        self.sigma = sigma
        self.dim = model.dim
        self.log = log
        self.structured = structured

    def change_srf(self, seed):
        """
        Reinitialize the GSTools random field with a new random seed.

        :param seed: Random seed used to reinitialize the underlying
            :class:`gstools.SRF` instance.
        :return: None
        """
        self.srf = gstools.SRF(self.model, seed=seed, mode_no=self.mode_no)

    def random_field(self, seed=None):
        """
        Generate a raw random field realization (without scaling or transformation).

        :param seed: Optional random seed for reproducibility. Default is None.
        :return: numpy.ndarray
            Field values evaluated at the points defined by :meth:`set_points`.
        """
        if self.dim == 1:
            x = self.points
            x.reshape(len(x))
            field = self.srf((x,))
        elif self.dim == 2:
            x, y = self.points.T
            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)
            field = self.srf((x, y))
        else:
            x, y, z = self.points.T
            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)
            z = z.reshape(len(z), 1)

            if self.structured:
                field = self.srf([np.squeeze(x), np.squeeze(y), np.squeeze(z)], seed=seed)
                field = field.flatten()
            else:
                if seed is not None:
                    field = self.srf(self.points.T, seed=seed)
                else:
                    field = self.srf(self.points.T)
        return field

    def sample(self, seed=None):
        """
        Evaluate the scaled random field at the defined points.

        :param seed: Optional random seed for reproducibility. Default is None.
        :return: numpy.ndarray
            Field values evaluated at the defined points, scaled by ``sigma``
            and shifted by ``mu``. If ``log=True``, returns
            ``exp(sigma * field + mu)`` instead.
        """
        if not self.log:
            return self.sigma * self.random_field(seed) + self.mu
        return np.exp(self.sigma * self.random_field(seed) + self.mu)


class FourierSpatialCorrelatedField(RandomFieldBase):
    """
    Deprecated: Fourier-based spatial random field generator.

    Generates spatial random fields using a truncated Fourier series.
    Use GSToolsSpatialCorrelatedField instead.
    """

    def _initialize(self, **kwargs):
        """
        Initialization specific to Fourier-based spatial fields.

        :param mode_no: Number of Fourier modes (default 1000)
        """
        warnings.warn(
            "FourierSpatialCorrelatedField class is deprecated, use GSToolsSpatialCorrelatedField instead",
            DeprecationWarning
        )
        self.len_scale = self._corr_length * 2 * np.pi
        self.mode_no = kwargs.get("mode_no", 1000)

    def get_normal_distr(self):
        """
        Generate normal distributed random coefficients for Fourier modes.

        :return: Array of shape (2, mode_no)
        """
        Z = np.empty((2, self.mode_no))
        rng = self._get_random_stream()
        for i in range(2):
            Z[i] = rng.normal(size=self.mode_no)
        return Z

    def _sample_sphere(self, mode_no):
        """
        Uniformly sample directions on the unit sphere (dim=1,2,3).

        :param mode_no: Number of modes
        :return: Array of unit vectors (dim, mode_no)
        """
        coord = self._create_empty_k(mode_no)
        rng = self._get_random_stream()
        if self.dim == 1:
            ang1 = rng.random_sample(mode_no)
            coord[0] = 2 * np.around(ang1) - 1
        elif self.dim == 2:
            ang1 = rng.uniform(0.0, 2 * np.pi, mode_no)
            coord[0] = np.cos(ang1)
            coord[1] = np.sin(ang1)
        elif self.dim == 3:
            raise NotImplementedError("3D implementation see GSTools repo")
        return coord

    def gau(self, mode_no=1000):
        """
        Gaussian Fourier spectrum.

        :param mode_no: Number of modes
        :return: Array of wave vectors (dim, mode_no)
        """
        len_scale = self.len_scale * np.sqrt(np.pi / 4)
        if self.dim == 1:
            k = self._create_empty_k(mode_no)
            k[0] = self._get_random_stream().normal(0., np.pi / 2.0 / len_scale ** 2, mode_no)
        elif self.dim == 2:
            coord = self._sample_sphere(mode_no)
            rad_u = self._get_random_stream().random_sample(mode_no)
            rad = np.sqrt(np.pi) / len_scale * np.sqrt(-np.log(rad_u))
            k = rad * coord
        elif self.dim == 3:
            raise NotImplementedError("3D implementation see GSTools repo")
        return k

    def exp(self, mode_no=1000):
        """
        Exponential Fourier spectrum.

        :param mode_no: Number of modes
        :return: Array of wave vectors (dim, mode_no)
        """
        if self.dim == 1:
            k = self._create_empty_k(mode_no)
            k_u = self._get_random_stream().uniform(-np.pi / 2.0, np.pi / 2.0, mode_no)
            k[0] = np.tan(k_u) / self.len_scale
        elif self.dim == 2:
            coord = self._sample_sphere(mode_no)
            rad_u = self._get_random_stream().random_sample(mode_no)
            rad = np.sqrt(1.0 / rad_u ** 2 - 1.0) / self.len_scale
            k = rad * coord
        elif self.dim == 3:
            raise NotImplementedError("3D implementation see GSTools repo")
        return k

    def _create_empty_k(self, mode_no=None):
        """
        Helper to create empty Fourier mode array.

        :param mode_no: Number of modes
        :return: Empty array of shape (dim, mode_no)
        """
        return np.empty((self.dim, mode_no)) if mode_no is not None else np.empty(self.dim)

    def _get_random_stream(self, seed=None):
        """
        Return a random number generator.

        :param seed: Optional seed
        """
        return rand.RandomState(rand.RandomState(seed).randint(2 ** 16 - 1))

    def random_field(self):
        """
        Generate a random field using Fourier series.

        :return: Field values at points
        """
        # Prepare coordinates
        if self.dim == 1:
            x = self.points.reshape(len(self.points), 1)
        elif self.dim == 2:
            x, y = self.points.T
            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)
        else:
            x, y, z = self.points.T
            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)
            z = z.reshape(len(z), 1)

        normal_distr_values = self.get_normal_distr()
        k = self.gau(self.mode_no) if self.correlation_exponent == 2 else self.exp(self.mode_no)

        summed_modes = np.zeros(len(self.points))
        # Fourier summation (memory safe chunks could be implemented here)
        for i in range(self.mode_no):
            phase = np.sum(k[:, i] * self.points.T, axis=0)
            summed_modes += normal_distr_values[0, i] * np.cos(2*np.pi*phase) + normal_distr_values[1, i] * np.sin(2*np.pi*phase)

        return np.sqrt(1.0 / self.mode_no) * summed_modes

    def _sample(self):
        """
        Generate a Fourier-based random field realization.

        :return: Field values
        """
        return self.random_field()
