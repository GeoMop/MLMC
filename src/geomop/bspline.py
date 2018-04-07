"""
Module with classes representing various B-spline and NURBS curves and surfaces.
These classes provide just basic functionality:
- storing the data
- evaluation of XYZ for UV
- evaluation and xy<->uv functions accepting np.arrays,
- evaluation of derivatives
In future:
- use de Boor algorithm for evaluation of curves and surfaces
- serialization and deserialization using JSONdata - must make it an installable module
- implement degree increasing and knot insertion
"""


import numpy as np
import numpy.linalg as la
import copy

__author__ = 'Jan Brezina <jan.brezina@tul.cz>, Jiri Hnidek <jiri.hnidek@tul.cz>, Jiri Kopal <jiri.kopal@tul.cz>'


class SplineBasis:
    """
    Represents a spline basis for a given knot vector and degree.
    Provides canonical evaluation for the bases functions and their derivatives, knot vector lookup etc.
    """

    @classmethod
    def make_equidistant(cls, degree, n_intervals, knot_range=[0.0, 1.0]):
        """
        Returns spline basis for an eqidistant knot vector
        having 'n_intervals' subintervals.
        :param degree: degree of the spline basis
        :param n_intervals: Number of subintervals.
        :param knot_range: support of the spline, min and max valid 't'
        :return: SplineBasis object.
        """
        n = n_intervals + 2 * degree + 1
        knots = np.array((knot_range[0],) * n)
        diff = (knot_range[1] - knot_range[0]) / n_intervals
        for i in range(degree + 1, n - degree):
            knots[i] = (i - degree) * diff + knot_range[0]
        knots[-degree - 1:] = knot_range[1]
        return cls(degree, knots)


    @classmethod
    def make_from_packed_knots(cls, degree, knots):
        """
        Construct basis from the vector of packed knots.
        :param degree: Degree of the basis.
        :param knots: List of knots with their multiplicities, [ (knot, mult), ..]
        :return: SplineBasis object.
        """
        full_knots = [ q for q, mult in knots for i in range(mult)  ]
        return cls(degree, full_knots)


    def __init__(self, degree, knots):
        """
        Constructor of the spline basis.
        :param degree: Degree of Bezier polynomials >=0.
        :param knots: Numpy array of the knots including multiplicities.
        """
        assert degree >=0
        self.degree = degree

        # check free ends
        for i in range(self.degree):
            assert knots[i] == knots[i+1]
            assert knots[-i-1] == knots[-i-2]
        self.knots = np.array(knots)

        self.size = len(self.knots) - self.degree -1
        # Number of basis functions.


        self.knots_idx_range = [self.degree, len(self.knots) - self.degree - 1]
        # Range of knot indices corrsponding to the basis domain.

        self.domain = self.knots[self.knots_idx_range]
        # Support domain of the spline.

        self.domain_size = self.domain[1] - self.domain[0]
        # Size of the domain.

        self.n_intervals = self.size - self.degree
        # Number of subintervals ( assuming multiplicities only on ends. )

        # Set optimized functions for specific degrees.
        if self.degree == 2:
            self.eval_base_vector = self._eval_vector_deg_2
            self.eval_diff_base_vector = self._eval_diff_vector_deg_2


    def pack_knots(self):
        """
        :return: Packed knot vector, [ (knot, multiplicity), .. ]
        """
        last, mult = self.knots[0], 0
        packed_knots = []
        for q in self.knots:
            if q == last:
                mult+=1
            else:
                packed_knots.append( (last, mult) )
                last, mult = q, 1
        packed_knots.append((last,mult))
        return packed_knots


    def check(self, t, rtol = 1e-10):
        """
        Check that 't' is inside basis domain. Fix small perturbations.
        :param t:
        :return:
        """
        if self.domain[0] <= t <= self.domain[1]:
            return t
        else:
            tol = (np.abs(t)  + 1e-4)*rtol
            if not self.domain[0] - tol <= t <= self.domain[1] + tol:
                raise IndexError("Evaluate spline, t={}, out of domain: {}.".format(t, self.domain))

            if t < self.domain[0]:
                return self.domain[0]
            else:
                return self.domain[1]

            self.domain[0]

    def find_knot_interval(self, t):
        """
        Find the first non-empty knot interval containing the value 't'.
        i.e. knots[i] <= t < knots[i+1], where  knots[i] < knots[i+1]
        Returns I = i  - degree, which is the index of the first basis function
        nonzero in 't'.

        :param t:  float, must be within knots limits.
        :return: I
        """
        idx = np.searchsorted(self.knots[self.degree: -self.degree -1], [t], side='right')[0] - 1
        assert 0 <= idx <= self.n_intervals, "Evaluation out of spline domain; t: {} min: {} max: {}".format(t, self.knots[0], self.knots[-1])
        return min(idx, self.n_intervals - 1)   # deals with t == self.knots[-1]



    def _basis(self, deg, idx, t):
        """
        Recursive evaluation of basis function of given degree and index.

        :param deg: Degree of the basis function
        :param idx: Index of the basis function to evaluate.
        :param t: Point of evaluation.
        :return Value of the basis function.
        """

        if deg == 0:
            t_0 = self.knots[idx]
            t_1 = self.knots[idx + 1]
            value = 1.0 if t_0 <= t < t_1 else 0.0
            return value
        else:
            t_i = self.knots[idx]
            t_ik = self.knots[idx + deg]
            top = t - t_i
            bottom = t_ik - t_i
            if bottom != 0:
                value = top / bottom * self._basis(deg-1, idx, t)
            else:
                value = 0.0

            t_ik1 = self.knots[idx + deg + 1]
            t_i1 = self.knots[idx + 1]
            top = t_ik1 - t
            bottom = t_ik1 - t_i1
            if bottom != 0:
                value += top / bottom * self._basis(deg-1, idx+1, t)

            return value

    def fn_supp(self, i_base):
        """
        Support of the base function 'i_base'.
        :param i_base:
        :return: (t_min, t_max)
        """
        return (self.knots[i_base], self.knots[i_base + self.degree + 1])


    def eval(self, i_base, t):
        """
        :param i_base: Index of base function to evaluate.
        :param t: point in which evaluate
        :return: b_i(y)
        """
        assert 0 <= i_base < self.size
        if i_base == self.size -1 and t == self.domain[1]:
            return 1.0
        return self._basis(self.degree, i_base, t)


    def eval_diff(self, i_base, t):
        assert 0 <= i_base < self.size
        deg = self.degree
        i = i_base
        if i_base == self.size - 2 and t == self.domain[1]:
            return - deg  / (self.knots[i + deg + 1] - self.knots[i + 1])

        if i_base == self.size - 1 and t == self.domain[1]:
            return  deg  / (self.knots[i + deg] - self.knots[i])

        diff = 0.0
        if i > 0:
            B = self._basis(deg-1, i, t)
            diff += deg * B / (self.knots[i + deg] - self.knots[i])
        if i < self.size - 1:
            B = self._basis(deg-1, i + 1, t)
            diff -= deg * B / (self.knots[i + deg + 1] - self.knots[i + 1])

        return diff

    def make_linear_poles(self):
        """
        Return poles of basis functions to get a f(x) = x.
        :return:
        """
        poles= [ 0.0 ]
        for i in range(self.size-1):
            pp = poles[-1] + (self.knots[i + self.degree + 1] - self.knots[i  + 1]) / float(self.degree)
            poles.append(pp)
        return poles


    def eval_vector(self, i_base, t):
        """
        This function compute base function of B-Spline curve on given subinterval.
        :param i_int: Interval in which 't' belongs. Three nonzero basis functions on this interval are evaluated.
        :param t: Where to evaluate.
        :return: Numpy array of three values.
        """
        values = []
        for ib in range(i_base, i_base + self.degree + 1):
            values.append( self.eval(ib, t))
        return values


    def eval_diff_vector(self, i_base, t):
        """
        This function compute derivative of base function of B-Spline curve on given subinterval.
        :param i_int: Interval in which 't' belongs. Derivatives of the 3 nonzero basis functions on this interval are evaluated.
        :param t: Where to evaluate.
        :return: Numpy array of three values.
        """
        values = []
        for ib in range(i_base, i_base + self.degree + 1):
            values.append( self.eval_diff(ib, t))
        return values


    """
    Specializations.
    TODO:
    - Try usage of scipy evaluation, compare speed with optimized eval_vector and diff_eval_vector.
    - Generalize optimized evaluation of eval_vector (If scipy is not faster), De Boor algortihm, Hoschek 4.3.3.
    - Optimize eval and eval_diff - without recursion, based on combinatorGeneralize optimized evaluation of eval_vector (If scipy is not faster)
    """
    def _eval_vector_deg_2(self, i_int, t):
        """
        This function compute base function of B-Spline curve on given subinterval.
        :param i_int: Interval in which 't' belongs. Three nonzero basis functions on this interval are evaluated.
        :param t: Where to evaluate.
        :return: Numpy array of three values.
        Note: Keep code redundancy with 'diff' as optimization.
        """

        basis_values = np.zeros(3)

        tk1, tk2, tk3, tk4 = self.knots[i_int + 1 : i_int + 5]

        d31 = tk3 - tk1
        d32 = tk3 - tk2
        d42 = tk4 - tk2

        dt1 = t - tk1
        dt2 = t - tk2
        d3t = tk3 - t
        d4t = tk4 - t

        d31_d32 = d31 * d32
        d42_d32 = d42 * d32

        basis_values[0] = (d3t * d3t) / d31_d32
        basis_values[1] = ((dt1 * d3t) / d31_d32) + ((dt2 * d4t) / d42_d32)
        basis_values[2] = (dt2 * dt2) / d42_d32

        return basis_values


    def _eval_diff_vector_deg_2(self, i_int, t):
        """
        This function compute derivative of base function of B-Spline curve on given subinterval.
        :param i_int: Interval in which 't' belongs. Derivatives of the 3 nonzero basis functions on this interval are evaluated.
        :param t: Where to evaluate.
        :return: Numpy array of three values.
        Note: Keep code redundancy with 'diff' as optimization.
        """

        basis_values = np.zeros(3)

        tk1, tk2, tk3, tk4 = self.knots[i_int + 1: i_int + 5]

        d31 = tk3 - tk1
        d32 = tk3 - tk2
        d42 = tk4 - tk2

        dt1 = t - tk1
        dt2 = t - tk2
        d3t = tk3 - t
        d4t = tk4 - t

        d31_d32 = d31 * d32
        d42_d32 = d42 * d32

        basis_values[0] = -2 * d3t / d31_d32
        basis_values[1] = (d3t - dt1) / d31_d32 + (d4t - dt2) / d42_d32
        basis_values[2] = 2 * dt2 / d42_d32

        return basis_values


class Curve:
    """
    Defines a D-dim B-spline curve.
    """

    @classmethod
    def make_raw(cls, poles, knots, rational=False, degree=2):
        """
        Construct a B-spline curve.
        :param poles: Numpy array N x (D+r) of poles (control points). N is number of poles, D is dimension of the curve, 'r' is 1 for rational curves.
         For rational case, poles[:, D] are weights of the control points.
        :param knots, degree: See SplineBasis.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles.
        :param degree: Non-negative int
        """
        basis = SplineBasis(degree, knots)
        return cls(basis, poles, rational)

    def __init__(self, basis, poles, rational = False):
        """
        Construct a B-spline curve.
        :param poles: Numpy array N x (D+r) of poles (control points). N is number of poles, D is dimension of the curve, 'r' is 1 for rational curves.
         For rational case, poles[:, D] are weights of the control points.
        :param basis: SplineBasis object.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles.
        """

        self.basis = basis
        # Spline basis.

        self.poles = np.array(poles, dtype=float)  # N x D
        assert self.poles.shape[0] == self.basis.size
        # Spline poles.

        self.dim = len(poles[0]) - rational
        # Dimension of the curve.

        self.rational = rational
        # Indicator of rational B-spline (NURBS).

        if rational:
            # precomputations
            self._weights = poles[:, self.dim]
            self._poles = (poles[:, 0:self.dim].T * self._weights ).T


    def eval_local(self, t, it):
        """
        Evaluate a B-spline curve for parameter 't' with given knotinterval 'it'.
        :param t: Evaluation point.
        :param it: Index of knot subinterval (see doc of 'find_knot_interval')
        :return: D-dimensional numpy array. D - is dimension given by dimension of poles.
        """
        dt = self.basis.degree + 1
        t_base_vec = self.basis.eval_vector(it, t)

        if self.rational:
            top_value = np.inner(t_base_vec, self._poles[it: it + dt, :].T)
            bot_value = np.inner(t_base_vec, self._weights[it: it + dt])
            return top_value / bot_value
        else:
            return  np.inner(t_base_vec, self.poles[it: it + dt, :].T)


    def eval(self, t):
        """
        Evaluate a B-spline curve for paramater 't'. Check and fix range of 't'.
        :param t: Evaluation point.
        :return: D-dimensional numpy array. D - is dimension given by dimension of poles.
        TODO:
        - test evaluation for rational curves
        """
        t = self.basis.check(t)
        it = self.basis.find_knot_interval(t)
        return self.eval_local(t, it)


    def eval_array(self, t_points):
        """
        Evaluate in array of t-points.
        :param t_points: array N x float
        :return: Numpy array N x D, D is dimension of the curve.
        """
        return np.array( [ self.eval(t) for t in t_points] )


    def aabb(self):
        """
        Return Axes Aligned Bounding Box of the poles, which should be also bounding box of the curve itself.
        :return: ( min_corner, max_corner); Box corners are numpy arryas of dimension D.
        """
        return np.array( [np.amin(self.poles, axis=0), np.amax(self.poles, axis=0)] )


class Surface:
    """
    Defines D-dim B-spline surface.
    """

    @classmethod
    def make_raw(cls, poles, knots, rational=False, degree=(2,2)):
        """
        Construct a B-spline curve.
        :param poles: List of poles (control points) ( X, Y, Z ) or weighted points (X,Y,Z, w). X,Y,Z,w are floats.
                   Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: List of tuples (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                   and multiplicity is positive int. Total number of knots, i.e. sum of their multiplicities, must be
                   degree + N + 1, where N is number of poles.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles.
        :param degree: Non-negative int
        """
        u_basis = SplineBasis(degree[0], knots[0])
        v_basis = SplineBasis(degree[1], knots[1])
        return cls( (u_basis, v_basis), poles, rational)


    def __init__(self, basis, poles, rational=False):
        """
        Construct a B-spline surface.
        :param poles: Numpy array Nu x Nv x (D+r) of poles (control points).
            Nu and Nv are sizes of u_basis, v_basis respectively.
            D is dimension of the surface, 'r' is 1 for rational surfaces.
            For rational case, poles[:, :, D] are weights of the control points.
        :param basis: (u_basis, v_basis) SplineBasis objects for U and V parameter axis.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles.
        """
        self.u_basis, self.v_basis = basis
        # Surface basis for U and V axis.

        self.poles = np.array(poles, dtype=float)
        self.dim = len(self.poles[0,0,:]) - rational
        # Surface dimension, D.


        # Surface poles matrix: Nu x Nv x (D+r)
        assert self.poles.shape == (self.u_basis.size, self.v_basis.size, self.dim + rational)

        self.rational = rational
        # Rational surface indicator.
        if rational:
            # precomputations
            self._weights = poles[:, :, self.dim]
            self._poles = (poles[:, :, 0:self.dim].T * self._weights.T ).T




    def eval_local(self, u, v, iu, iv):
        """
        Evaluate a B-spline surface for paramaters u,v with given knot subintervals.
        :param u, v: Evaluation point.
        :param iu, iv: Knot subintervals of u, v.
        :return: D-dimensional numpy array. D - is dimension given by dimension of poles.
        """
        du = self.u_basis.degree + 1
        dv = self.v_basis.degree + 1
        u_base_vec = self.u_basis.eval_vector(iu, u)
        v_base_vec = self.v_basis.eval_vector(iv, v)

        if self.rational:
            top_value = np.inner(u_base_vec, self._poles[iu: iu + du, iv: iv + dv, :].T)
            top_value = np.inner(top_value, v_base_vec)
            bot_value = np.inner(u_base_vec, self._weights[iu: iu + du, iv: iv + dv].T)
            bot_value = np.inner(bot_value, v_base_vec)
            return top_value / bot_value
        else:
            #print("u: {} v: {} p: {}".format(u_base_vec.shape, v_base_vec.shape, self.poles[iu: iu + du, iv: iv + dv, :].shape))
            # inner product sums along the last axis of its parameters
            value = np.inner(u_base_vec, self.poles[iu: iu + du, iv: iv + dv, :].T )
            #print("val: {}".format(value.shape))
            return np.inner( value, v_base_vec)



    def eval(self, u, v):
        """
        Evaluate a B-spline surface for paramaters u,v. Check and fix range of 'u, v'.
        :param u, v: Evaluation point.
        :return: D-dimensional numpy array. D - is dimension given by dimension of poles.
        TODO:
        - test evaluation for rational curves
        """
        u = self.u_basis.check(u)
        v = self.v_basis.check(v)
        iu = self.u_basis.find_knot_interval(u)
        iv = self.v_basis.find_knot_interval(v)
        return self.eval_local(u, v, iu, iv)






    def deep_copy(self):
        u_basis = copy.copy(self.u_basis)
        v_basis = copy.copy(self.v_basis)
        poles = copy.copy(self.poles)
        return Surface( ( u_basis, v_basis), poles)

    def eval_array(self, uv_points):
        """
        Evaluate in array of uv-points.
        :param uv_points: numpy array N x [u, v]
        :return: Numpy array N x D; D is dimension of the curve.
        """
        assert uv_points.shape[1] == 2
        return np.array( [ self.eval(u, v) for u, v in uv_points] )

    def aabb(self):
        """
        Return Axes Aligned Bounding Box of the poles, which should be also bounding box of the curve itself.
        :return: ( min_corner, max_corner); Box corners are numpy arryas of dimension D.
        TODO: test
        """
        return np.array( [np.amin(self.poles, axis=(0,1)), np.amax(self.poles, axis=(0,1))] )


class Z_Surface:
    """
    Simplified B-spline surface that use just linear or bilinear transform between XY and UV.
    """
    def __init__(self, xy_quad, z_surface):
        """
        Construct a surface given by the 1d surface for the Z coordinate and XY quadrilateral
        for the bilinear UV <-> XY mapping.
        :param xy_quad: np array N x 2
            Four or three points, determining bilinear or linear mapping, respectively.
            Four points giving XY coordinates for the uv corners: (0,1), (0,0), (1,0),  (1,1)
            Three points giving XY coordinates for the uv corners:  (0,1), (0,0), (1,0)
            Linear case is also detected for the four points.
        :param z_surface: 1D Surface object.
        """
        assert z_surface.dim == 1
        self.dim = 3
        # Fixed surface dimension.

        self.z_surface = z_surface
        # Underlaying 1d surface object for Z coord evaluation.

        self.u_basis = z_surface.u_basis
        self.v_basis = z_surface.v_basis
        # Basis for UV directions.


        self._reset_transform_xy(xy_quad)
        self._reset_transform_z()

        self.orig_quad = self.quad
        # Boundary quadrilateral.

        # Set further private attributes, see comment there:
        # _z_mat, _have_z_mat, _xy_shift, _mat_xy_to_uv, _mat_uv_to_xy

    def make_full_surface(self):
        """
        Return representation of the surface by the 3d Surface object.
        Compute redundant XY poles.
        :return: Surface.
        """
        basis = (self.z_surface.u_basis, self.z_surface.v_basis)

        u = basis[0].make_linear_poles()
        v = basis[1].make_linear_poles()
        V, U = np.meshgrid(v,u)
        uv_poles_vec = np.stack([U.ravel(), V.ravel()], axis=1)
        xy_poles = self.uv_to_xy(uv_poles_vec).reshape(U.shape[0], U.shape[1], 2)
        z_poles = self.z_surface.poles.copy()
        if self._have_z_mat:
            z_poles *= self._z_mat[0]
            z_poles += self._z_mat[1]
        poles = np.concatenate( (xy_poles, z_poles), axis = 2 )

        return Surface(basis, poles)

    def _reset_transform_z(self):
        self.z_mat = self._z_mat = np.array([1.0, 0.0])
        # [ z_scale, z_shift ]

        self._have_z_mat = False
        # Indicate that we have z_mat different from identity.
        # Optimization for array evaluation methods. z_mat must be set anyway.

    def _reset_transform_xy(self, xy_quad=None):
        """
        Set XY transform according to given domain quadrilateral (or triangle for linear mapping case).
        :param xy_quad: np array N x 2
            Four or three points, determining bilinear or linear mapping, respectively.
            Four points giving XY coordinates for the uv corners: (0,1), (0,0), (1,0),  (1,1)
            Three points giving XY coordinates for the uv corners:  (0,1), (0,0), (1,0)
            Linear case is also detected for the four points.

            If no xy_quad is given, we reset to the quad used in constructor.
        :return: None
        """

        self._xy_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Build envelope quadrilateral polygon in XY plane.
        if xy_quad is None:
            xy_quad = self.orig_quad
        xy_quad = np.array(xy_quad, dtype=float)
        assert xy_quad.shape[0] in [3, 4], "Three or four points must be given."
        assert xy_quad.shape[1] == 2

        v11 = xy_quad[0] + xy_quad[2] - xy_quad[1]
        if xy_quad.shape[0] == 3:
            xy_quad = np.concatenate( (xy_quad, v11[None, :]), axis = 0)

        if np.allclose(xy_quad[3], v11):
            # linear case
            self.quad = xy_quad
            self._xy_shift = self.quad[1]
            v_vec = self.quad[0] - self.quad[1]
            u_vec = self.quad[2] - self.quad[1]
            self._mat_uv_to_xy = np.column_stack((u_vec, v_vec))
            self._mat_xy_to_uv = la.inv(self._mat_uv_to_xy)

            self.xy_to_uv = self._linear_xy_to_uv
            self.uv_to_xy = self._linear_uv_to_xy

        else:
            # bilinear case
            self.quad = xy_quad
            self.xy_to_uv = self._bilinear_xy_to_uv
            self.uv_to_xy = self._bilinear_uv_to_xy


    def transform(self, xy_mat, z_mat = None ):
        """
        Set XY a Z transform of the Z-surface by arbitrary XY linear transform and Z linear transform.
        Combine the prescribed transform with the original XY transform, which represents original grid.

        :param xy_mat: np array, 2 rows 3 cols, last column is xy shift
        :param z_shift: np.array, [ z_scale, z_shift]
        :return: None
        """
        if xy_mat is not None:
            self._reset_transform_xy()
            surf_center = self.center()
            xy_mat = np.array(xy_mat)
            assert xy_mat.shape == (2, 3)

            self._xy_mat = xy_mat
            self._mat_uv_to_xy = np.dot(xy_mat[0:2,0:2], self._mat_uv_to_xy)
            quad_center = surf_center[0:2]
            self._xy_shift = xy_mat[0:2, 2] + np.dot(xy_mat[0:2,0:2], (self._xy_shift - quad_center)) + quad_center
            self._mat_xy_to_uv = la.inv(self._mat_uv_to_xy)
            # transform quad
            self.quad = np.dot(self.quad - quad_center, xy_mat[0:2,0:2].T) + xy_mat[0:2, 2] + quad_center


        if z_mat is not None:
            self._reset_transform_z()
            surf_center = self.center()
            z_mat = np.array(z_mat)
            assert z_mat.shape == (2,)

            self.z_mat = z_mat
            self._z_mat[0] = z_mat[0]
            z_center = surf_center[2]
            self._z_mat[1] = z_mat[1] + (1 - self._z_mat[0]) * z_center
            self._have_z_mat = True

    def get_transform(self):
        """
        Return additional transform applied through the 'transform' method.
        :return:
        """
        return (self._xy_mat, self.z_mat)

    def apply_z_transform(self):
        """
        Make copy of the bs.Surface for Z coordinates and  apply current Z transform to the poles of the copy.
        Reset the Z transform.
        :return:
        """
        if self._have_z_mat:
            self.z_surface = self.z_surface.deep_copy()
            self.z_surface.poles *= self._z_mat[0]
            self.z_surface.poles += self._z_mat[1]
            self._z_mat = np.array([1.0, 0.0])
            self._have_z_mat = False


    def get_copy(self):
        """
        Returns a copy of the Z_surface with own Z and XY transforms, but shared
        bs.Surface for Z coordinates (Z poles).
        :return: A copy of the Z_Surface object.
        """
        return copy.copy(self)


    """
    def uv_to_xy(self, uv_points):

    Abstract method. Converts array of uv points to array of xy points.
    :param uv_points: numpy array N x [u,v]
    :return: numpy array N x [x,y]
    """
    def _linear_uv_to_xy(self, uv_points):
        assert uv_points.shape[1] == 2, "Size: {}".format(uv_points.shape)
        return ( np.dot(uv_points, self._mat_uv_to_xy.T) + self._xy_shift)


    def _bilinear_uv_to_xy(self, uv_points):
        assert uv_points.shape[1] == 2, "Size: {}".format(uv_points.shape)
        xy_list = []
        for u,v in uv_points:
            weights = np.array([ (1-u)*v, (1-u)*(1-v), u*(1-v), u*v ])
            xy_list.append( self.quad.T.dot(weights) )
        return np.array(xy_list)

    """
    def xy_to_uv(self, xy_points):

    Abstract method. Converts array of xy points to array of uv points.
    :param xy_points: numpy array N x [x,y]
    :return: numpy array N x [u,v]
    """
    def _linear_xy_to_uv(self, xy_points):
        # assert xy_points.shape[0] == 2
        assert xy_points.shape[1] == 2
        return  np.dot((xy_points - self._xy_shift), self._mat_xy_to_uv.T)


    def _bilinear_xy_to_uv(self, xy_points):
        assert xy_points.shape[1] == 2
        assert False, "Not implemented yet."


    def eval(self, u, v):
        """
        Evaluate a B-spline surface for paramaters u,v.
        :param u, v: Evaluation point.
        :return: D-dimensional numpy array. D - is dimension given by dimension of poles.
        """
        z = self.z_surface.eval(u, v)
        z = self._z_mat[0] * z + self._z_mat[1]
        uv_points = np.array([[u, v]])
        x, y = self.uv_to_xy( uv_points )[0]
        return np.array( [x, y, z] )


    def eval_array(self, uv_points):
        """
        Evaluate a B-spline surface in array of UV points.
        :param uv_points: numpy array N x [u, v]
        :return: array N x D; D - is dimension given by dimension of poles.
        """
        assert uv_points.shape[1] == 2
        z_points = self.z_surface.eval_array(uv_points)
        if self._have_z_mat:
            z_points *= self._z_mat[0]
            z_points += self._z_mat[1]

        xy_points = self.uv_to_xy(uv_points)
        return np.concatenate( (xy_points, z_points), axis = 1)


    def eval_xy_array(self, xy_points):
        """
        Evaluate a B-spline surface in array of XY points.
        :param xy_points: numpy array N x [x, y]
        :return: array N x D; D - is dimension given by dimension of poles.
        """
        uv_points  = self.xy_to_uv(xy_points)
        return self.eval_array(uv_points)


    def z_eval_array(self, uv_points):
        """
        Evaluate just Z coordinate for array of UV points.
        :param uv_points: numpy array N x [u, v]
        :return: array N x D; D - is dimension given by dimension of poles.
        """
        assert uv_points.shape[1] == 2
        z_points = self.z_surface.eval_array(uv_points)
        if self._have_z_mat:
            z_points *= self._z_mat[0]
            z_points += self._z_mat[1]
        return z_points.reshape(-1)


    def z_eval_xy_array(self, xy_points):
        """
        Evaluate just Z coordinate for array of XY points.
        :param uv_points: numpy array N x [x, y]
        :return: array N x D; D - is dimension given by dimension of poles.
        """
        uv_points = self.xy_to_uv(xy_points)
        return self.z_eval_array(uv_points)


    def center(self):
        """
        Evaluate surface in center of UV square.
        :return: (X, Y, Z)
        """
        return self.eval_array( np.array([ [0.5, 0.5] ]))[0]


    def aabb(self):
        xyz_box = np.empty( (2, 3) )
        xyz_box[0, 0:2] = np.amin(self.quad, axis=0)
        xyz_box[1, 0:2] = np.amax(self.quad, axis=0)
        xyz_box[:, 2] = self._z_mat[0] * self.z_surface.aabb()[:, 0] + self._z_mat[1]
        return xyz_box



class GridNotInShapeExc(Exception):
    pass

class IrregularGridExc(Exception):
    pass


class GridSurface:

    step_tolerance = 1e-5
    # relative step_tolerance of 


    @staticmethod
    def load(filename):
        """
        Load the grid surface from file
        :param filename:
        :return: GridSurface object.
        """
        point_seq = np.loadtxt(filename)
        assert min(point_seq.shape) > 1
        return GridSurface(point_seq)

    """
    Can load and check grid of XYZ points and construct a
    Z-surface of degree 1 for them.
    """
    # TODO: calling transform lose mapping between original point grid and unit square, approximation can not be performed
    

    def __init__(self, point_seq, tolerance = 1e-6):
        """
        Construct the GridSurface from sequence of points.
        :param point_seq: N x 3 numpy array; organized as Nu x Nv grid.
            Nu, Nv are detected automaticaly, both must be greater then 1.
        :param step_tolerance: Tolerance between XY position given by envelope quad and actual point position. 
            Relative to the length of maximal side of the envelope quad.          
        """
        self.tolerance = tolerance
        
        assert point_seq.shape[1] == 3
        n_points = point_seq.shape[0]
        point_seq_xy = point_seq[:, 0:2]

        self.quad = None
        # Envelope quad - polygon, oriented counter clockwise.

        self.shape = None
        # Number of points along axis: Nu x Nv

        self._get_grid_corners(point_seq_xy)
        self._check_grid_regularity(point_seq_xy)

        self.z_scale = 1.0
        self.z_shift = 0.0

        self._uv_step = (1.0 / float(self.shape[0] - 1), 1.0 / float(self.shape[1] - 1))
        # Grid step in u, v direction respectively.

        self.grid_uvz = None
        # numpy array nu x nv x 3 with original XYZ values transformed to unit square.

        self._make_z_surface( point_seq)
        self._check_map()


    def _get_grid_corners(self, xy_points):
        n_points = len(xy_points)
        vtx_00 = xy_points[0, 0:2]
        vtx_dv = xy_points[1, 0:2]

        # detect grid shape
        v_step = vtx_dv - vtx_00
        step_tolerance = self.tolerance * la.norm(v_step, np.inf)
        for i in range(2, n_points):
            step = xy_points[i,:] - xy_points[i - 1, :]
            if la.norm(v_step - step, np.inf) > step_tolerance:
                break
        else:
            raise GridNotInShapeExc("End of the first row not detected.")
        nv = i
        nu = int(n_points / nv)
        if not n_points == nu * nv:
            raise GridNotInShapeExc("Not a Nu x Nv grid.")
        self.shape = (nu, nv)

        # make envelope quad
        vtx_01 = xy_points[nv - 1, :]
        vtx_10 = xy_points[-nv, :]
        vtx_11 = xy_points[-1, :]
        self.quad = np.array( [ vtx_01, vtx_00, vtx_10, vtx_11 ], dtype = float )
        self._orig_quad = self.quad

        # check that quad is parallelogram.
        diff = np.roll(self.quad, -1, axis = 0) - self.quad
        if not la.norm(diff[0] + diff[2]) < self.tolerance * la.norm(diff[0]) or \
            not la.norm(diff[1] + diff[3]) < self.step_tolerance * la.norm(diff[1]):
            raise GridNotInShapeExc("Grid XY envelope is not a parallelogram.")

        self._point_tol = self.tolerance * np.max( la.norm(diff, axis = 1) )
        self._u_step = diff[1] / (nu-1)      # v10 - v00
        self._v_step = diff[2] / (nv-1)      # v11 - v10


    def _check_grid_regularity(self, points_xy):
        # check regularity of the grid
        nu, nv = self.shape
        for i in range(nu * nv):
            pred_y = i - 1
            pred_x = i - nv
            if i%nv == 0:
                pred_y = -1
            if pred_x > 0 and not la.norm(points_xy[i, :] - points_xy[pred_x, :] - self._u_step) < self._point_tol:
                raise IrregularGridExc("Irregular grid in X direction, point %d"%i)
            if pred_y > 0 and not la.norm(points_xy[i, :] - points_xy[pred_y, :] - self._v_step) < self._point_tol:
                raise IrregularGridExc("Irregular grid in Y direction, point %d"%i)

    def _make_z_surface(self, points):
        nu, nv = self.shape
        u_basis = SplineBasis.make_equidistant(1, nu - 1)
        v_basis = SplineBasis.make_equidistant(1, nv - 1)



        poles_z = points[:, 2].reshape(nu, nv, 1)
        self.z_surface = Z_Surface(self.quad[0:3], Surface((u_basis, v_basis), poles_z) )
        self.u_basis = self.z_surface.u_basis
        self.v_basis = self.z_surface.v_basis

        uv_points = self.z_surface.xy_to_uv( points[:, 0:2] )
        grid_uv = uv_points.reshape(nu, nv, 2)
        self.grid_uvz = np.concatenate((grid_uv, poles_z), axis=2)

        # related bilinear Z-surface, all evaluations just call this object.


    def _check_map(self):
        # check that xy_to_uv works fine
        uv_quad = self.xy_to_uv(self.quad)
        #print( "Check quad: ", uv_quad )
        assert np.allclose( uv_quad, np.array([[0, 1], [0, 0], [1, 0], [1, 1]]) )

    # def _reset_transform(self):
    #     """
    #     Set identify transform just as after construction.
    #     :param xy_mat: np array, 2 rows 3 cols, last column is xy shift
    #     :param z_mat: [ z_scale, z_shift]
    #     :return:
    #     """
    #     self.z_scale = 1.0
    #     self.z_shift = 0.0
    #     self.z_surface._reset_transform(self._orig_quad)
    #     self.quad = self._orig_quad
    #     self._check_map()



    def transform(self, xy_mat, z_mat):
        """
        Set transform of the GridSurface to arbitrary XY linear transform and Z linear transform.
        :param xy_mat: np array, 2 rows 3 cols, last column is xy shift
        :param z_shift: [ z_scale, z_shift]
        :return: None
        Note:
        """
        self.z_surface.transform(xy_mat, z_mat)
        # transform quad
        self.quad = self.z_surface.uv_to_xy( np.array([[0, 1], [0, 0], [1, 0], [1, 1]]) )


    def xy_to_uv(self, xy_points):
        """
        :param xy_points: numpy matrix 2 rows N cols
        :return: matrix of UV coordinates
        """
        return self.z_surface.xy_to_uv(xy_points)


    def uv_to_xy(self, uv_points):
        """
        :param xy_points: numpy matrix 2 rows N cols
        :return: matrix of UV coordinates
        """
        return self.z_surface.uv_to_xy(uv_points)


    def eval_array(self, uv_points):
        xyz = self.z_surface.eval_array(uv_points)
        xyz[:, 2] *= self.z_scale
        xyz[:, 2] += self.z_shift
        return xyz


    def z_eval_xy_array(self, xy_points):
        """
        Return np array of z-values. Optimized version.
        :param points: np array N x 2 of XY cooridinates
        :return: array of Z values
        """
        assert xy_points.shape[1] == 2, "Size: {}".format(xy_points.shape)
        uv_points = self.z_surface.xy_to_uv(xy_points)
        return self.z_eval_array(uv_points)


    def z_eval_array(self, uv_points):
        """
        Return np array of z-values. Optimized version.
        :param points: np array N x 2 of XY cooridinates
        :return: array of Z values
        """

        assert uv_points.shape[1] == 2

        result = np.zeros(uv_points.shape[0])
        for i, uv in enumerate(uv_points):
            iuv = np.floor(uv / self._uv_step)
            iu = max(0, min(self.shape[0] - 2, int(iuv[0])))
            iv = max(0, min(self.shape[1] - 2, int(iuv[1])))
            iuv = np.array([iu, iv])

            uv_loc = uv / self._uv_step - iuv
            u_loc = np.array([1 - uv_loc[0], uv_loc[0]])
            v_loc = np.array([1 - uv_loc[1], uv_loc[1]])
            Z_mat = self.grid_uvz[iu: (iu + 2), iv: (iv + 2), 2]
            result[i] = self.z_surface.z_mat[0]*(u_loc.dot(Z_mat).dot(v_loc)) + self.z_surface.z_mat[1]
        return result


    def center(self):
        return self.eval_array( np.array([ [0.5, 0.5] ]))[0]


    def aabb(self):
        z_surf_box = self.z_surface.aabb()
        z_surf_box[:,2] *= self.z_scale
        z_surf_box[:, 2] += self.z_shift
        return z_surf_box


def make_function_grid(fn, nu, nv):
    """
    Make a grid of points on a graph of the function.
    :param fn: fn( [x, y] ) -> z
    :param nu: n-points in u-direction
    :param nv: n-points in v-direction
    :return: array of points: nu x nv x 3
    """
    X_grid = np.linspace(0, 1.0, nu)
    Y_grid = np.linspace(0, 1.0, nv)
    Y, X = np.meshgrid(Y_grid, X_grid)

    points_uv = np.stack([X.ravel(), Y.ravel()], 1)
    Z = np.apply_along_axis(fn, 1, points_uv)
    points = np.stack([X.ravel(), Y.ravel(), Z], 1)

    return points.reshape( (nu, nv, 3) )










