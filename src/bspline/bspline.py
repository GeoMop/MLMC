"""
Module with classes representing various B-spline and NURMS curves and surfaces.
These classes provide just basic functionality:
- storing the data
- evaluation of XYZ for UV
In future:
- evaluation and xy<->uv functions accepting np.arrays,
- serialization and deserialization using JSONdata - must make it an installable module
- use de Boor algorithm for evaluation of curves and surfaces
- evaluation of derivatives
- implement degree increasing and knot insertion
"""

"""
This module tries to approximate 2.5D array of terrain points
using B-Spline surface.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.linalg as la


__author__ = 'Jan Brezina <jan.brezina@tul.cz>, Jiri Hnidek <jiri.hnidek@tul.cz>, Jiri Kopal <jiri.kopal@tul.cz>'

    
class ParamError(Exception):
    pass

def check_matrix(mat, shape, values, idx=[]):
    '''
    Check shape and type of scalar, vector or matrix.
    :param mat: Scalar, vector, or vector of vectors (i.e. matrix). Vector may be list or other iterable.
    :param shape: List of dimensions: [] for scalar, [ n ] for vector, [n_rows, n_cols] for matrix.
    If a value in this list is None, the dimension can be arbitrary. The shape list is set fo actual dimensions
    of the matrix.
    :param values: Type or tuple of  allowed types of elements of the matrix. E.g. ( int, float )
    :param idx: Internal. Used to pass actual index in the matrix for possible error messages.
    :return:
    '''
    try:
        if len(shape) == 0:
            if not isinstance(mat, values):
                raise ParamError("Element at index {} of type {}, expected instance of {}.".format(idx, type(mat), values))
        else:
            if shape[0] is None:
                shape[0] = len(mat)
            l=None
            if not hasattr(mat, '__len__'):
                l=0
            elif len(mat) != shape[0]:
                l=len(mat)
            if not l is None:
                raise ParamError("Wrong len {} of element {}, should be  {}.".format(l, idx, shape[0]))
            for i, item in enumerate(mat):
                sub_shape = shape[1:]
                check_matrix(item, sub_shape, values, idx = [i] + idx)
                shape[1:] = sub_shape
        return shape
    except ParamError:
        raise
    except Exception as e:
        raise ParamError(e)


def check_knots(deg, knots, N):
    total_multiplicity = 0
    for knot, mult in knots:
        # This condition must hold if we assume only (0,1) interval of curve or surface parameters.
        #assert float(knot) >= 0.0 and float(knot) <= 1.0
        total_multiplicity += mult
    assert total_multiplicity == deg + N + 1


scalar_types = (int, float, np.int64)














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
        :param n_intervals: length of vector
        :param knot_range: support of the spline, min and max valid 't'
        :return: np array of knots
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
        full_knots = [ q for q, mult in knots for i in range(mult)  ]
        return cls(degree, full_knots)


    def __init__(self, degree, knots):
        """
        Constructor of the basis.
        :param degree: Degree of Bezier polynomials >=0.
        :param knots: Numpy array of the knots including multiplicities.
        """
        assert degree >=0
        self.degree = degree

        # check free ends (and  full degree along the whole curve)
        for i in range(self.degree):
            assert knots[i] == knots[i+1]
            assert knots[-i-1] == knots[-i-2]
        self.knots = np.array(knots)

        self.size = len(self.knots) - self.degree -1
        self.knots_idx_range = [self.degree, len(self.knots) - self.degree - 1]
        self.domain = self.knots[self.knots_idx_range]
        self.domain_size = self.domain[1] - self.domain[0]
        self.n_intervals = self.size - self.degree
        # Number of basis functions.

        if self.degree == 2:
            self.eval_base_vector = self._eval_base_vector_deg_2
            self.eval_diff_base_vector = self._eval_diff_base_vector_deg_2

    def pack_knots(self):
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
        if idx < 0 or idx > self.n_intervals - 1:
            print("Warning: evaluation out of spline domain; t: {} min: {} max: {}".format(t, self.knots[0], self.knots[-1]))

        return max(min(idx, self.n_intervals - 1), 0)   # deals with t == self.knots[-1]


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


    def eval_base_vector(self, i_base, t):
        values = []
        for ib in range(i_base, i_base + self.degree + 1):
            values.append( self.eval(ib, t))
        return values


    def eval_diff_base_vector(self, i_base, t):
        values = []
        for ib in range(i_base, i_base + self.degree + 1):
            values.append( self.eval_diff(ib, t))
        return values


    def _eval_base_vector_deg_2(self, i_base, t):
        """
        This function compute normalized blending function aka base function of B-Spline curve or surface.
        :param knot_vec:
        :param t_param:
        :param order: (0: function value, 1: derivative function value)
        :param sparse:
        :return:
        """

        basis_values = np.zeros(3)

        tk1, tk2, tk3, tk4 = self.knots[i_base + 1 : i_base + 5]

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


    def _eval_diff_base_vector_deg_2(self, i_base, t):

        basis_values = np.zeros(3)

        tk1, tk2, tk3, tk4 = self.knots[i_base + 1: i_base + 5]

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

    @classmethod
    def make_raw(cls, poles, knots, rational=False, degree=2):
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
        basis = SplineBasis(degree, knots)
        return cls(basis, poles, rational)

    """
    Defines a 3D (or (dim -D) curve as B-spline. We shall work only with B-splines of degree 2.
    Corresponds to "B-spline Curve - <3D curve record 7>" from BREP format description.
    """
    def __init__(self, basis, poles, rational = False):
        self.basis = basis
        self.dim = len(poles[0]) - rational
        check_matrix(poles, [self.basis.size, self.dim + rational], scalar_types )

        self.poles=np.array(poles)  # N x D
        self.rational=rational
        if rational:
            self._weights = poles[:, self.dim]
            self._poles = (poles[:, 0:self.dim].T * self._weights ).T


    def eval(self, t):

        it = self.basis.find_knot_interval(t)
        dt = self.basis.degree + 1
        t_base_vec = np.array([self.basis.eval(jt, t) for jt in range(it, it + dt)])

        if self.rational:
            top_value = np.inner(t_base_vec, self._poles[it: it + dt, :].T)
            bot_value = np.inner(t_base_vec, self._weights[it: it + dt])
            return top_value / bot_value
        else:
            return  np.inner(t_base_vec, self.poles[it: it + dt, :].T)

    def eval_array(self, t_points):
        """
        Evaluate in array of t-points.
        :param t_points: array N x float
        :return: Numpy array N x D, D is dimension of the curve.
        """
        return np.array( [ self.eval(t) for t in t_points] )

    def aabb(self):
        return (np.amin(self.poles, axis=0), np.amax(self.poles, axis=0))


class Surface:
    """
    Defines a B-spline surface.
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
        v_basis = SplineBasis(degree[0], knots[0])
        return cls( (u_basis, v_basis), poles, rational)


    def __init__(self, basis, poles, rational=False):
        """
        Construct a B-spline in 3d space.
        :param poles: Matrix (list of lists) of Nu times Nv poles (control points).
                      Single pole is a points ( X, Y, Z ) or weighted point (X,Y,Z, w). X,Y,Z,w are floats.
                      Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: Tuple (u_knots, v_knots). Both u_knots and v_knots are lists of tuples
                      (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                      and multiplicity is positive int. For both U and V knot vector the total number of knots,
                      i.e. sum of their multiplicities, must be degree + N + 1, where N is number of poles.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles. BREP format have two independent flags
                      for U and V parametr, but only choices 0,0 and 1,1 have sense.
        :param degree: (u_degree, v_degree) Both positive ints.
        """

        self.u_basis, self.v_basis = basis
        self.rational = rational
        self.dim = len(poles[0][0]) - rational
        check_matrix(poles, [self.u_basis.size, self.v_basis.size, self.dim + rational], scalar_types )
        self.poles=np.array(poles)
        assert self.poles.shape == (self.u_basis.size, self.v_basis.size, self.dim + rational)
        if rational:
            self._weights = poles[:, :, self.dim]
            self._poles = (poles[:,:,0:self.dim].T * self._weights.T ).T

    def eval(self, u, v):
        iu = self.u_basis.find_knot_interval(u)
        iv = self.v_basis.find_knot_interval(v)
        du = self.u_basis.degree + 1
        dv = self.v_basis.degree + 1
        u_base_vec = np.array([self.u_basis.eval(ju, u) for ju in range(iu, iu + du)])
        v_base_vec = np.array([self.v_basis.eval(jv, v) for jv in range(iv, iv + dv)])

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

    def eval_array(self, uv_points):
        """
        Evaluate in array of t-points.
        :param uv_points: numpy array N x [u, v]
        :return: Numpy array N x D, D is dimension of the curve.
        """
        assert uv_points.shape[1] == 2
        return np.array( [ self.eval(u, v) for u, v in uv_points] )



class Z_Surface:
    """
    Simplified B-spline surface that use just linear or bilinear transform between XY  and UV.

    TODO:
    - We need conversion to full 3D surface for the BREP output
    - Optimization: simplified Bspline evaluation just for the singel coordinate
    """
    def __init__(self, xy_quad, z_surface):
        """
        Construct a surface given by the  1d surface for the Z coordinate and XY quadrilateral
        for the bilinear UV -> XY mapping.
        :param xy_quad: np array N x 2
            Four or three points, determining bilinear or linear mapping, respectively.
            Four points giving XY coordinates for the uv corners: (0,1), (0,0), (1,0),  (1,1)
            Three points giving XY coordinates for the uv corners:  (0,1), (0,0), (1,0)
        :param z_surface: !D Surface object.
        """
        assert z_surface.dim == 1
        self.z_surface = z_surface
        self.u_basis = z_surface.u_basis
        self.v_basis = z_surface.v_basis
        self.dim = 3

        if len(xy_quad) == 3 or \
           np.allclose(xy_quad[3], xy_quad[0] + xy_quad[2] - xy_quad[1]):
            # linear case
            self.xy_shift = xy_quad[1]
            v_vec = xy_quad[0] - xy_quad[1]
            u_vec = xy_quad[2] - xy_quad[1]
            self.mat_uv_to_xy = np.column_stack((u_vec, v_vec))
            self.mat_xy_to_uv = la.inv(self.mat_uv_to_xy)

            self.xy_to_uv = self._linear_xy_to_uv
            self.uv_to_xy = self._linear_uv_to_xy

        elif len(xy_quad) == 4:
            # bilinear case
            self.quad = xy_quad

            self.xy_to_uv = self._bilinear_xy_to_uv
            self.uv_to_xy = self._bilinear_uv_to_xy

        else:
            assert False, "Three or four points must be given."

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
        poles = np.concatenate( (xy_poles, self.z_surface.poles), axis = 2 )

        return Surface(basis, poles)


    def transform(self, xy_mat, z_mat):
        """
        Transform the surface by arbitrary XY linear transform and Z linear transform.
        :param xy_mat: np array, 2 rows 3 cols, last column is xy shift
        :param z_shift: [ z_scale, z_shift]
        :return: None
        """
        assert xy_mat.shape == (2, 3)
        assert z_mat.shape == (2, )



        self.mat_uv_to_xy = xy_mat[0:2,0:2].dot( self.mat_uv_to_xy )
        self.xy_shift = xy_mat[0:2,0:2].dot( self.xy_shift ) + xy_mat[0:2, 2]
        self.mat_xy_to_uv = la.inv(self.mat_uv_to_xy)

        # apply z-transfrom directly to the poles
        self.z_surface.poles *= z_mat[0]
        self.z_surface.poles += z_mat[1]



    """
    def uv_to_xy(self, uv_points):

    Abstract method. Converts array of uv points to array of xy points.
    :param uv_points: numpy array N x [u,v]
    :return: numpy array N x [x,y]
    """
    def _linear_uv_to_xy(self, uv_points):
        assert uv_points.shape[1] == 2, "Size: {}".format(uv_points.shape)
        return ( np.dot(uv_points, self.mat_uv_to_xy.T) + self.xy_shift)


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
        return  np.dot((xy_points - self.xy_shift), self.mat_xy_to_uv.T)


    def _bilinear_xy_to_uv(self, xy_points):
        assert xy_points.shape[1] == 2
        assert False, "Not implemented yet."


    def eval(self, u, v):
        z = self.z_surface.eval(u, v)
        uv_points = np.array([[u, v]])
        x, y = self.uv_to_xy( uv_points )[0]
        return np.array( [x, y, z] )


    def eval_array(self, uv_points):
        assert uv_points.shape[1] == 2
        z_points = self.z_surface.eval_array(uv_points)
        xy_points = self.uv_to_xy(uv_points)
        return np.concatenate( (xy_points, z_points), axis = 1)



    def eval_xy_array(self, xy_points):
        uv_points  = self.xy_to_uv(xy_points)
        z_points = self.z_surface.eval_array(uv_points)
        return np.concatenate( (xy_points, z_points), axis = 1)

    def z_eval_array(self, uv_points):
        assert uv_points.shape[1] == 2
        z_points = self.z_surface.eval_array(uv_points)
        return z_points.reshape(-1)

    def z_eval_xy_array(self, xy_points):
        uv_points = self.xy_to_uv(xy_points)
        z_points = self.z_surface.eval_array(uv_points)
        return z_points.reshape(-1)

class InvalidGridExc(Exception):
    pass


class GridSurface:
    """
    Surface given as bilinear interpolation of a regular grid of points.
    """
    # TODO: calling transform lose mapping between original point grid and unit square, approximation can not be performed
    step_tolerance = 1e-10

    def __init__(self):
        """
        Initialize point grid from numpy array.
        :param grid: NxMx3 numpy array of NxM grid of #D coordinates
        """
        self.grid=None
        self.mat_xy_to_uv=None
        self.mat_uv_to_xy=None
        self.shift=None
        self.shape = (0,0)
        self.uv_step = (0,0)


    def load(self, filename):
        """
        Load the grid surface from file
        :param filename:
        :return:
        """
        point_seq = np.loadtxt(filename)
        assert min(point_seq.shape) > 1
        self.init_from_seq(point_seq.T)


    def init_from_seq(self, point_seq):
        """
        Get 2d transform matrix 2 rows 3 cols to map a grid of XY points to unit square
        :param point_seq: numpy array N x 2
        :return:
        """

        assert point_seq.shape[0] == 3
        n_points = point_seq.shape[1]
        point_seq_xy = point_seq[0:2,:]

        vtx_00 = point_seq_xy[0:2, 0]
        vtx_du = point_seq_xy[0:2, 1]

        u_step = vtx_du - vtx_00
        for i in range(2, n_points):
            step = point_seq_xy[:,i] - point_seq_xy[:,i-1]
            if la.norm(u_step - step) > self.step_tolerance:
                break

        vtx_dv = point_seq_xy[:,i]
        v_step = vtx_dv - vtx_00

        nu = i
        nv = int(n_points / nu)
        if not n_points == nu*nv:
            raise InvalidGridExc("Not a M*N grid.")

        # check total range of the grid
        vtx_10 = point_seq_xy[:, nu-1]
        vtx_01 = point_seq_xy[:, -nu]
        vtx_11 = point_seq_xy[:, -1]
        u_range_0 = vtx_10 - vtx_00
        u_range_1 = vtx_11 - vtx_01
        v_range_0 = vtx_01 - vtx_00
        v_range_1 = vtx_11 - vtx_10

        if not la.norm(u_range_0 - u_range_1) < self.step_tolerance or \
            not la.norm(v_range_0 - v_range_1) < self.step_tolerance:
            raise InvalidGridExc("Grid XY envelope is not a parallelogram.")

        u_step = u_range_0 / (nu-1)
        v_step = v_range_0 / (nv-1)

        # check regularity of the grid
        for i in range(nu*nv):
            pred_x = i - 1
            pred_y = i - nu
            if i%nu == 0:
                pred_x= -1
            if pred_x > 0 and not la.norm(point_seq_xy[:, i] - point_seq_xy[:, pred_x] - u_step) < 2*self.step_tolerance:
                raise InvalidGridExc("Irregular grid in X direction, point %d"%i)
            if pred_y > 0 and not la.norm(point_seq_xy[:, i] - point_seq_xy[:, pred_y] - v_step) < 2*self.step_tolerance:
                raise InvalidGridExc("Irregular grid in Y direction, point %d"%i)

        #self.uv_to_xy(np.array([[0, 1], [0, 0], [1, 0], [1, 1]]).T)
        self.quad = quad = np.stack([vtx_01, vtx_00, vtx_10, vtx_11], axis = 0)
        # Envelope quad - polygon, oriented counter clockwise.

        self.grid_z = point_seq[2, :].reshape(nv, nu)
        # grid of original Z values, format as matrix

        self.shape = (nu, nv)
        # Grid shape.

        self.uv_step = (1.0 / float(nu - 1), 1.0 / float(nv - 1))
        # Grid step in u, v direction respectively.

        self.points_xyz = point_seq.T
        # Original sequance of XYZ points

        u_basis = SplineBasis.make_equidistant(1, nu-1)
        v_basis = SplineBasis.make_equidistant(1, nv-1)
        poles_z = np.transpose( point_seq[2, :].reshape(nv, nu, 1), axes = [1, 0, 2] )
        self.z_surface = Z_Surface(quad[0:3], Surface((u_basis, v_basis), poles_z) )
        # related bilinear Z-surface, all evaluations just call this object.
        self.check_map()

    def check_map(self):
        # check that xy_to_uv works fine
        uv_quad = self.xy_to_uv(self.quad)
        print( uv_quad )
        assert np.allclose( uv_quad, np.array([[0, 1], [0, 0], [1, 0], [1, 1]]) )

    def transform(self, xy_mat, z_mat):
        self.z_surface.transform(xy_mat, z_mat)


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
        return self.z_surface.eval_array(uv_points)


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
            iuv = np.floor(uv / self.uv_step)
            iu = max(0, min(self.shape[0] - 2, int(iuv[0])))
            iv = max(0, min(self.shape[1] - 2, int(iuv[1])))
            iuv = np.array([iu, iv])

            uv_loc = uv / self.uv_step - iuv
            u_loc = np.array([1 - uv_loc[0], uv_loc[0]])
            v_loc = np.array([1 - uv_loc[1], uv_loc[1]])
            Z_mat = self.grid_z[iv: (iv + 2), iu: (iu + 2)]
            result[i] = self.z_surface.z_scale*(v_loc.dot(Z_mat).dot(u_loc)) + self.z_surface.z_shift
        return result





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










