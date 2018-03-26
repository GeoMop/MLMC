"""
Collection of functions to produce Bapline curves and surfaces as approximation of various analytical curves and surfaces.
"""

import logging
import time
#import math

import numpy as np
import numpy.linalg as la
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

import geomop.bspline as bs
import csv

#logging.basicConfig(level=logging.DEBUG)
#logging.info("Test info mesg.")
"""
Approximation methods for B/splines of degree 2.

"""
def plane_surface(vtxs, overhang=0.0):
    """
    Returns B-spline surface of a plane given by 3 points.
    We retun also list of UV coordinates of the given points.
    U direction v0 -> v1
    V direction v0 -> v2
    :param vtxs: List of tuples (X,Y,Z)
    :return: ( Surface, vtxs_uv )
    """
    assert len(vtxs) == 3, "n vtx: {}".format(len(vtxs))
    vtxs = np.array(vtxs)
    vv = vtxs[1] + vtxs[2] - vtxs[0]
    vtx4 = [ vtxs[0], vtxs[1], vv, vtxs[2]]
    return bilinear_surface(vtx4, overhang)



def bilinear_surface(vtxs, overhang=0.0):
    """
    Returns B-spline surface of a bilinear surface given by 4 corner points:
    uv coords:
    We retun also list of UV coordinates of the given points.
    :param vtxs: List of tuples (X,Y,Z)
    :return: ( Surface, vtxs_uv )
    """
    assert len(vtxs) == 4, "n vtx: {}".format(len(vtxs))
    vtxs = np.array(vtxs)
    if overhang > 0.0:
        dv = np.roll(vtxs, -1, axis=0) - vtxs
        dv *= overhang
        vtxs +=  np.roll(dv, 1, axis=0) - dv

    def mid(*idx):
        return np.mean( vtxs[list(idx)], axis=0)

    # v - direction v0 -> v2
    # u - direction v0 -> v1
    poles = [ [vtxs[0],  mid(0, 3), vtxs[3]],
                [mid(0,1), mid(0,1,2,3), mid(2,3)],
                [vtxs[1], mid(1,2), vtxs[2]]
                ]
    knots = 3 * [0.0 - overhang] + 3 * [1.0 + overhang]
    basis = bs.SplineBasis(2, knots)
    surface = bs.Surface((basis, basis), poles)
    #vtxs_uv = [ (0, 0), (1, 0), (1, 1), (0, 1) ]
    return surface




def line(vtxs, overhang = 0.0):
    '''
    Return B-spline approximation of a line from two points
    :param vtxs: [ X0, X1 ], Xn are point coordinates in arbitrary dimension D
    :return: Curve2D
    '''
    assert len(vtxs) == 2
    vtxs = np.array(vtxs)
    if overhang > 0.0:
        dv = overhang*(vtxs[1] - vtxs[0])
        vtxs[0] -= dv
        vtxs[1] += dv
    mid = np.mean(vtxs, axis=0)
    poles = [ vtxs[0],  mid, vtxs[1] ]
    knots = 3*[0.0 - overhang] + 3*[1.0 + overhang]
    basis = bs.SplineBasis(2, knots)
    return bs.Curve(basis, poles)




def surface_from_grid(grid_surface, nuv):
    """
    Make a Z_Surface of degree 2 as an approximation of the GridSurface.
    :param grid_surface: grid surface to approximate
    :param (nu, nv) Prescribed number of poles in u and v directions.
    :return: Z_surface object.
    """
    approx = SurfaceApprox.approx_from_grid_surface(grid_surface)
    return approx.compute_approximation(nuv=nuv)



def curve_from_grid(points, **kwargs):
    """
    Make a Curve (of degree 3) as an approximation of a sequence of points.
    :param points - N x D array, D is dimension
    :param nt Prescribed number of poles of the resulting spline.
    :return: Curve object.

    TODO:
    - Measure efficiency. Estimate how good we can be. Do it our self if we can do at leas 10 times better.
    - Find out which method is used. Hoschek (4.4.1) refers to several methods how to determine parametrization of
    the curve, i.e. Find parameters t_i to the given approximation points P_i.
    - Further on it is not clear what is the mening of the 's' parameter and how one cna influence tolerance and smoothness.
    - Some sort of adaptivity is used.


    """
    deg = kwargs.get('degree', 3)
    tol = kwargs.get('tol', 0.01)
    weights = np.ones(points.shape[0])
    weights[0] = weights[-1] = 1000.0
    tck = scipy.interpolate.splprep(points.T, k=deg, s=tol, w = weights)[0]
    knots, poles, degree  = tck
    curve_poles=np.array(poles).T
    curve_poles[0] = points[0]
    curve_poles[-1] = points[-1]
    basis = bs.SplineBasis(degree, knots)
    curve = bs.Curve(basis, curve_poles)
    return curve



def convex_hull_2d(sample):
    link = lambda a, b: np.concatenate((a, b[1:]))
    edge = lambda a, b: np.concatenate(([a], [b]))

    def dome(sample, base):
        """
        Return convex hull of the points on the right side from the base.
        :param sample: Nx2 nupy array of points
        :param base: A segment, np array  [[x0,y0], [x1,y1]]
        :return: np array of points Nx2 on forming the convex hull
        """
        # End points of line.
        h, t = base
        normal = np.dot(((0,-1),(1,0)),(t-h))
        # Distances from the line.
        dists = np.dot(sample-h, normal)

        outer = np.repeat(sample, dists>0, 0)
        if len(outer):
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                    dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
        axis = sample[:,0]
        # Get left most and right most points.
        base = np.take(sample, [np.argmin(axis), np.argmax(axis)], 0)
        return link(dome(sample, base), dome(sample, base[::-1]))
    else:
        return sample


def min_bounding_rect(hull):
    """
    Compute minimal area bounding box from a convex hull.
    Quadratic algorithm with respect to number of hull points is used, anyway calculation a convex hull
    takes longer since number of hull points is about sqrt of all points.
    :param hull: Nx2 numpy array of the convex hull points. First and last must be the same.
    :return: Corners of the rectangle.
    """
    # Compute edges (x2-x1,y2-y1)
    edges = hull[1:, :] - hull[:-1, :]

    # Calculate edge angles   atan2(y/x)
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])

    # Check for angles in 1st quadrant
    edge_angles = np.abs( edge_angles%(np.pi/2))

    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, float("inf"), 0, 0, 0, 0, 0, 0) # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    for i in range( len(edge_angles) ):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        angle = edge_angles[i]
        R = np.array([[np.cos(angle), np.cos(angle - (np.pi / 2))],
                      [np.cos(angle + (np.pi / 2)), np.cos(angle)]])

        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(hull)) # 2x2 * 2xn

        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)

        # Calculate height/width/area of this bounding rectangle
        area = (max_x - min_x) * (max_y - min_y)

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = ( edge_angles[i], area, min_x, max_x, min_y, max_y )

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]
    R = np.array([[np.cos(angle), np.cos(angle- (np.pi / 2))],
                  [np.cos(angle + (np.pi / 2)), np.cos(angle)]])


    # min/max x,y points are against baseline
    min_x = min_bbox[2]
    max_x = min_bbox[3]
    min_y = min_bbox[4]
    max_y = min_bbox[5]

    # Calculate corner points and project onto rotated frame
    corner_points = np.zeros( (4,2) ) # empty 2 column array
    corner_points[0] = np.dot( [ min_x, max_y ], R )
    corner_points[1] = np.dot( [ min_x, min_y ], R )
    corner_points[2] = np.dot( [ max_x, min_y ], R )
    corner_points[3] = np.dot( [ max_x, max_y ], R )

    return corner_points


class SurfaceApprox:
    """
    Class to compute a Bspline surface approximation from given set of XYZ points.
    TODO:
    - Check efficiency of scipy methods, compare it to our approach assuming theoretical number of operations.
    - Compute BtB directly during single assembly pass, local 9x9 matricies as in A matrix.
    - In contradiction to some literature (Hoschek) solution of the LS system is fast as long as the basis is local (
      this is true for B-splines).
    - Extensions to fitting X and Y as well - general Surface

    """

    @staticmethod
    def approx_from_file(filename):
        """
        Load a sequence of XYZ points on a surface to be approximated.
        Optionally points may have weights (i.e. four values per line: XYZW)
        :param filename: Path to the input text file.
        :return: The approximation object.
        """
        with open(filename, 'r') as f:
            point_seq = np.array([l for l in csv.reader(f, delimiter=' ')], dtype=float)

        # too slow: alternatives: loadtxt (16s), csv.reader (1.6s), pandas. read_csv (0.6s)
        #point_seq = np.loadtxt(filename)
        return SurfaceApprox(point_seq)


    @staticmethod
    def approx_from_grid_surface(grid_surface):
        """
        Approximation from a GrodSurface object. Use grid of Z coords in
        XY pozitions of poles.
        :param grid_surface: GridSurface.
        :return:
        """
        u_basis, v_basis = grid_surface.u_basis, grid_surface.v_basis

        u_coord = u_basis.make_linear_poles()
        v_coord = v_basis.make_linear_poles()

        U, V = np.meshgrid(u_coord, v_coord)
        uv_points = np.stack( [U.ravel(), V.ravel()], axis = 1 )

        xyz = grid_surface.eval_array(uv_points)
        approx = SurfaceApprox(xyz)
        approx.quad = grid_surface.quad
        return approx

    def __init__(self, points):
        """
        Initialize the approximation object with the points.
        :param points: Nx3 (XYZ) or Nx4 (XYZW - points with weights)
        """

        # Degree of approximation in U anv V directions, fixed to 2.
        self._degree = np.array((2, 2))


        assert( points.shape[1] >= 3 )
        # XYZ points
        self._n_points = points.shape[0]
        self._xy_points = points[:, 0:2]
        self._z_points = points[:, 2]

        # point weights
        if points.shape[1] > 3:
            self._weights = points[:, 3]
        else:
            self._weights = None

        ## Approximation parameters.

        # Bounding quadrilateral of the approximation (currently only parallelograms are supported).
        # Only first three points P0, P1, P2 are considered. V direction is P0 - P1, U direction is P2 - P1.
        # I.e. points are sorted counter-clockwise.
        self.quad = None

        # (nu, nv) number of subintervals of the BSSurface on U and V axis.
        # Default is estimated from the number of input points N as nu=nv=sqrt(N)/3.
        self.nuv = None

        # Weight of the regularizing term.
        self.regularization_weight = 0.001

        ## Approximation results

        # Approximationg BSSurface
        self.surface = None

        # Error of the approximation
        self.error = None

    def set_quad(self, quad = None):
        if quad is None:
            quad = np.array([[0,1], [0,0], [1,0], [1,1]])
        self.quad = quad

    def compute_default_quad(self):
        """
        Compute and set boundary quad as a minimum area bounding box of the input XY point set.
        :return: The quadrilateral vertices.
        """
        hull = convex_hull_2d(self._xy_points)
        self.quad = min_bounding_rect(hull)
        return self.quad


    # def transformed_quad(self, xy_mat):
    #     """
    #     Return actual quad transformed by given transform matrix.
    #     Boudary quadrilateral of the approximation is not touched.
    #     :param xy_mat: np array, 2 rows 3 cols, last column is xy shift
    #     :return: transformed quad as 4x2 numpy array or None
    #     """
    #     if self.quad is None:
    #         return None
    #     assert xy_mat.shape == (2, 3)
    #     quad_center = np.average(self.quad, axis=0)
    #     # transform quad
    #     return np.dot( (self.quad - quad_center) , xy_mat[0:2, 0:2].T) + quad_center + xy_mat[0:2, 2]
    #
    #     #return np.dot((self.quad), xy_mat[0:2, 0:2].T) + xy_mat[0:2, 2]


    def compute_default_nuv(self):
        """
        Compute default quad (if not set) filter points in the quad and estimate
        nuv from their count. Set self.nuv
        :return: nuv = (nu, nv)
        """
        if self.quad is None:
            self.quad = self.compute_default_quad()
        self._compute_uv_points()

        nuv = self._compute_default_nuv(len(self._z_quad_points))
        self.nuv = nuv.astype(int)
        if self.nuv[0] < 1 or self.nuv[1] < 1:
            raise Exception("Two few points, {}, to make approximation, degree: {}".format(self._n_points, self._degree))
        return self.nuv




    def compute_approximation(self, **kwargs):
        """
        Compute approximation of the point set (given to constructor).
        Approximation parameters can be passed in through kwargs or set in the object before the call.
        :param quad: [(x1,y1), .. , (x4,y4)] Set vertices of different quad for the point set.
        :param nuv: (nu, nv) Set number of intervals of the resulting B-spline, in U and V direction
        :param regularization_wight: Default 0.001, is scaled by the max singular value of B.
        :return: B-Spline surface
        """

        self.quad = kwargs.get("quad", self.quad)
        self.nuv = kwargs.get("nuv", self.nuv)
        self.regularization_weight = kwargs.get("regularization_weight", self.regularization_weight)

        logging.info('Transforming points (n={}) ...'.format(self._n_points))
        start_time = time.time()
        if self.quad is None:
            self.compute_default_quad()
        if self.nuv is None:
            self.compute_default_nuv()

        # TODO: better logic, since this has to be recomputed only if quad is changed.
        self._compute_uv_points()

        logging.info("Using {} x {} B-spline approximation.".format(self.nuv[0], self.nuv[1]))
        self._u_basis = bs.SplineBasis.make_equidistant(2, self.nuv[0])
        self._v_basis = bs.SplineBasis.make_equidistant(2, self.nuv[1])

        end_time = time.time()
        logging.info('Computed in: {} s'.format(end_time - start_time))

        # Approximation itself
        logging.info('Creating B matrix ...')
        start_time = time.time()
        b_mat, interval = self._build_ls_matrix()
        end_time = time.time()
        logging.info('Computed in: {} s'.format(end_time - start_time))

        logging.info('Creating A matrix ...')
        start_time = time.time()
        a_mat = self._build_sparse_reg_matrix()
        end_time = time.time()
        logging.info('Computed in: {} s'.format(end_time - start_time))

        logging.info('Scaling + B^T B ...')
        start_time = time.time()
        g_vec = self._z_quad_points[:]
        if self._weights is not None:
            W = scipy.sparse.diags(self._w_quad_points, 0)
            wg_vec = np.dot(W, g_vec)
            wb_mat = np.dot(W, b_mat)
        else:
            wg_vec = g_vec
            wb_mat = b_mat
        b_vec = b_mat.transpose().dot( wg_vec )
        bb_mat = b_mat.transpose().dot(wb_mat)
        end_time = time.time()
        logging.info('Computed in: {} s'.format(end_time - start_time))

        logging.info('Computing A and B svds approximation ...')
        start_time = time.time()
        bb_norm = scipy.sparse.linalg.svds(bb_mat, k=1, ncv=10, tol=1e-2, which='LM', v0=None,
                                           maxiter=300, return_singular_vectors=False)
        a_norm = scipy.sparse.linalg.eigsh(a_mat, k=1, ncv=10, tol=1e-2, which='LM',
                                          maxiter=300, return_eigenvectors=False)
        c_mat = bb_mat + self.regularization_weight * (bb_norm[0] / a_norm[0]) * a_mat
        end_time = time.time()
        logging.info('Computed in: {} s'.format(end_time - start_time))

        logging.info('Solving for Z coordinates ...')
        start_time = time.time()
        z_vec = scipy.sparse.linalg.spsolve(c_mat, b_vec)
        assert not np.isnan(np.sum(z_vec)), "Singular matrix for approximation."
        end_time = time.time()
        logging.info('Computed in: {} s'.format(end_time - start_time))

        logging.info('Computing error ...')
        start_time = time.time()
        diff = b_mat.dot(z_vec) - g_vec
        self.error = max_diff = np.max(diff)
        logging.info("Approximation error (max norm): {}".format(max_diff) )
        end_time = time.time()
        logging.info('Computed in: {} s'.format(end_time - start_time))

        # Construct Z-Surface
        poles_z = z_vec.reshape(self._v_basis.size, self._u_basis.size).T
        #poles_z *= self.grid_surf.z_scale
        #poles_z += self.grid_surf.z_shift
        surface_z = bs.Surface((self._u_basis, self._v_basis), poles_z[:,:,None])
        self.surface = bs.Z_Surface(self.quad[0:3], surface_z)

        return self.surface


    def _compute_default_nuv(self, n_points):
        """
        Default nu and nv for given number of points inside of quad.
        :return: (nu, nv)
        """
        assert(self.quad is not None)

        dv = la.norm(self.quad[0, :] - self.quad[1, :])
        du = la.norm(self.quad[2, :] - self.quad[1, :])

        # try to make number of unknowns less then number of remaining points
        # +1 to improve determination
        nv = np.sqrt( n_points * dv / du )
        nu = nv * du / dv
        nuv = np.array( [np.floor(nu / 3), np.floor(nv / 3)] ) - self._degree
        self.nuv = np.maximum(1, nuv)
        return self.nuv



    def _compute_uv_points(self):
        """
        Map XY points to quad, remove points out of quad.
        Results: self._uv_quad_points, self._z_quad_points, self._w_quad_points
        :return:
        """
        xy_shift = self.quad[1,:]
        v_vec = self.quad[0, :] - self.quad[1, :]
        u_vec = self.quad[2, :] - self.quad[1, :]
        mat_uv_to_xy = np.column_stack((u_vec, v_vec))
        mat_xy_to_uv = la.inv(mat_uv_to_xy)
        points_uv = np.dot((self._xy_points - xy_shift), mat_xy_to_uv.T)

        # remove points far from unit square
        eps = 1.0e-15
        cut_min = np.array( [ -eps, -eps ])
        cut_max = np.array( [ 1+eps, 1+eps ])
        in_idx = np.all(np.logical_and(cut_min < points_uv,  points_uv <= cut_max), axis=1)
        points_uv = points_uv[in_idx]

        logging.debug("Number of points out of the grid domain: {}".format(len(points_uv) - np.sum(in_idx)))

        # snap to unit square
        points_uv = np.maximum(points_uv, np.array([0.0, 0.0]))
        self._uv_quad_points = np.minimum(points_uv, np.array([1.0, 1.0]))
        self._z_quad_points = self._z_points[in_idx]
        if self._weights is not None:
            self._w_quad_points = self._weights[in_idx]






    def _build_ls_matrix(self):
        """
        Construction of the matrix (B) of the system of linear algebraic
        equations for control points of the 2th order B-spline surface
        :param u_knots:
        :param v_knots:
        :param terrain:
        :param sparse:
        :return:
        """
        u_n_basf = self._u_basis.size
        v_n_basf = self._v_basis.size
        n_points = self._uv_quad_points.shape[0]

        n_uv_loc_nz =  (self._u_basis.degree +  1) * (self._v_basis.degree +  1)
        row = np.zeros(n_points * n_uv_loc_nz)
        col = np.zeros(n_points * n_uv_loc_nz)
        data = np.zeros(n_points * n_uv_loc_nz)

        nnz_b = 0

        interval = np.empty((n_points, 2))

        for idx in range(0, n_points):
            u, v = self._uv_quad_points[idx, 0:2]
            iu = self._u_basis.find_knot_interval(u)
            iv = self._v_basis.find_knot_interval(v)
            u_base_vec = self._u_basis.eval_base_vector(iu, u)
            v_base_vec = self._v_basis.eval_base_vector(iv, v)
            # Hard-coded Kronecker product (problem based)
            for n in range(0, 3):
                data[nnz_b + 3 * n:nnz_b + 3 * (n + 1)] = v_base_vec[n] * u_base_vec
                for m in range(0, 3):
                    col_item = (iv + n) * u_n_basf + iu + m
                    col[nnz_b + (3 * n) + m] = col_item
            row[nnz_b:nnz_b + 9] = idx
            nnz_b += 9

            interval[idx][0] = iu
            interval[idx][1] = iv

        mat_b = scipy.sparse.csr_matrix((data, (row, col)), shape=(n_points, u_n_basf * v_n_basf))

        return mat_b, interval

    def _basis_in_q_points(self, basis):
        n_int = basis.n_intervals
        nq_points = len(self._q_points)
        q_point = np.zeros((n_int * nq_points, 1))
        point_val_outer = np.zeros((3, 3, n_int)) # "3" considers degree 2
        d_point_val_outer = np.zeros((3, 3, n_int)) # "3" considers degree 2

        #TODO: use numpy functions for quadrature points
        n = 0
        for i in range(n_int):
            us = basis.knots[i + 2]
            uil = basis.knots[i + 3] - basis.knots[i + 2]
            for j in range(nq_points):
                up = us + uil * self._q_points[j]
                q_point[n] = up
                u_base_vec = basis.eval_base_vector(i, up)
                u_base_vec_diff = basis.eval_diff_base_vector(i, up)
                point_val_outer[:, :, i] += self._q_weights[j] * np.outer(u_base_vec,u_base_vec)
                d_point_val_outer[:, :, i] += self._q_weights[j] * np.outer(u_base_vec_diff,u_base_vec_diff)
                n += 1

        return  point_val_outer, d_point_val_outer,q_point


    def _build_sparse_reg_matrix(self):
        """
        Construction of the regularization matrix (A) to decrease variation of the terrain
        B z = b  ---> (B^T B + A)z = B^T b
        :param u_knots: vector of v-knots
        :param v_knots: vector of u-knots
        :param quad: points defining quadrangle area (array)
        :return: matrix

        -
        """

        #a = quad[:, 3] - quad[:, 2]
        #b = quad[:, 0] - quad[:, 1]
        #c = quad[:, 1] - quad[:, 2]
        #d = quad[:, 0] - quad[:, 3]

        u_n_basf = self._u_basis.size
        v_n_basf = self._v_basis.size
        u_n_inter = self._u_basis.n_intervals
        v_n_inter = self._v_basis.n_intervals
        n_uv_loc_nz = (self._u_basis.degree + 1) * (self._v_basis.degree + 1)

        # TODO: use Gauss quadrature from scipy
        # in fact for general degrees we should use different quadrature for u and different for v
        self._q_points =  [0, (0.5 - 1 / np.sqrt(20)), (0.5 + 1 / np.sqrt(20)), 1]
        self._q_weights = [1.0 / 6, 5.0 / 6, 5.0 / 6, 1.0 / 6]
        nq_points = len(self._q_points)

        u_val_outer, u_diff_val_outer, q_u_point = self._basis_in_q_points(self._u_basis)
        v_val_outer, v_diff_val_outer, q_v_point = self._basis_in_q_points(self._v_basis)
        # xy_outer shape is (3, 3, n_inter)

        row_m = np.zeros((v_n_inter * u_n_inter * n_uv_loc_nz * n_uv_loc_nz))
        col_m = np.zeros((v_n_inter * u_n_inter * n_uv_loc_nz * n_uv_loc_nz))
        data_m = np.zeros((v_n_inter * u_n_inter * n_uv_loc_nz * n_uv_loc_nz))

        nnz_a = 0
        #linsp = np.linspace(0, self._u_basis.degree, self._u_basis.degree+1)
        #llinsp = np.tile(linsp, self._u_basis.degree+1)
        #np.repeat((iv + linsp) * u_n_basf, self._u_basis.degree + 1) + llinsp
        i_local = np.arange(self._u_basis.degree+1, dtype=int)
        iuv_local = (u_n_basf * i_local[:, None] + i_local[None,:]).ravel() # 0,1,2, N+[0,1,2], 2*N+[0,1,2]
        #print("vnint: {} unint: {} nqp: {} prod: {}".format(v_n_inter, u_n_inter, nq_points, v_n_inter* u_n_inter* nq_points*nq_points))
        jac = 1.0 / u_n_inter / v_n_inter
        idx_range = n_uv_loc_nz * n_uv_loc_nz      # 9 * 9 = 81 NZ per single bspline square
        for iv in range(v_n_inter):
            v_val_outer_loc = v_val_outer[:, :, iv]
            dv_val_outer_loc = v_diff_val_outer[:, :, iv]

            for iu in range(u_n_inter):
                u_val_outer_loc = u_val_outer[:, :, iu]
                du_val_outer_loc = u_diff_val_outer[:, : , iu]
                # xy_outer_loc have shape 3x3

                v_du = np.kron(v_val_outer_loc, du_val_outer_loc)
                dv_u = np.kron(dv_val_outer_loc, u_val_outer_loc)
                data_m[nnz_a:nnz_a + idx_range] = jac * ( v_du + dv_u).ravel()  # 9x9 values

                iuv = iu + iv * u_n_basf
                colv = iuv + iuv_local
                col_m[nnz_a:nnz_a + idx_range] = np.repeat(colv, n_uv_loc_nz)
                row_m[nnz_a:nnz_a + idx_range] = np.tile(colv, n_uv_loc_nz)
                nnz_a += idx_range
        #print("Assembled")
        mat_a = scipy.sparse.coo_matrix((data_m, (row_m, col_m)),
                                        shape=(u_n_basf * v_n_basf, u_n_basf * v_n_basf)).tocsr()
        return mat_a