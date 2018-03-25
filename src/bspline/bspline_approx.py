"""
Collection of functions to produce Bapline curves and surfaces as approximation of various analytical curves and surfaces.
"""

import bspline as bs
import numpy as np
import math
import time
#import numpy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

            
"""
Approximation methods for B/splines of degree 2.

"""
def plane_surface(vtxs):
    """
    Returns B-spline surface of a plane given by 3 points.
    We retun also list of UV coordinates of the given points.
    :param vtxs: List of tuples (X,Y,Z)
    :return: ( Surface, vtxs_uv )
    """
    assert len(vtxs) == 3, "n vtx: {}".format(len(vtxs))
    vtxs = np.array(vtxs)
    vv = vtxs[1] + vtxs[2] - vtxs[0]
    vtx4 = [ vtxs[0], vtxs[1], vv, vtxs[2]]
    return bilinear_surface(vtx4)



def bilinear_surface(vtxs):
    """
    Returns B-spline surface of a bilinear surface given by 4 corner points:
    uv coords:
    We retun also list of UV coordinates of the given points.
    :param vtxs: List of tuples (X,Y,Z)
    :return: ( Surface, vtxs_uv )
    """
    assert len(vtxs) == 4, "n vtx: {}".format(len(vtxs))
    vtxs = np.array(vtxs)
    def mid(*idx):
        return np.mean( vtxs[list(idx)], axis=0)

    # v - direction v0 -> v2
    # u - direction v0 -> v1
    poles = [ [vtxs[0],  mid(0, 3), vtxs[3]],
                [mid(0,1), mid(0,1,2,3), mid(2,3)],
                [vtxs[1], mid(1,2), vtxs[2]]
                ]
    knots = 3 * [0.0] + 3 * [1.0]
    basis = bs.SplineBasis(2, knots)
    surface = bs.Surface((basis, basis), poles)
    #vtxs_uv = [ (0, 0), (1, 0), (1, 1), (0, 1) ]
    return surface




def line(vtxs):
    '''
    Return B-spline approximation of a line from two points
    :param vtxs: [ X0, X1 ], Xn are point coordinates in arbitrary dimension D
    :return: Curve2D
    '''
    assert len(vtxs) == 2
    vtxs = np.array(vtxs)
    mid = np.mean(vtxs, axis=0)
    poles = [ vtxs[0],  mid, vtxs[1] ]
    knots = 3*[0.0] + 3*[1.0]
    basis = bs.SplineBasis(2, knots)
    return bs.Curve(basis, poles)




def surface_from_grid(grid_surface, nuv, **kwargs):
    """
    Make a Z_Surface of degree 2 as an approximation of the GridSurface.
    :param grid_surface: grid surface to approximate
    :param (nu, nv) Prescribed number of poles in u and v directions.
    :return: Z_surface object.
    """
    approx = _SurfaceApprox(grid_surface, nuv, **kwargs)
    return  approx.get_approximation()


def curve_from_grid(points, **kwargs):
    """
    Make a Curve (of degree 3) as an approximation of a sequence of points.
    :param points - N x D array, D is dimension
    :param nt Prescribed number of poles of the resulting spline.
    :return: Curve object.
    """
    deg = kwargs.get('degree', 3)
    tol = kwargs.get('tol', 0.01)
    weights = np.ones(points.shape[0])
    weights[0] = weights[-1] = 1000.0
    tck = scipy.interpolate.splprep(points.T, k=deg, s=tol, w = weights)[0]
    knots, poles, degree  = tck
    basis = bs.SplineBasis(degree, knots)
    curve = bs.Curve(basis, np.array(poles).T)
    return curve






class _SurfaceApprox:



    def __init__(self, grid_surface, nuv, **kwargs):
        self.grid_surf = grid_surface
        self.u_basis = bs.SplineBasis.make_equidistant(2, nuv[0])
        self.v_basis = bs.SplineBasis.make_equidistant(2, nuv[1])
        self.regularization_weight = kwargs.get('reg_weight', 1.0)

    def get_approximation(self):
        if not hasattr(self, 'z_surf'):
            self.z_surf = self.approx_chol()
        return self.z_surf


    def approx_chol(self):
        """
        This function tries to approximate terrain data with B-Spline surface patches
        using Cholesky decomposition
        :param terrain_data: matrix of 3D terrain data
        :param quad: points defining quadrangle area (array)
        :param u_knots: array of u knots
        :param v_knots: array of v knots
        :param sparse: if sparse matrix is used
        :param filter_thresh: threshold of filter
        :return: B-Spline patch
        """


        print('Transforming points to parametric space ...')
        start_time = time.time()
        points = self.grid_surf.points_xyz
        points_uv = self.grid_surf.xy_to_uv( points[:, 0:2] )


        # remove points far from unit square
        eps = 1.0e-15
        cut_min = np.array( [ -eps, -eps ])
        cut_max = np.array( [ 1+eps, 1+eps ])

        in_idx = np.all(np.logical_and(cut_min < points_uv,  points_uv <= cut_max), axis=1)
        points_uv = points_uv[in_idx]
        points_z = points[in_idx, 2][:,None]

        # snap to unit square
        points_uv = np.maximum(points_uv, np.array([0.0, 0.0]))
        points_uv = np.minimum(points_uv, np.array([1.0, 1.0]))


        self.grid_uvz = np.concatenate((points_uv, points_z), axis=1)
        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))

        # Own computation of approximation
        print('Creating B matrix ...')
        start_time = time.time()
        b_mat, interval = self.build_ls_matrix()
        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))

        print('Creating A matrix ...')
        start_time = time.time()
        a_mat = self.build_sparse_reg_matrix()
        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))

        print('Computing B^T B matrix ...')
        start_time = time.time()
        bb_mat = b_mat.transpose() * b_mat

        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))


        bb_norm = scipy.sparse.linalg.svds(bb_mat, k=1, ncv=10, tol=1e-4, which='LM', v0=None,
                                           maxiter=300, return_singular_vectors=False)
        a_norm = scipy.sparse.linalg.svds(a_mat, k=1, ncv=10, tol=1e-4, which='LM', v0=None,
                                          maxiter=300, return_singular_vectors=False)
        c_mat = bb_mat + self.regularization_weight * (bb_norm[0] / a_norm[0]) * a_mat

        g_vec = self.grid_uvz[:, 2]
        b_vec = b_mat.transpose() * g_vec


        print('Computing Z coordinates ...')
        start_time = time.time()
        z_vec = scipy.sparse.linalg.spsolve(c_mat, b_vec)
        print(type(z_vec))
        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))


        # print('Computing differences ...')
        # start_time = time.time()
        # diff = (numpy.matrix(b_mat * z_vec).transpose() - g_vec).tolist()
        # diff = [item[0] for item in diff]
        # end_time = time.time()
        # print('Computed in {0} seconds.'.format(end_time - start_time))

        # Construct Z-Surface
        poles_z = z_vec.reshape(self.u_basis.size, self.v_basis.size, 1)
        surface_z = bs.Surface((self.u_basis, self.v_basis), poles_z)
        z_surf = bs.Z_Surface(self.grid_surf.quad, surface_z)

        return z_surf



    def build_ls_matrix(self):
        """
        Construction of the matrix (B) of the system of linear algebraic
        equations for control points of the 2th order B-spline surface
        :param u_knots:
        :param v_knots:
        :param terrain:
        :param sparse:
        :return:
        """
        u_n_basf = self.u_basis.size
        v_n_basf = self.v_basis.size
        n_points = self.grid_uvz.shape[0]

        n_uv_loc_nz =  (self.u_basis.degree +  1) * (self.v_basis.degree +  1)
        row = np.zeros(n_points * n_uv_loc_nz)
        col = np.zeros(n_points * n_uv_loc_nz)
        data = np.zeros(n_points * n_uv_loc_nz)

        nnz_b = 0

        interval = np.empty((n_points, 2))

        for idx in range(0, n_points):
            u, v = self.grid_uvz[idx, 0:2]
            iu = self.u_basis.find_knot_interval(u)
            iv = self.u_basis.find_knot_interval(v)
            u_base_vec = self.u_basis.eval_base_vector(iu, u)
            v_base_vec = self.u_basis.eval_base_vector(iv, v)
            # Hard-coded Kronecker product (problem based)
            for n in range(0, 3):
                data[nnz_b + 3 * n:nnz_b + 3 * (n + 1)] = v_base_vec[n] * u_base_vec
                for m in range(0, 3):
                    col[nnz_b + (3 * n) + m] = (iv + n) * u_n_basf + iu + m
            row[nnz_b:nnz_b + 9] = idx
            nnz_b += 9

            interval[idx][0] = iu
            interval[idx][1] = iv

        mat_b = scipy.sparse.csr_matrix((data, (row, col)), shape=(n_points, u_n_basf * v_n_basf))

        return mat_b, interval

    def _basis_in_q_points(self, basis):
        n_int = basis.n_intervals
        nq_points = len(self._q_points)
        point_val = np.zeros((3, n_int * nq_points))
        d_point_val = np.zeros((3, n_int * nq_points))
        point_idx = np.zeros((n_int * nq_points, 1))
        q_point = np.zeros((n_int * nq_points, 1))

        n = 0
        for i in range(n_int):
            us = basis.knots[i + 2]
            uil = basis.knots[i + 3] - basis.knots[i + 2]
            for j in range(nq_points):
                up = us + uil * self._q_points[j]
                q_point[n] = up
                idx = basis.find_knot_interval(up)
                u_base_vec = basis.eval_base_vector(idx, up)
                u_base_vec_diff = basis.eval_diff_base_vector(idx, up)
                point_val[:, n] = u_base_vec
                d_point_val[:, n] = u_base_vec_diff
                point_idx[n] = idx
                n += 1
        return point_val, d_point_val, point_idx, q_point

    def build_sparse_reg_matrix(self):
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

        u_n_basf = self.u_basis.size
        v_n_basf = self.v_basis.size
        u_n_inter = self.u_basis.n_intervals
        v_n_inter = self.v_basis.n_intervals
        n_uv_loc_nz = (self.u_basis.degree + 1) * (self.v_basis.degree + 1)

        # TODO: use Gauss quadrature from scipy
        # in fact for general degrees we should use different quadrature for u and different for v
        self._q_points =  [0, (0.5 - 1 / math.sqrt(20)), (0.5 + 1 / math.sqrt(20)), 1]
        self._weights = [1.0 / 6, 5.0 / 6, 5.0 / 6, 1.0 / 6]
        nq_points = len(self._q_points)

        # TODO: rename: u_vals, u_diffs, u_idxs, u_points
        u_point_val, ud_point_val, u_point_idx, q_u_point = self._basis_in_q_points(self.u_basis)
        v_point_val, vd_point_val, v_point_idx, q_v_point = self._basis_in_q_points(self.v_basis)

        # Matrix construction
        # TODO: Assembly local dense blocks 9*9 and then put these nonzeroes into sparse matirx
        # TODO: use numpy opperations to make assembly of local blocks readable, eliminate loops
        colv = np.zeros(n_uv_loc_nz)
        # TODO:rename data and data2 to something meqningful
        data = np.zeros(n_uv_loc_nz)
        data2 = np.zeros(n_uv_loc_nz)
        row_m = np.zeros((v_n_inter * u_n_inter * nq_points * nq_points * n_uv_loc_nz * n_uv_loc_nz))
        col_m = np.zeros((v_n_inter * u_n_inter * nq_points * nq_points * n_uv_loc_nz * n_uv_loc_nz))
        data_m = np.zeros((v_n_inter * u_n_inter * nq_points * nq_points * n_uv_loc_nz * n_uv_loc_nz))
        nnz_a = 0


        for i in range(v_n_inter):
            for k in range(nq_points):
                v_point = v_point_val[:, i * nq_points + k]
                vd_point = vd_point_val[:, i * nq_points + k]
                j_idx = v_point_idx[i * nq_points + k]
                for l in range(u_n_inter):
                    for m in range(nq_points):
                        u_point = u_point_val[:, l * nq_points + m]
                        ud_point = ud_point_val[:, l * nq_points + m]
                        i_idx = u_point_idx[l * nq_points + m]
                        for n in range(0, 3):
                            # Hard-coded Kronecker product: vd = numpy.kron(vd_point, u_point)
                            data[3 * n:3 * (n + 1)] = vd_point[n] * u_point
                            # Hard-coded Kronecker product: ud = numpy.kron(v_point, ud_point)
                            data2[3 * n:3 * (n + 1)] = v_point[n] * ud_point
                            # column indices for data & data2
                            for p in range(0, 3):
                                colv[3 * n + p] = (j_idx + n) * u_n_basf + i_idx + p

                        # Hard-coded Outer product:
                        # Jacobian * weights[m] * weights[k] * (numpy.outer(ud, ud) + numpy.outer(vd, vd))
                        #u_q = q_u_point[l * nq_points + m, 0]
                        #v_q = q_v_point[i * nq_points + k, 0]

                        # jacobian for UV coordinates should be used
                        jac = 1.0 / u_n_inter / v_n_inter
                        coef = self._weights[m] * self._weights[k] * jac
                        for n in range(0, 9):
                            row_m[nnz_a + 9 * n:nnz_a + 9 * (n + 1)] = colv
                            col_m[nnz_a + 9 * n:nnz_a + 9 * (n + 1)] = colv[n]
                            data_m[nnz_a + 9 * n:nnz_a + 9 * (n + 1)] = coef * (data[n] * data + data2[n] * data2)
                        nnz_a += n_uv_loc_nz * n_uv_loc_nz

        mat_a = scipy.sparse.coo_matrix((data_m, (row_m, col_m)),
                                        shape=(u_n_basf * v_n_basf, u_n_basf * v_n_basf)).tocsr()

        return mat_a


