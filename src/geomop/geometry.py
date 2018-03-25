"""
This file contains algorithms for
1. constructing a 3D geometry in the BREP format
   (see https://docs.google.com/document/d/1qWq1XKfHTD-xz8vpINxpfQh4k6l1upeqJNjTJxeeOwU/edit#)
   from the Layer File format (see geometry_structures.py).
2. meshing the 3D geometry (e.g. using GMSH)
3. setting regions to the elements of the resulting mesh and other mesh postprocessing


TODO:
- check how GMSH number surfaces standing alone,
  seems that it number object per dimension by the IN time in DFS from solids down to Vtx,
  just ignoring main compound, not sure how numbering works for free standing surfaces.

- finish usage of polygon decomposition (without split)
- implement intersection in interface_finish_init
- tests

Heterogeneous mesh step:

- storing mesh step from regions into shape info objects
- ( brep created )
-
- add include


"""


import os
import sys



#geomop_src = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "gm_base")
#intersections_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), "intersections","src")
#sys.path.append(geomop_src)
#sys.path.append(intersections_src)

import gm_base.json_data as js
import gm_base.geometry_files.format_last as gs
import gm_base.geometry_files.layers_io as layers_io
import gm_base.geometry_files.polygons as polygons
import gm_base.geometry_files.polygons_io as polygons_io
import gm_base.geometry_files.bspline_io as bspline_io
import Geometry.gmsh_io as gmsh_io
import numpy as np
import numpy.linalg as la
import math


import gm_base.b_spline
import bspline as bs
import bspline_approx as bs_approx
import brep_writer as bw
# def import_plotting():
# global plt
# global bs_plot

#import gm_base.geometry_files.plot_polygons as plot_polygons

#import matplotlib
#import matplotlib.pyplot as plt

#import bspline_plot as bs_plot


###
#netgen_install_prefix="/home/jb/local/"
#netgen_path = "opt/netgen/lib/python3/dist-packages"
#sys.path.append( netgen_install_prefix + netgen_path )

#import netgen.csg as ngcsg
#import netgen.meshing as ngmesh


def check_point_tol(A, B, tol):
    diff = B - A
    norm = la.norm(diff)
    if norm > tol:
        raise Exception("Points too far: {} - {} = {}, norm: {}".format(A,B,diff, norm))


class ShapeInfo:
    # count_by_dim = [0,0,0,0]
    """
    Class to capture information about individual shapes finally meshed by GMSH as independent objects.
    """

    _shapes_dim ={'Vertex': 0, 'Edge': 1, 'Face': 2, 'Solid': 3}

    def __init__(self, shape, i_reg=None, top=None, bot=None):
        self.shape = shape
        # self.dim = shapes_dim.get(type(shape).__name__)
        # assert self.dim is not None

        # self.dim_spec_id = self.count_by_dim[dim]
        # self.count_by_dim += 1

        if i_reg is None:
            self.free = False
        else:
            self.i_reg = i_reg
            self.top_iface = top
            self.bot_iface = bot
            self.free = True

    def set_shape(self, i_reg, top, bot):
        assert not self.free
        self.i_reg = i_reg
        self.top_iface = top
        self.bot_iface = bot
        self.free = True

    def dim(self):
        return self._shapes_dim.get(type(self.shape).__name__, None)

    def vert_curve(self, v_to_z, xy_to_u):
        """
        Make top/bot boundary of a vertical face in its UV coordinates from
        the self.curve_z, which is the full 3d curve of the edge set in Interface.make_shapes.
        TODO: set curve_z through a method.
        :param v_to_z: (min_z, max_z) ... bounds of the vertical face
        :param xy_to_u: (min_u, max_u)
        :return: Curve for U or V parameter.
        """
        z_min, z_max = v_to_z
        u_min, u_max = xy_to_u
        poles_uv = self.curve_z.poles.copy()
        poles_uv[:, 1] -= z_min
        poles_uv[:, 1] /= (z_max - z_min)
        poles_uv[:, 0] *= (u_max - u_min)
        poles_uv[:, 0] += u_min
        return bs.Curve(self.curve_z.basis, poles_uv)


class Curve(gs.Curve):
    pass


class SurfaceApproximation(gs.SurfaceApproximation):
    """
    TODO: allow user to determine configuration of the approximation
    """
    pass


class Surface(gs.Surface):
    """
    Represents a z(x,y) function surface given by grid of points, but optionaly approximated by a B-spline surface.
    """
    def __init__(self):
        """
        Construct a planar surface from a depth.
        """
        self.z_surface = None

    def init(self):
        """
        Initialize a B-spline surface.
        :param geom_file_base:
        :return:
        """
        self.z_surface = bspline_io.bs_zsurface_read(self.approximation)
        # Surface approx conatains transform
        #self.z_surface.transform(self.xy_transform)

    def make_bumpy_surface(self, z_transform):
        # load grid surface and make its approximation
        #
        # self.grid_surf = bs.GridSurface.load(self.grid_file)
        # self.approx_surf = bs_approx.surface_from_grid(self.grid_surf, (16, 16))
        # self.mat_xy = mat_xy = np.array(self.transform_xy)
        # mat_z = np.array(self.transform_z)
        # self.approx_surf.transform(mat_xy, mat_z)
        assert self.z_surface is not None
        surf = self.z_surface.get_copy()
        surf.transform(None, z_transform)
        return surf



    def make_flat_surface(self, xy_aabb, z_transform):
        # flat surface
        #z_const = z_transform[1]

        corners = np.zeros( (3, 3) )
        corners[[1, 0], 0] = xy_aabb[0][0]  # min X
        corners[2, 0] = xy_aabb[1][0]  # max X
        corners[[1, 2], 1] = xy_aabb[0][1]  # min Y
        corners[0, 1] = xy_aabb[1][1]  # max Y
        corners[:,2] = 0.0

        basis = bs.SplineBasis.make_equidistant(2, 1)

        poles = np.zeros( (3, 3, 1 ) )
        surf_z = bs.Surface( (basis, basis), poles)
        self.z_surface = bs.Z_Surface(corners[:, 0:2], surf_z)
        return self.make_bumpy_surface(z_transform)




    def plot_nodes(self, nodes):
        """
        Plot nodes with the surface boundary.
        :param nodes: array Nx2 of XY points
        """
        import_plotting()

        # plot nodes
        x_nodes = np.array(nodes)[:, 0]
        y_nodes = np.array(nodes)[:, 1]
        plt.plot(x_nodes, y_nodes, 'bo')

        # plot boundary
        vtx = np.array([[0, 1], [0, 0], [1, 0], [1, 1], [0, 1]])
        xy = self.grid_surf.uv_to_xy(vtx)

        # variants of the same (for debugging)
        # xy = self.approx_surf.uv_to_xy(vtx)
        # Just for bump surface:
        # xy = np.array([ self.mat_xy[0:2, 0:2].dot(v) + self.mat_xy[:,2] for v in xy_vtx ])
        # xy = self.grid_surf.quad

        plt.plot(xy[:, 0], xy[:, 1], color='green')
        plt.show()




class Segment(gs.Segment):
    pass

class Polygon(gs.Polygon):
    pass

class Topology(gs.Topology):
    pass
    # """
    # TODO:
    # - support for polygons with holes, here and in creation of faces and solids
    # """
    # def check(self, nodeset):
    #     """
    #     Check that topology is compatible with given nodeset.
    #     Check and possibly fix orientation of polygons during first call.
    #     :param nodeset:
    #     :return:
    #     """
    #     if hasattr(self, 'n_nodes'):
    #         # already checked with other nodeset, just check nodeset size
    #         assert len(nodeset.nodes) == self.n_nodes, \
    #             "Nodeset (id: {}) size {} is not compatible with topology (id: {}) with size {}"\
    #             .format(nodeset.id, len(nodeset.nodes), self.id, self.n_nodes)
    #         return
    #
    #     nodes = nodeset.nodes
    #     self.n_nodes = len(nodes)
    #     self.n_segments = len(self.segments)
    #     for segment in self.segments:
    #         for nid in segment.node_ids:
    #             assert 0 <= nid < len(nodes), "Node ID: {} of topology (id: {}) is out of nodeset (id: {}".format(nid, self.id, nodeset.id)
    #
    #     for poly in self.polygons:
    #         self.orient_polygon(poly)
    #
    #
    # def orient_polygon(self, poly):
    #     """
    #     Find orientation of polygon segments so that they have same orientation within the polygon.
    #     TODO: Make orientation counter clock wise.
    #     :param poly:
    #     :return:
    #     """
    #     last_node = None
    #     first_node = poly.segment_ids
    #     oriented_ids = []
    #     segs = poly.segment_ids
    #     for i_seg, i_seg_next in zip(segs, segs[1:] + segs[:1]):
    #         segment = self.segments[i_seg]
    #         next_segment = self.segments[i_seg_next]
    #         nodes = segment.node_ids
    #         if nodes[1] in next_segment.node_ids:
    #             oriented_ids.append( self.code_oriented_segment(i_seg, False) )
    #         else:
    #             assert nodes[0] in next_segment.node_ids, "Wrong order of polygon edges."
    #             oriented_ids.append(self.code_oriented_segment(i_seg, True))
    #     poly.segment_ids = oriented_ids
    #
    #
    # def code_oriented_segment(self, id_seg, reversed):
    #     """
    #     Coded segment id with orientation.
    #     :param id_seg:
    #     :param reversed:
    #     :return:
    #     """
    #     return id_seg + reversed*self.n_segments
    #
    #
    # def orient_segment(self, id_seg):
    #     """
    #     Decode segment_id withi polygon to the true segment ID end orientation.
    #     :param id_seg:
    #     :return:
    #     """
    #     if id_seg >= len(self.segments):
    #         return id_seg - self.n_segments, True
    #     else:
    #         return id_seg, False

class NodeSet(gs.NodeSet):
    pass


class Interface:
    """
    Interfaceis a gluing object between layers joined to a common surface.
    It maps nodes onto surface, create complete decompositions and their intersection, and produce all
    shapes laying on the surface.
    """
    def init(self, lg):
        """
        Interface is bounded to a single surface.
        :param surface:
        """
        if self.surface_id is None:
            self._surface = None
        else:
            self._surface = lg.surfaces[self.surface_id]
        self.common_decomp = None
        self.decompositions = {}

    def add_decomposition(self, nodes, topology):
        """
        Related layers (at most two stratum layers and one fracture layer) bound their decompositions given as nodeset
        and topology to the interface.
        :param nodes: 2d Nodes
        :param topology: Topology object.
        :return: Id of the decomposition within interface. Necessary to get right polygon map later on.
        """
        if not topology.id in self.decompositions:
            decomp = polygons_io.deserialize(nodes, topology)
            self.decompositions[topology.id] = decomp
        else:
            decomp = self.decompositions[topology.id]
        return decomp

    def _finish_init(self):
        """
        Finalize interface initialization after decompositions are added.
        """
        if self.common_decomp is not None:
            return

        decomps = list(self.decompositions.values())
        decomp_ids = list(self.decompositions.keys())
        self.common_decomp, all_maps = polygons.intersect_decompositions(decomps)
        #plot_polygons.plot_polygon_decomposition(self.common_decomp)
        # make subpolygon lists
        self.subobj_lists={}
        for decomp_idx, one_decomp_maps in enumerate(all_maps):
            subobjs=[{}, {}, {}]
            for dim, one_dim_map in enumerate(one_decomp_maps):
                for new_id, orig_id in one_dim_map.items():
                    subobjs[dim].setdefault(orig_id, list())    # make new instance every time
                    subobjs[dim][orig_id].append(new_id)

            # sort subsegment lists
            orig_decomp = decomps[decomp_idx]
            for orig_seg_id, seg_list in subobjs[1].items():
                if orig_seg_id is None:
                    continue
                pt_to_seg={}
                for seg_id in seg_list:
                    seg = self.common_decomp.segments[seg_id]
                    pt_id = seg.vtxs[polygons.out_vtx].id
                    pt_to_seg[pt_id] = seg
                new_seg_list = []
                pt_id = orig_decomp.segments[orig_seg_id].vtxs[polygons.out_vtx].id
                new_pt_id = subobjs[0][pt_id]
                assert len(new_pt_id) == 1
                new_pt_id = new_pt_id[0]
                while new_pt_id in pt_to_seg:
                    seg = pt_to_seg[new_pt_id]
                    new_seg_list.append(seg.id)
                    new_pt_id = seg.vtxs[polygons.in_vtx].id
                subobjs[1][orig_seg_id] = new_seg_list
            self.subobj_lists[decomp_ids[decomp_idx]] = subobjs


        #self.common_decomp = decomps[0]
        nodes_xy = { pt.id: pt.xy for pt in self.common_decomp.points.values() }
        self._check_nodes(list(nodes_xy.values()))
        self.nodes = { id: (x, y, self.approx_eval_z(x, y) ) for id, (x,y) in nodes_xy.items() }


    def make_shapes(self):
        """
        Make dictionaries of shapes for points, segments, polygons in common decomposition.
        :return:
        """
        self._finish_init()

        decomp = self.common_decomp
        # TODO: compute Z (use intersections for inclined vertical segments)
        self.vertices = { id: ShapeInfo(bw.Vertex(node)) for id, node in self.nodes.items() }

        for id, vert in self.vertices.items():
            vert.shape.dbg_id = id

        self.edges = {}
        for segment in decomp.segments.values():
            #nodes_id, surface_id = segment
            pa, pb = segment.vtxs
            edge = bw.Edge( [self.vertices[pa.id].shape, self.vertices[pb.id].shape] )
            curve_z = self.add_curve_to_edge(edge)
            si = ShapeInfo(edge)
            si.curve_z = curve_z
            self.edges[segment.id] =  si

        self.faces = {}
        #n_segments = len(self.topology.segments)
        for poly in decomp.polygons.values():
            #segment_ids, surface_id = poly      # segment_id > n_segments .. reversed edge
            wires = [self._make_bw_wire(poly.outer_wire)]
            for hole in poly.outer_wire.childs:
                wires.append(self._make_bw_wire(hole).m())

            face = bw.Face(wires, surface = self.bw_surface)
            self.faces[poly.id] = ShapeInfo(face)

    def _make_bw_wire(self, decomp_wire):
        """
        Make shape Wire from decomposition wire.
        """
        edges = []
        for seg, side in decomp_wire.segments():
            reversed = (side == polygons.right_side)
            edge = self.edges[seg.id].shape
            if reversed:
                edges.append(edge.m())
            else:
                edges.append(edge)
        return bw.Wire(edges)

    def get_polygon_division(self, decomp_id, poly_id):
        """
        For given decomposition_id (returned by add_decomposition) and polygon_id
        withing that decomposition, return a list of polygon ids from common decomposition
        that form a subdivision of the given polygon. These polygon IDs match shape ids in self.faces.
        :param decomp_id: Original decomposition ID.
        :param poly_id: Polygon ID within original deocmposition.
        :return: Subpolygons ids in common decomposition.
        """
        return self.subpoly_lists[decomp_id][poly_id]

    def interpolate_nodes(self, a_iface, a_nodes, b_iface, b_nodes):
        """
        Used by InterpolatedNodeSet.
        """
        assert len(a_nodes) == len(b_nodes)
        nodes=[]
        for (ax, ay), (bx, by) in zip(a_nodes, b_nodes):
            az = a_iface.approx_eval_z(ax, ay)
            bz = b_iface.approx_eval_z(bx, by)
            line = ( (ax, ay, az), (bx, by, bz) )
            x,y,z = self.line_intersect( *line )
            nodes.append( (x,y,z) )
        return nodes

    def iter_shapes(self):
        """
        Generator of all shapes on the interface.
        """
        for s_list in [ self.vertices, self.edges, self.faces ]:
            for shp in s_list.values():
                yield shp


    def _check_nodes(self, nodes):
        """
        :param nodes:
        :return:
        """
        if self._surface is None:
            # planar surface
            nod = np.array(nodes)
            nod_aabb = (np.amin(nod, axis=0), np.amax(nod, axis=0))
            self._surface = Surface()

            # TODO: remove after test correctness of Layer editor
            if self.transform_z[1] != self.elevation:
                self.transform_z = [ 1.0, self.elevation]
            self.surface_approx = self._surface.make_flat_surface(nod_aabb, self.transform_z)

        else:
            # bumpy surface
            self.surface_approx = self._surface.make_bumpy_surface(self.transform_z)

            uv_nodes = self.surface_approx.xy_to_uv(np.array(nodes))
            for i, uv in enumerate(uv_nodes):
                if not ( 0.0 < uv[0] < 1.0 and 0.0 < uv[1] < 1.0 ):
                    raise IndexError("Node {}: {} is out of surface domain, uv: {}".format(i, nodes[i], uv))
        self.bw_surface = bw.surface_from_bs(self.surface_approx.make_full_surface())


    def approx_eval_z(self, x, y):
        return self.surface_approx.z_eval_xy_array(np.array([[x, y]]))[0]


    # TODO:
    # - test

    @staticmethod
    def interpol(a, b, t):
        return (t * b + (1-t) * a)

    def line_intersect(self, a, b):
        # TODO: Compute true intersection.
        t = 0.5
        z = (a[2] + b[2]) / 2
        z_diff = 1.0
        tol = np.abs(a[2] - b[2]) * 0.001
        while abs(z_diff) > tol:
            x = self.interpol(a[0], b[0], t)
            y = self.interpol(a[1], b[1], t)
            z_new = self.approx_eval_z(x, y)
            z_diff = z - z_new
            z = z_new
            t = (z - b[2]) / (a[2] - b[2])
        return (x, y, z)


    def add_curve_to_edge(self, edge):
        """
        Make the projection curve for an edge on the surface.
        :param edge: BRepWriter Edge object
        :return:
        """
        axyz, bxyz = edge.points()

        n_points = 16
        x_points = np.linspace(axyz[0], bxyz[0], n_points)
        y_points = np.linspace(axyz[1], bxyz[1], n_points)
        xy_points = np.stack( (x_points, y_points), axis =1)
        xyz_points = self.surface_approx.eval_xy_array(xy_points)
        curve_xyz = bs_approx.curve_from_grid(xyz_points)
        start, end = curve_xyz.eval_array(np.array([0, 1]))
        check_point_tol( start, axyz, 1e-3)
        check_point_tol( end, bxyz, 1e-3)
        edge.attach_to_3d_curve((0.0, 1.0), bw.curve_from_bs(curve_xyz))

        # TODO: make simple line approximation
        xy_points = np.array([ axyz[0:2], bxyz[0:2]])
        uv_points = self.surface_approx.xy_to_uv(xy_points)
        curve_uv = bs_approx.line( uv_points )
        start, end = self.surface_approx.eval_array(curve_uv.eval_array(np.array([0, 1])))
        check_point_tol( start, axyz, 1e-3)
        check_point_tol( end, bxyz, 1e-3)
        edge.attach_to_2d_curve((0.0, 1.0), bw.curve_from_bs(curve_uv), self.bw_surface)

        # vertical curve
        poles_z = curve_xyz.poles[:, 2].copy()
        x_diff, y_diff, z_diff = np.abs(bxyz - axyz)
        if x_diff > y_diff:
            axis = 0
        else:
            axis = 1
        poles_t = curve_xyz.poles[:, axis].copy()
        poles_t -= axyz[axis]
        poles_t /= (bxyz[axis] - axyz[axis])
        poles_tz = np.stack( (poles_t, poles_z), axis=1 )
        # overhang = 0.1
        # scale = (1 - 2*overhang) / (bxyz[1] - axyz[1])
        # poles_tz[:, 0] -= axyz[1]
        # poles_tz[:, 0] *= scale
        # poles_tz[:, 0] += overhang


        #poles_tz[:, 0] -= axyz[1]
        #poles_tz[:, 0] /= (bxyz[1] - axyz[1])

        return bs.Curve(curve_xyz.basis, poles_tz)




class InterfaceNodeSet(gs.InterfaceNodeSet):

    def init(self, lg):
        self.interface = lg.interfaces[self.interface_id]
        self.nodeset = lg.node_sets[self.nodeset_id]
        self.topology = lg.topologies[self.nodeset.topology_id]
        #self.topology.check(self.nodeset)

    def make_interface(self, lg ):
        self.init(lg)
        interface = lg.interfaces[self.interface_id]
        self.decomp = interface.add_decomposition(self.nodeset.nodes, self.topology)
        return interface


class InterpolatedNodeSet(gs.InterpolatedNodeSet):

    def make_interface(self, lg):
        interface = lg.interfaces[self.interface_id]
        #surface = lg.surfaces[self.interface_id]
        a, b = self.surf_nodesets
        a.init(lg)
        b.init(lg)
        assert a.topology.id == b.topology.id
        self.topology = a.topology
        if a.nodeset_id == b.nodeset_id:
            self.nodes = a.nodeset.nodes
        else:
            assert False, "Interpolation ofr different nodesets not supported yet."
            assert a.interface_id != b.interface_id
            self.nodes = interface.interpolate_nodes(a.interface, a.nodeset.nodes, b.interface, b.nodeset.nodes)
        self.decomp = interface.add_decomposition(self.nodes, self.topology)
        return interface


class Region(gs.Region):

    def init(self, topo_dim, extrude):
        #assert topo_dim == self.topo_dim,
        if self.not_used:
            return
        assert self.dim == topo_dim + extrude,\
            "Region ('{}') dimension ({})  do not match layer dimension ({}).".format( self.name, self.dim, topo_dim + extrude)

        # fix names of boundary regions
        if self.boundary:
            if self.name[0] != '.':
                self.name = '.' + self.name
        else:
            while len(self.name) > 0 and self.name[0] == '.':
                self.name = self.name[1:]
            assert len(self.name) > 0, "Empty name of region after removing leading dots."
                

    def is_active(self, dim):
        active = not self.not_used
        if active:
            assert dim == self.dim, "Can not create shape of dim: {} in region '{}' of dim: {}.".format(dim, self.name, self.dim)
        return active

#class GeoLayer(gs.GeoLayer):
#    pass

def make_layer_region_maps(layer, regions, extrude):
    """
    Return list of obj->region map for every dimension.
    This replace just list of region ids on the input.
    :param layer:
    :param regions:
    :param extrude:
    :return:
    """
    top_decomp = layer.top.decomp
    #bot_decomp = layer.bottom.decomp
    #assert top_decomp == bot_decomp

    id_to_region = []
    region_id_lists = [layer.node_region_ids, layer.segment_region_ids, layer.polygon_region_ids]
    top_objs = [top_decomp.points.values(), top_decomp.segments.values(), top_decomp.polygons.values()]
    for dim, (reg_list, obj_list) in enumerate(zip(region_id_lists, top_objs)):
        reg_map={}
        for reg, obj in zip(reg_list, obj_list):
            i_reg = reg_list[obj.index]
            regions[i_reg].init(topo_dim=dim, extrude=extrude)
            reg_map[obj.id] = regions[i_reg]
        id_to_region.append(reg_map)
    return id_to_region

class FractureLayer(gs.FractureLayer):
    def init(self, lg):
        self.i_top = self.top.make_interface(lg)
        # Top interface.
        self.topology = self.top.topology
        # Common topology.
        self.regions = make_layer_region_maps(self, lg.regions, False)
        # List of dictionaries.
        # [ points_ids to regions, segment_ids to regions, polygon_ids to regions]

    def make_shapes(self):
        """
        Make shapes in main
        :return:
        """

        # no point regions
        #for i, i_reg in enumerate(self.node_region_ids):
        #    if self.regions[i_reg].is_active(0):
        #        shapes.append( (self.i_top.vertices[i], i_reg) )
        decomp = self.top.decomp
        obj_maps = self.i_top.subobj_lists[self.top.topology.id]
        for seg in decomp.segments.values():
            reg = self.regions[1][seg.id]
            if reg.is_active(1):
                for sub_seg_id in obj_maps[1][seg.id]:
                    self.i_top.edges[sub_seg_id].set_shape(reg.id, self.i_top, self.i_top)

        for poly in decomp.polygons.values():
            reg = self.regions[2][poly.id]
            if reg.is_active(2):
                for sub_poly_id in obj_maps[2][poly.id]:
                    self.i_top.faces[sub_poly_id].set_shape(reg.id, self.i_top, self.i_top)
        return []

class StratumLayer(gs.StratumLayer):
    def init(self, lg):
        self.i_top = self.top.make_interface(lg)
        self.i_bot = self.bottom.make_interface(lg)
        assert self.top.topology.id == self.bottom.topology.id
        self.topology = self.top.topology
        self.regions = make_layer_region_maps(self, lg.regions, True)
        #for tdim, reg_list in enumerate([self.node_region_ids, self.segment_region_ids, self.polygon_region_ids]):
        #    for i_reg in reg_list:
        #        self.regions[i_reg].init(topo_dim=tdim, extrude = True)

    def plot_vert_face(self, v_to_z, si_top, si_bot):
        #import_plotting()
        top_curve = si_top.vert_curve(v_to_z)
        bot_curve = si_bot.vert_curve(v_to_z)
        bs_plot.plot_curve_2d(top_curve, poles=True)
        bs_plot.plot_curve_2d(bot_curve, poles=True)
        plt.show()



    def make_vert_bw_surface(self, top_edges, bot_edges, edge_start, edge_end):
        """
        Make vertical surface surface from set of 4 boundary edges.

        Sequence of edge shape info, which are subdivision (after partition) of a single edge:
        :param top_edges: Subdivision of the top edge (V01 - > V11)
        :param bot_edges: Subdivision of the bottom edge (V00 -> V10)

        Vertical edges oriented from bottom to top:
        :param edge_start: Connecting start points of (top and bot edges), V00 -> V01
        :param edge_end: Connecting end points of (top and bot edges), V10 -> V11
        :return: BW surface object.
        """
        #

        top_boxes = [edg_si.curve_z.aabb() for edg_si in top_edges]
        bot_boxes = [edg_si.curve_z.aabb() for edg_si in bot_edges]

        # Z range
        top_z = max([ box[:, 1].max() for box in top_boxes]) + 1.0
        bot_z = min([ box[:, 1].min() for box in bot_boxes]) - 1.0

        # XYZ of corners, vUV,
        # U is horizontal start to end, V is vertical bot to top
        edg_start_vtxs = np.array(edge_start.shape.points())
        edg_end_vtxs = np.array(edge_end.shape.points())
        v00, v01 = edg_start_vtxs.copy()
        v10, v11 = edg_end_vtxs.copy()
        v00[2] = v10[2] = bot_z
        v11[2] = v01[2] = top_z

        # allow just vertical extrusion, same XY on bot and top
        assert la.norm(v00[0:2] - v01[0:2]) < 1e-10
        assert la.norm(v10[0:2] - v11[0:2]) < 1e-10
        surf = bs_approx.plane_surface([v00, v10, v01], overhang=0.0)
        bw_surf = bw.surface_from_bs(surf)

        v_to_z = [bot_z, top_z]
        #self.plot_vert_face(v_to_z, si_top, si_bot)

        v00, v01 = edg_start_vtxs.copy()
        v10, v11 = edg_end_vtxs.copy()

        # attach 2D curves to horizontal edges
        xy_vtxs = (v00[0:2], v10[0:2])
        self._curve_for_horizontal_edges(top_edges, v_to_z, xy_vtxs, bw_surf)
        self._curve_for_horizontal_edges(bot_edges, v_to_z, xy_vtxs, bw_surf)


        # check precision of corners
        # xyz_v_top = surf.eval_array(uv_v_top)
        # xyz_v_bot = surf.eval_array(uv_v_bot)
        # v00, v10 = si_bot.shape.points()
        # v01, v11 = si_top.shape.points()
        # for new, orig in zip( list(xyz_v_bot) + list(xyz_v_top), [v00, v10, v01, v11]):
        #     check_point_tol(new, orig, 1e-3)


        # attach 2D and 3D curves to vertical edges
        self._curve_for_vertical_edge(edge_start, 0.0,  v_to_z, bw_surf)
        self._curve_for_vertical_edge(edge_end, 1.0, v_to_z, bw_surf)

        return bw_surf

    def _curve_for_horizontal_edges(self, edge_list, v_to_z, xy_vtxs, bw_surf):
        """
        Scale and attach boundary curves of edges to the surface bounded by the edges.
        U - horizontal parameter, V - vertical parameter; parameters of the vertical surface
        'bw_surf'. Boundary curves (U->V) are part of the shape info objects.


        :param edge_list: List of edge shape info objects, forming a subdivision of a segment
         with endpoints 'xy_vtxs'
        :param v_to_z: Mapping vertica V parameter to real Z coordinate.
        :param xy_vtxs: Segment end points.
        :param bw_surf: Surface to which attache the
        :return: None
        """
        axis = np.argmax(np.abs(xy_vtxs[1] - xy_vtxs[0]))
        axis_diff = xy_vtxs[1][axis] - xy_vtxs[0][axis]
        for edg_si in edge_list:
            # Compute U range of the sub-edge, edg_si relative to the original edge
            # with endpoints xy_vtxs
            pts = edg_si.shape.points()
            xy_to_u = [ (pt[axis] - xy_vtxs[0][axis]) / axis_diff for pt in pts]
            assert 0.0 <= xy_to_u[0] <= 1.0
            assert 0.0 <= xy_to_u[1] <= 1.0
            boundary_curve = edg_si.vert_curve(v_to_z, xy_to_u)
            uv_vtx = boundary_curve.eval_array(np.array([0, 1]))
            # Fix errors in U coordinate which should be a sequence from 0 to 1.
            uv_vtx[:, 0] = np.clip(uv_vtx[:,0], 0.0, 1.0)

            if not (np.all( 0 <= uv_vtx ) and np.all( uv_vtx <= 1)):
                raise Exception("Top point < bottom point, for layer id = {}. Z-range: {}. {}"\
                .format(self.id, v_to_z, uv_vtx))
            print("H-UV: ", uv_vtx)
            #xyz_v_bot = surf.eval_array(uv_v_bot)

            edg_si.shape.attach_to_2d_curve((0.0, 1.0), bw.curve_from_bs(boundary_curve), bw_surf )

    def _curve_for_vertical_edge(self, v_edge, u_coord, v_to_z, bw_surf):
        edg_vtxs = np.array(v_edge.shape.points())
        v0, v1 = edg_vtxs
        curve = bs_approx.line( [v0, v1] )
        v_edge.shape.attach_to_3d_curve((0.0, 1.0), bw.curve_from_bs(curve))
        v_vtxs = (edg_vtxs[:, 2] - v_to_z[0]) / (v_to_z[1] - v_to_z[0])

        #print("V-UV: ", list(zip(u_vtxs, v_vtxs)))
        v_edge.shape.attach_to_plane(bw_surf, [u_coord, v_vtxs[0]], [u_coord, v_vtxs[1]])


    def make_shapes(self):
        shapes = []
        top_subobjs = self.i_top.subobj_lists[self.top.topology.id]
        bot_subobjs = self.i_bot.subobj_lists[self.bottom.topology.id]

        vert_edges={}

        # TODO: vertical edges and faces
        for id, pt in self.top.decomp.points.items():
            assert len(top_subobjs[0][id]) == 1
            assert len(bot_subobjs[0][id]) == 1
            top_new_pt = self.i_top.vertices[top_subobjs[0][id][0]]
            bot_new_pt = self.i_bot.vertices[bot_subobjs[0][id][0]]


            edge = bw.Edge( [bot_new_pt.shape, top_new_pt.shape] )
            edge.implicit_curve()
            edge_info = ShapeInfo(edge)
            vert_edges[id] = edge_info
            reg = self.regions[0][id]
            if reg.is_active(1):
                edge_info.set_shape( reg.id, self.i_top, self.i_bot)
            shapes.append(edge_info)

        assert len(vert_edges) == len(self.top.decomp.points) #, "n_vert_edges: %d n_nodes: %d"%(len(vert_edges), self.topology.n_nodes)
        assert len(vert_edges) == len(self.node_region_ids)

        vert_faces = {}
        for id, segment in self.top.decomp.segments.items():
            # make face oriented to the right side of the segment when looking from top
            edge_start = vert_edges[segment.vtxs[0].id]
            edge_end = vert_edges[segment.vtxs[1].id]

            wire_edges = [edge_start.shape.m()]
            bot_edges = []
            for sub_edge_id in bot_subobjs[1][segment.id]:
                edge = self.i_bot.edges[sub_edge_id]
                wire_edges.append(edge.shape)
                bot_edges.append(edge)
            wire_edges.append(edge_end.shape)
            top_edges = []
            for sub_edge_id in reversed(top_subobjs[1][segment.id]):
                edge = self.i_top.edges[sub_edge_id]
                wire_edges.append(edge.shape.m())
                top_edges.append(edge)

            # make planar surface
            # attach curves to top and bot edges
            vert_surface = self.make_vert_bw_surface(top_edges, bot_edges, edge_start, edge_end)


            # edges oriented counter clockwise = positively oriented face
            wire = bw.Wire(wire_edges)
            face = bw.Face([wire], surface = vert_surface)
            face_info = ShapeInfo(face)
            vert_faces[id] = face_info
            reg = self.regions[1][id]
            if reg.is_active(2):
                face_info.set_shape(reg.id, self.i_top, self.i_bot)
            shapes.append(face_info)

        assert len(vert_faces) == len(self.top.decomp.segments)
        assert len(vert_faces) == len(self.segment_region_ids)

        for id, polygon in self.top.decomp.polygons.items():
            if polygon.is_outer_polygon():
                continue
            #segment_ids = polygon.segment_ids  # segment_id > n_segments .. reversed edge

            # we orient all faces inwards (assuming normal oriented up for counter clockwise edges, right hand rule)
            # assume polygons oriented upwards
            faces = []
            wires = [ polygon.outer_wire ] + list(polygon.outer_wire.childs)
            for wire in wires:
                for seg, side in wire.segments():
                    if side == polygons.right_side:
                        faces.append( vert_faces[seg.id].shape.m() )
                    else:
                        faces.append( vert_faces[seg.id].shape )

            for subpoly_id in top_subobjs[2][polygon.id]:
                faces.append( self.i_top.faces[subpoly_id].shape )
            for subpoly_id in bot_subobjs[2][polygon.id]:
                faces.append( self.i_bot.faces[subpoly_id].shape.m() )

            shell = bw.Shell( faces )
            solid = bw.Solid([shell])
            solid_info = ShapeInfo(solid)
            reg = self.regions[2][id]
            if reg.is_active(3):
                solid_info.set_shape(reg.id, self.i_top, self.i_bot)
            shapes.append(solid_info)

        return shapes

class UserSupplement(gs.UserSupplement):
    pass


class LayerGeometry(gs.LayerGeometry):

    el_type_to_dim = {15: 0, 1: 1, 2: 2, 4: 3}

    """
        - create BREP B-spline approximation from Z-grids (Bapprox)
        - load other surfaces
        - create BREP geometry from it (JB)

        - write BREP geometry into a file (JE)
        - visualize BREP geometry or part of it
        - call GMSH to create the mesh (JB)
        //- convert mesh into GMSH file format from Netgen
        - name physical groups (JB)
        - scale the mesh nodes verticaly to fit original interfaces (JB, Jakub)
        - find rivers and assign given regions to corresponding elements (JB, Jakub)
    """

    @staticmethod
    def set_ids(xlist):
        """
        Set .id attribute to all items of the xlist.
        :param xlist:
        :return:
        """
        for i, item in enumerate(xlist):
            item.id = i

    def init(self):
        # keep unique interface per surface
        self.brep_shapes=[]     # Final shapes in top compound to being meshed.
        self.min_step = np.inf
        self.max_step = 0
        self.set_ids(self.surfaces)
        self.set_ids(self.regions)
        #self.interfaces = [ Interface(surface) for surface in self.surfaces ]
        self.set_ids(self.topologies)
        #self.set_ids(self.nodesets)

        # funish initialization of interfaces
        for iface in self.interfaces:
            iface.init(self)

        # load and construct grid surface functions
        for surf in self.surfaces:
            surf.init()


        # initialize layers, neigboring layers refer to common interface
        for id, layer in enumerate(self.layers):
            layer.id = id
            layer.init(self)

    def construct_brep_geometry(self):
        """
        Algorithm for creating geometry from Layers:

        3d_region = CompoundSolid of Solids from extruded polygons
        2d_region, .3d_region = Shell of:
            - Faces from extruded segments (vertical 2d region)
            - Faces from polygons (horizontal 2d region)
        1d_region, .2d_region = Wire of:
            - Edges from extruded points (vertical 1d regions)
            - Edges from segments (horizontal 1d regions)

        1. on every noncompatible interface make a subdivision of segments and polygons for all topologies on it
            i.e. every segment or polygon cen get a list of its sub-segments, sub-polygons

        2. Create Vertexes, Edges and Faces for points, segments and polygons of subdivision on interfaces, attach these to
           point, segment and polygon objects of subdivided topologies

        3. For every 3d layer:

                for every segment:
                    - create face, using top and bottom subdivisions, and pair of verical edges as boundary
                for every polygon:
                    - create list of faces for top and bottom using BREP shapes of polygon subdivision on top and bottom
                    - make shell and solid from faces of poligon side segments and top and bottom face

                for every region:
                    create compound object from its edges/faces/solids

        4. for 2d layer:
                for every region:
                    make compoud of subdivision BREP shapes

        """

        for iface in self.interfaces:
            iface.make_shapes()

        self.split_to_blocks()

        # self.vertices={}            # (interface_id, interface_node_id) : bw.Vertex
        # self.extruded_edges = {}    # (layer_id, node_id) : bw.Edge, orented upward, Loacl to Layer

        self.all_shapes = []
        self.free_shapes = []

        for block in self.blocks:
            for layer in block:
                self.all_shapes += layer.make_shapes()

        for i_face in self.interfaces:
            for shp in i_face.iter_shapes():
                self.all_shapes.append(shp)

        self.free_shapes = [shp_info for shp_info in self.all_shapes if shp_info.free]
        # sort down from solids to vertices
        self.free_shapes.sort(key=lambda shp: shp.dim(), reverse=True)
        free_shapes = [shp_info.shape for shp_info in self.free_shapes]

        compound = bw.Compound(free_shapes)
        compound.set_free_shapes()
        self.brep_file = os.path.abspath(self.filename_base + ".brep")
        with open(self.brep_file, 'w') as f:
            bw.write_model(f, compound, bw.Location())

    def make_gmsh_shape_dict(self):
        """
        Construct a dictionary self.gmsh_shape_dict, mapping the pair (dim, gmsh_object_id) -> shape info object
        :return:
        """
        # ignore shapes without ID - not part of the output
        output_shapes = [si for si in self.all_shapes if hasattr(si.shape, 'id')]

        # prepare dict: (dim, shape_id) : shape info
        output_shapes.sort(key=lambda si: si.shape.id, reverse=True)
        shape_by_dim = [[] for i in range(4)]
        for shp_info in output_shapes:
            dim = shp_info.dim()
            shape_by_dim[dim].append(shp_info)

        self.gmsh_shape_dist = {}
        for dim, shp_list in enumerate(shape_by_dim):
            for gmsh_shp_id, si in enumerate(shp_list):
                self.gmsh_shape_dist[(dim, gmsh_shp_id + 1)] = si

    def set_free_si_mesh_step(self, si, step):
        """
        Set the mesh step to the free SI (root of local DFS tree).
        :param si: A free shape info object
        :param step: Meash step from corresponding region.
        :return:
        """
        if step <= 0.0:
            step = self.global_mesh_step
        self.min_step = min(self.min_step, step)
        self.max_step = max(self.max_step, step)
        si.mesh_step = step

    def distribute_mesh_step(self):
        """
        For every free shape:
         1. get the mesh step from the region
         2. pass down through its tree using DFS
         3. set the mesh_step  to all child vertices, take minimum of exisiting and new mesh_step
        :return:
        """
        print("distribute mesh\n")
        self.compute_bounding_box()
        self.global_mesh_step = self.mesh_step_estimate()

        # prepare map from shapes to their shape info objs
        # initialize mesh_step of individual shape infos
        shape_dict = {}
        for shp_info in self.all_shapes:
            shape_dict[shp_info.shape] = shp_info
            shp_info.mesh_step = np.inf
            shp_info.visited = -1

        # Propagate mesh_step from the free_shapes to vertices via DFS
        # use global mesh step if the local mesh_step is zero.
        for i_free, shp_info in enumerate(self.free_shapes):
            self.set_free_si_mesh_step(shp_info, self.regions[shp_info.i_reg].mesh_step)
            shape_dict[shp_info.shape].visited = i_free
            stack = [shp_info.shape]
            while stack:

                shp = stack.pop(-1)
                print("shp: {} id: {}\n".format(type(shp), shp.id))
                for sub in shp.subshapes():
                    if isinstance(sub, (bw.Vertex, bw.Edge, bw.Face, bw.Solid)):
                        if shape_dict[sub].visited < i_free:
                            shape_dict[sub].visited = i_free
                            stack.append(sub)
                    else:

                        stack.append(sub)
                if isinstance(shp, bw.Vertex):
                    shape_dict[shp].mesh_step = min(shape_dict[shp].mesh_step, shp_info.mesh_step)

        self.min_step *= 0.2
        self.vtx_char_length = []
        for (dim, gmsh_shp_id), si in self.gmsh_shape_dist.items():
            if dim == 0:
                mesh_step = si.mesh_step
                if mesh_step == np.inf:
                    mesh_step = self.global_mesh_step
                self.vtx_char_length.append((gmsh_shp_id, mesh_step))



            # debug listing
            # xx=[ (k, v.shape.id) for k, v in self.shape_dict.items()]
            # xx.sort(key=lambda x: x[0])
            # for i in xx:
            #    print(i[0][0], i[0][1], i[1])

    def split_to_blocks(self):
        blocks=[]
        block=[]
        last_id=None
        for layer in self.layers:
            if last_id == None:
                last_id = layer.topology.id
            if layer.topology.id == last_id:
                block.append(layer)
            else:
                last_id = layer.topology.id
                blocks.append(block)
                block=[layer]
        blocks.append(block)
        self.blocks=blocks

    def compute_bounding_box(self):
        min_vtx = np.ones(3) * (np.inf)
        max_vtx = np.ones(3) * (-np.inf)
        assert len(self.all_shapes) > 0, "Empty list of shapes to mesh."
        for si in self.all_shapes:
            if hasattr(si.shape, 'point'):
                min_vtx = np.minimum(min_vtx, si.shape.point)
                max_vtx = np.maximum(max_vtx, si.shape.point)
        assert np.all(min_vtx < np.inf)
        assert np.all(max_vtx > -np.inf)
        self.aabb = [ min_vtx, max_vtx ]


    def mesh_step_estimate(self):
        char_length = np.max(self.aabb[1] - self.aabb[0])
        mesh_step = char_length / 20
        print("Char length: {} mesh step: {}", char_length, mesh_step)
        return mesh_step

    def call_gmsh(self, mesh_step):
        """

        :param mesh_step:
        :return:

        """
        if mesh_step == 0.0:
            mesh_step = self.mesh_step_estimate()
        self.geo_file = self.filename_base + ".tmp.geo"
        with open(self.geo_file, "w") as f:
            print(r'SetFactory("OpenCASCADE");', file=f)
            # print(r'Mesh.Algorithm = 2;', file=f)
            """
            TODO: GUI interface for algorithm selection and element optimizaion.
            Related options:
            Mesh.Algorithm
            2D mesh algorithm (1=MeshAdapt, 2=Automatic, 5=Delaunay, 6=Frontal, 7=BAMG, 8=DelQuad)

            Mesh.Algorithm3D
            3D mesh algorithm (1=Delaunay, 2=New Delaunay, 4=Frontal, 5=Frontal Delaunay, 6=Frontal Hex, 7=MMG3D, 9=R-tree)
            """
            """
            TODO: ? Meaning of char length limits. Possibly to prevent to small elements at intersection points,
            they must be derived from min and max mesh step.
            """
            print(r'Mesh.CharacteristicLengthMin = %s;'% self.min_step, file=f)
            print(r'Mesh.CharacteristicLengthMax = %s;'% self.max_step, file=f)
            # rand_factor has to be increased when the triangle/model ratio
            # multiplied by rand_factor approaches 'machine accuracy'
            rand_factor = 1e-14 * np.max(self.aabb[1] - self.aabb[0]) / self.min_step
            print(r'Mesh.RandomFactor = %s;'%rand_factor , file=f)
            print(r'ShapeFromFile("%s")' % self.brep_file, file=f)

            for id, char_length in self.vtx_char_length:
                print(r'Characteristic Length {%s} = %s;' % (id, char_length), file=f)

        from subprocess import call
        gmsh_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../gmsh/gmsh.exe")
        if not os.path.exists(gmsh_path):
            gmsh_path = "gmsh"
        #call([gmsh_path, "-3", "-rand 1e-10", self.geo_file])
        call([gmsh_path, "-3",  self.geo_file])

    def deform_mesh(self):
        """
        In fact three different algorithms are necessary:
        I. Modification of extruded mesh, surfaces are horizontal planes.
        II. Small modification of curved mesh, modify just surface nodes and possibly
            small number of neighbouring elements.
        III. Big modification o curved mesh, need to evaluate discrete surface. Line/triangle intersection, need BIH.
        :return:
        """
        # The I. algorithm:
        # new empty nodes list
        # go through volume elements; every volume region should have reference to its top and bot interface;
        # move nodes of volume element

        assert False, "Mesh deformation code must be revisited."

        nodes_shift = { id: [] for id, el in self.mesh.nodes.items()}
        for id, elm in self.mesh.elements.items():
            el_type, tags, nodes = elm
            if len(tags) < 2:
                raise Exception("Less then 2 tags.")
            dim = self.el_type_to_dim[el_type]
            shape_id = tags[1]
            shape_info = self.shape_dict[ (dim, shape_id) ]
            if not shape_info.free:
                continue
            for i_node in nodes:
                x,y,z = self.mesh.nodes[i_node]
                top_ref_z = shape_info.top_iface.surface.depth
                top_z = shape_info.top_iface.surface.eval_z(x,y)
                bot_ref_z = shape_info.bot_iface.surface.depth
                bot_z = shape_info.bot_iface.surface.eval_z(x, y)
                assert bot_ref_z >= z and z >= top_ref_z, "{} >= {} >= {}".format(bot_ref_z, z, top_ref_z)
                if shape_info.top_iface.surface.id == shape_info.bot_iface.surface.id:
                    z_shift = top_z - z
                else:
                    t = (z - bot_ref_z) / (top_ref_z - bot_ref_z)
                    z_shift = (1 - t) * bot_z + t * top_z - z
                nodes_shift[i_node].append( z_shift )

        for id, shift_list in nodes_shift.items():
            assert len(shift_list) != 0, "Node: {}".format(id)
            mean_shift = sum(shift_list) / float(len(shift_list))
            assert sum([ math.fabs(x - mean_shift) for x in shift_list]) / float(len(shift_list)) < math.fabs(mean_shift)/100.0,\
                "{} List: {}".format(id, shift_list)
            self.mesh.nodes[id][2] += mean_shift

    def modify_mesh(self):
        self.tmp_msh_file = self.filename_base + ".tmp.msh"
        self.mesh = gmsh_io.GmshIO()
        with open(self.tmp_msh_file, "r") as f:
            self.mesh.read(f)

        # deform mesh, nontrivial evaluation of Z for the interface mesh
        #self.deform_mesh()


        new_elements = {}
        for id, elm in self.mesh.elements.items():
            el_type, tags, nodes = elm
            if len(tags) < 2:
                raise Exception("Less then 2 tags.")
            dim = self.el_type_to_dim[el_type]
            shape_id = tags[1]
            shape_info = self.gmsh_shape_dist[ (dim, shape_id)]

            if not shape_info.free:
                continue
            region = self.regions[shape_info.i_reg]
            if not region.is_active(dim):
                continue
            assert region.dim == dim
            physical_id = shape_info.i_reg + 10000
            if region.name in self.mesh.physical:
                assert self.mesh.physical[region.name][0] == physical_id
            else:
                self.mesh.physical[region.name] = (physical_id, dim)
            tags[0] = physical_id
            new_elements[id] = (el_type, tags, nodes)
        self.mesh.elements = new_elements
        self.msh_file = self.filename_base + ".msh"
        with open(self.msh_file, "w") as f:
            self.mesh.write_ascii(f)



    # def mesh_export(self, mesh, filename):
    #     """ export Netgen mesh to neutral format """
    #
    #     print("export mesh in neutral format to file = ", filename)
    #
    #     f = open(filename, 'w')
    #
    #     points = mesh.Points()
    #     print(len(points), file=f)
    #     for p in points:
    #         print(p.p[0], p.p[1], p.p[2], file=f)
    #
    #     volels = mesh.Elements3D();
    #     print(len(volels), file=f)
    #     for el in volels:
    #         print(el.index, end="   ", file=f)
    #         for p in el.points:
    #             print(p.nr, end=" ", file=f)
    #         print(file=f)

    # def mesh_netgen(self):
    #     """
    #     Use Netgen python interface to make a mesh.
    #     :return:
    #     """
    #
    #     geo = ngcsg.CSGeometry("shaft.geo")
    #
    #     param = ngmesh.MeshingParameters()
    #     param.maxh = 10
    #     print(param)
    #
    #     m1 = ngcsg.GenerateMesh(geo, param)
    #     m1.SecondOrder()
    #
    #     self.mesh_export(m1, "shaft.mesh")
    #
    #     ngcsg.Save(m1, "mesh.vol", geo)
    #     print("Finished")
    #
    # def netgen_to_gmsh(self):
    #     pass



def construct_derived_geometry(gs_obj):
    if issubclass(gs_obj.__class__, js.JsonData):
        geo_name = gs_obj.__class__.__name__
        class_obj = getattr(sys.modules[__name__], geo_name)
        geo_obj =  class_obj.__new__(class_obj)
        for key, item in  gs_obj.__dict__.items():
            item = construct_derived_geometry(item)
            geo_obj.__dict__[key] = item
    elif isinstance(gs_obj, (list, tuple)):
        new_item = [ construct_derived_geometry(i) for i in gs_obj ]
        geo_obj = gs_obj.__class__(new_item)
    elif isinstance(gs_obj, dict):
        for k, v in gs_obj.items():
            gs_obj[k] = construct_derived_geometry(v)
        geo_obj = gs_obj
    else:
        geo_obj = gs_obj
    return geo_obj


def make_geometry(**kwargs):
    layers_file = kwargs.get("layers_file", None)
    mesh_step = kwargs.get("mesh_step", 0.0)

    layers_file = layers_file
    filename_base = os.path.splitext(layers_file)[0]
    gs_lg = layers_io.read_geometry(layers_file)
    lg = construct_derived_geometry(gs_lg)
    lg.filename_base = filename_base

    lg.init()   # initialize the tree with ids and references where necessary

    lg.construct_brep_geometry()
    lg.make_gmsh_shape_dict()
    lg.distribute_mesh_step()

    #geom.mesh_netgen()
    #geom.netgen_to_gmsh()

    lg.call_gmsh(mesh_step)
    lg.modify_mesh()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('layers_file', help="Input Layers file (JSON).")
    parser.add_argument("--mesh-step", type=float, default=0.0, help="Maximal global mesh step.")
    args = parser.parse_args()

    make_geometry(layers_file=args.layers_file, mesh_step=args.mesh_step)
