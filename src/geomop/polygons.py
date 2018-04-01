import numpy as np
import numpy.linalg as la
import enum
import geomop.aabb_lookup as aabb_lookup
import geomop.decomp as decomp

from .decomp import PolygonChange

# TODO: rename point - > node
# TODO: careful unification of tolerance usage.
# TODO: Performance tests:
# - snap_point have potentialy very bad complexity O(Nlog(N)) with number of segments
# - add_line linear with number of segments
# - other operations are at most linear with number of segments per wire or point


in_vtx = left_side = 1
# vertex where edge comes in; side where next segment is connected through the in_vtx
out_vtx = right_side = 0
# vertex where edge comes out; side where next segment is connected through the out_vtx



class PolygonDecomposition:
    """
    Frontend to Decomposition.
    Provides high level operations.

    Methods that works with some tolerance:
    Segment:

      intersection - tolerance for snapping to the end points, fixed eps = 1e-10
                   - snapping only to one of intersectiong segments

      is_on_x_line - no tolerance, but not sure about numerical stability

    Wire:
        contains_point(self, xy):   called by Polygon.contains_point
            -> seg.is_on_x_line(xy)

        contains_wire(self, wire):
            - fixed tolerance eps=1e-10
            -> self.contains_point(inner_point)

    PD.snap_point, use slef. tolerance consistently

    """

    def __init__(self):
        """
        Constructor.
        """
        self.points_lookup = aabb_lookup.AABB_Lookup()
        self.segments_lookup = aabb_lookup.AABB_Lookup()
        self.decomp = decomp.Decomposition()
        self.tolerance = 0.01


    def __repr__(self):
        stream = "PolygonDecomposition\n"
        stream+=str(self.decomp)
        return stream

    # def __eq__(self, other):
    #     return len(self.points) == len(other.points) \
    #         and len(self.segments) == len(other.segments) \
    #         and len(self.polygons) == len(other.polygons)

    @property
    def points(self):
        return self.decomp.points

    @property
    def segments(self):
        return self.decomp.segments

    @property
    def polygons(self):
        return self.decomp.polygons

    @property
    def outer_polygon(self):
        return self.decomp.outer_polygon

    ##################################################################
    # Interface for LayerEditor. Should be changed.
    ##################################################################
    def add_free_point(self, point_id, xy, polygon_id):
        """
        LAYERS
        :param point_id: ID of point to add.
        :param xy: point: (X,Y)
        :param polygon_id: Hit in which polygon place the point.
        :return: Point instance
        """

        #print("add_free_point", point_id, xy, polygon_id)
        polygon = self.decomp.polygons[polygon_id]
        assert polygon.contains_point(xy), "Point {} not in polygon: {}.\n{}".format(xy, polygon, self)
        return self._add_point(xy, polygon, id = point_id)


    def remove_free_point(self, point_id):
        """
        LAYERS
        :param: point_id - ID of free point to remove
        :return: None
        """
        point = self.decomp.points[point_id]
        self._rm_point(point)
 
    def new_segment(self, a_pt, b_pt):
        """
        LAYERS
        Add segment between given existing points. Assumes that there is no intersection with other segment.
        Just return the segment if it exists.
        :param a_pt: Start point of the segment.
        :param b_pt: End point.
        :return: new segment
        """
        return self._add_segment(a_pt, b_pt)


    def delete_segment(self, segment):
        """
        LAYERS
        Remove specified segment.
        :param segment:
        :return: None
        """
        return self._rm_segment(segment)


    def check_displacment(self, points, displacement, margin):
        """
        LAYERS
        param: points: List of Points to move.
        param: displacement: Numpy array, 2D vector of displacement to add to the points.
        param: margin: float between (0, 1), displacement margin as a fraction of maximal displacement
        TODO: Check fails for internal wires and nonconvex poygons.
        :return: Shortened displacement to not cross any segment.
        """
        # Collect fixed sides of segments connecting fixed and moving point.
        segment_set = set()
        changed_polygons = set()
        for pt in points:
            for seg, side in pt.segments():
                changed_polygons.add(seg.wire[out_vtx].polygon)
                changed_polygons.add(seg.wire[in_vtx].polygon)
                opposite = (seg, 1-side)
                if opposite in segment_set:
                    segment_set.remove(opposite)
                else:
                    segment_set.add((seg, side))

        # collect segments fomring envelope(s) of the moving points
        envelope = set()
        for seg, side in segment_set:
            for e_seg_side in seg.wire[side].segments(start = seg.next[side]):
                if e_seg_side in segment_set:
                    break
                e_seg, e_side = e_seg_side
                envelope.add(e_seg)

        new_displ = np.array(displacement)
        for seg in envelope:
            for pt in points:
                (t0, t1) = self.seg_intersection(seg, pt.xy, pt.xy + new_displ)
                # TODO: Treat case of vector and segment in line.
                # TODO: Check bound checks in intersection.
                if t0 is not None:
                    new_displ *= (1.0 - margin) * t1
        self.decomp.last_polygon_change = (decomp.PolygonChange.shape, changed_polygons, None)
        return new_displ

    def move_points(self, points, displacement):
        """
        Move given points by given displacement vector. Assumes no intersections. But possible
        segment splitting (add_point is called).
        param: points: List of Points to move.
        param: displacement: Numpy array, 2D vector of displacement to add to the points.
        :return: None
        """
        for pt in points:
            pt.xy += displacement


    def get_last_polygon_changes(self):
        """
        LAYERS
        Return information about polygon changes during last new_segment or delete_segment operations.
        :return: ( PolygonChange, p0.id, p1.id)
        cases:
        (PolygonChange.none, None, None) - no change in any polygon, already existed segment
        (PolygonChange.shape, list_poly_id, None) - list of polygons that have changed shape, e.g. add/remove dendrite
        (PolygonChange.add, orig_poly_id, new_poly_id) - new polygon inside other polygon
        (PolygonChange.remove, orig_poly_id, new_poly_id) - deleted polygon inside other polygon
        (PolygonChange.split, orig_poly_id, new_poly_id) - split new_poly from orig_poly
        (PolygonChange.join, orig_poly_id, del_poly_id) - join both polygons into orig_poly

        After init of PolygonDecomposition this method returns:
        (PolygonChange.add, outer_polygon_id, outer_polygon_id)
        """
        type, p0, p1 = self.decomp.last_polygon_change
        if type == decomp.PolygonChange.shape:
            poly_ids = [poly.id for poly in p0]
            return (type, poly_ids, None)
        id0 = None if p0 is None else p0.id
        id1 = None if p1 is None else p1.id
        return (type, id0, id1)

    def get_childs(self, polygon_id):
        """
        Return list of child ploygons (including itself).
        :param polygon_id:
        :return: List of polygon IDs.
        """
        # TODO: remove after problem with infinite recursion is solved
        for w in self.decomp.wires.values():
            w._get_child_passed = False

        #print(self)
        child_poly_id_set = set()
        root_poly = self.decomp.polygons[polygon_id]
        for poly in  root_poly.child_polygons():
            child_poly_id_set.add(poly.id)
        return child_poly_id_set

    ########################################
    # Other public methods.

    def set_tolerance(self, tolerance):
        """
        Set tolerance for snapping to existing points and lines.
        Should be given by actual zoom level.
        :param tolerance: float, a distance used to snap points to existing objects.
        :return: None
        """
        self.tolerance = tolerance






    ###################################################################
    # Macro operations that change state of the decomposition.
    def add_point(self, point):
        """
        Try to add a new point, snap to lines and existing points.
        :param point: numpy array with XY coordinates
        :return: Point instance.

        This operation translates to atomic operations: add_free_point and split_line_by_point.
        TODO: make consisten system to check ide effects of decomp operations.
        This is partly done with get_last_polygon_changes but we need similar for segment in this method.
        This is necessary in intersections.
        """
        point = np.array(point, dtype=float)
        dim, obj, t = self._snap_point(point)
        if dim == 0:
            # nothing to add
            return obj
        elif dim == 1:
            seg = obj
            mid_pt, new_seg = self._point_on_segment(seg, t)
            return mid_pt
        else:
            poly = obj
            return self._add_point(point, poly)

    def pt_dist(self, pt, point):
        return la.norm(pt.xy - point)

    def _snap_point(self, point):
        """
        Find object (point, segment, polygon) within tolerance from given point.
        :param point: numpy array X, Y
        :return: (dim, obj, param) Where dim is object dimension (0, 1, 2), obj is the object (Point, Segment, Polygon).
        'param' is:
          Point: None
          Segment: parameter 't' of snapped point on the segment
          Polygon: None
        """
        point = np.array(point, dtype=float)

        # First snap to points
        candidates = self.points_lookup.closest_candidates(point)
        #candidates = self.points.keys()
        for pt_id in candidates:
            pt = self.points[pt_id]
            if self.pt_dist(pt, point) <  self.tolerance:
                return (0, pt, None)

        # Snap to segments, keep the closest to get polygon.
        closest_seg = (np.inf, None, None)
        candidates = self.segments_lookup.closest_candidates(point)
        #candidates = self.segments.keys()
        for seg_id in candidates:
            seg = self.segments[seg_id]
            t = self.seg_project_point(seg, point)
            dist = la.norm(point - seg.parametric(t))
            if dist < self.tolerance:
                return (1, seg, t)
            elif dist < closest_seg[0]:
                closest_seg = (dist, seg, t)
        assert closest_seg[0] < np.inf or len(self.segments) == 0

        # cs = closest_seg
        #
        # closest_seg = (np.inf, None, None)
        # candidates = self.segments.keys()
        # for seg_id in candidates:
        #     seg = self.segments[seg_id]
        #     t = self.seg_project_point(seg, point)
        #     dist = la.norm(point - seg.parametric(t))
        #     if dist < self.tolerance:
        #         return (1, seg, t)
        #     elif dist < closest_seg[0]:
        #         closest_seg = (dist, seg, t)
        #
        # if cs != closest_seg:
        #     self.segments_lookup.closest_candidates(point)
        #     assert False

        # Snap to polygon,
        # have to deal with nonconvex case
        poly = None
        dist, seg, t = closest_seg
        if seg is None:
            return (2, self.decomp.outer_polygon, None)
        if t == 0.0:
            pt = seg.vtxs[out_vtx]
        elif t == 1.0:
            pt = seg.vtxs[in_vtx]
        else:
            # convex case
            tangent = seg.vector
            normal = np.array([tangent[1], -tangent[0]])
            point_n = (point - seg.vtxs[out_vtx].xy).dot(normal)
            assert point_n != 0.0
            side = right_side if point_n > 0 else left_side
            poly = seg.wire[side].polygon

        if poly is None:
            # non-convex case
            prev, next, wire = pt.insert_vector(point - pt.xy)
            poly = wire.polygon
        if not poly.contains_point(point):
            assert False
        return (2, poly, None)


    def add_line(self, a, b):
        """
        Try to add new line from point A to point B. Check intersection with any other line and
        call add_point for endpoints, call split_segment for intersections, then call operation new_segment for individual
        segments.
        :param a: numpy array X, Y
        :param b: numpy array X, Y
        :return: List of subdivided segments. Split segments are not reported.
        """
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        a_point = self.add_point(a)
        b_point = self.add_point(b)
        if a_point == b_point:
            return a_point
        return self.add_line_for_points(a_point, b_point)


    def add_line_for_points(self, a_pt, b_pt):
        """
        Same as add_line, but for known end points.
        :param a_pt:
        :param b_pt:
        :return:
        """
        line_div = self._add_line_seg_intersections(a_pt, b_pt)
        return [seg    for seg, change, side in self._add_line_new_segments(a_pt, b_pt, line_div)]


    def _point_on_segment(self, seg, t):
        if t < self.tolerance:
            mid_pt = seg.vtxs[out_vtx]
            new_seg = seg
        elif t > 1.0 - self.tolerance:
            mid_pt = seg.vtxs[in_vtx]
            new_seg = seg
        else:
            xy_point = seg.parametric(t)
            mid_pt = self._add_point(xy_point, self.decomp.outer_polygon)
            new_seg = self.decomp.split_segment(seg, mid_pt)
            self.segments_lookup.add_object(new_seg.id,
                aabb_lookup.make_aabb([new_seg.vtxs[0].xy, new_seg.vtxs[1].xy], margin=self.tolerance))

        return (mid_pt, new_seg)


    def _add_line_seg_intersections(self, a_pt, b_pt):
        """
        Generator for intersections of a new line with existing segments.
        Every segment is split and intersection point is yield.
        :param a_pt, b_pt: End points of the new line.
        :returns a dictionary t -> (isec_pt, seg0, seg1),
            - parameter of the intersection on the new line
            - the Point object of the intersection point.
            - old and new subsegments of the segment split
            - new seg == old seg if point is snapped to the vertex
        TODO: Points can collide even for different t,
        rather return just mid points and new segments and use point ID as key in dict.
        TODO: intersectiong two segments with very small angle we may add two
        points that are closer then tolerance. This may produce an error later on.
        However healing this is nontrivial, since we have to merge two segments.
        """
        line_division = {}
        box = aabb_lookup.make_aabb([a_pt.xy, b_pt.xy], margin=self.tolerance)
        candidates = self.segments_lookup.intersect_candidates(box)
        #candidates = list(self.segments.keys()) # need copy since we change self.segments
        for seg_id in candidates:
            seg = self.segments[seg_id]
            (t0, t1) = self.seg_intersection(seg, a_pt.xy, b_pt.xy)
            if t1 is not None:
                mid_pt, new_seg = self._point_on_segment(seg, t0)
                line_division[t1] = (mid_pt, seg, new_seg)
        return line_division

    def _add_line_new_segments(self, a_pt, b_pt, line_div):
        """
        Generator for added new segments of the new line.
        """
        start_pt = a_pt
        for t1, (mid_pt, seg0, seg1) in sorted(line_div.items()):
            if start_pt == mid_pt:
                continue
            new_seg = self._add_segment(start_pt, mid_pt)
            yield (new_seg, self.decomp.last_polygon_change, new_seg.vtxs[out_vtx] == start_pt)
            start_pt = mid_pt

        if start_pt != b_pt:
            new_seg = self._add_segment(start_pt, b_pt)
            yield (new_seg, self.decomp.last_polygon_change, new_seg.vtxs[out_vtx] == start_pt)



    def delete_point(self, point):
        """
        Delete given point with all connected segments.
        :param point:
        :return:
        """
        segs_to_del = [ seg for seg, side in point.segments()]
        for seg in segs_to_del:
            self._rm_segment(seg)
        self._rm_point(point)

    #################################
    # Internal interface for adding and removing points and segments to Decomposition.
    # We need it to keep lookup up to date.

    def _add_point(self, pt, poly, id=None):
        pt = self.decomp.add_free_point(pt, poly, id)
        self.points_lookup.add_object(pt.id, aabb_lookup.make_aabb([pt.xy], margin=self.tolerance))
        return pt

    def _add_segment(self, a_pt, b_pt):
        seg = self.decomp.new_segment(a_pt, b_pt)
        self.segments_lookup.add_object(seg.id, aabb_lookup.make_aabb([a_pt.xy, b_pt.xy], margin=self.tolerance))
        return seg

    def _rm_point(self, pt):
        self.points_lookup.rm_object(pt.id)
        self.decomp.remove_free_point(pt)

    def _rm_segment(self, seg):
        self.segments_lookup.rm_object(seg.id)
        self.decomp.delete_segment(seg)

    #################################
    # Segment calculations.

    @staticmethod
    def seg_project_point(seg, pt):
        """
        Return parameter t of the projection to the segment.
        :param pt: numpy [X,Y]
        :return: t
        """
        Dxy = seg.vector
        AX = pt - seg.vtxs[out_vtx].xy
        dxy2 = Dxy.dot(Dxy)
        assert dxy2 != 0.0
        t = AX.dot(Dxy)/dxy2
        return min(max(t, 0.0), 1.0)


    @staticmethod
    def seg_intersection(seg, a, b):
        """
        Find intersection of 'self' and (a,b) edges.
        :param a: start vtx of edge1
        :param b: end vtx of edge1
        :return: (t0, t1) Parameters of the intersection for 'self' and other edge.
        """

        mat = np.array([ seg.vector, a - b])
        rhs = a - seg.vtxs[0].xy
        try:
            t0, t1 = la.solve(mat.T, rhs)
        except la.LinAlgError:
            return (None, None)
            # TODO: possibly treat case of overlapping segments
        eps = 1e-10
        if 0 <= t0 <= 1  and 0 + eps <= t1 <= 1 - eps:
            return (t0, t1)
        else:
            return (None, None)


    ################################
    # Serialization methods - should move into polygon_io

    def make_segment(self, node_ids):
        """
        Used in deep_copy and deserialize.

        :param node_ids:
        :return:
        """
        v_out_id, v_in_id = node_ids
        vtxs = (self.decomp.points[v_out_id], self.decomp.points[v_in_id])
        seg = self.decomp._make_segment(vtxs)
        self.segments_lookup.add_object(seg.id, aabb_lookup.make_aabb([vtxs[0].xy, vtxs[1].xy], margin=self.tolerance))
        return seg

    def make_wire_from_segments(self, seg_ids, polygon):
        """
        Used in  deserialize.

        Set half segments of the wire, and the wire itself.
        :param seg_ids: Segment ids, at least 2 and listed in the orientation matching the wire (cc wise)
        :param polygon: Polygon the wire is part of.
        :return: None
        """
        assert len(seg_ids) >= 2, "segments: {}".format(seg_ids)
        wire = decomp.Wire()
        self.decomp.wires.append(wire)

        # detect orientation of the first segment
        last_seg = self.decomp.segments[seg_ids[0]]
        seg1 = self.decomp.segments[seg_ids[1]]
        last_side = out_vtx
        vtx0_side = seg1.point_side(last_seg.vtxs[last_side])
        if vtx0_side is None:
            last_side = in_vtx
            assert seg1.point_side(last_seg.vtxs[last_side]) is not None, "Can not connect segments: {} {}".format(last_seg, seg1)
        start_seg_side = (last_seg, last_side)

        # set segment sides along the wire
        for id in seg_ids[1:]:
            seg = self.decomp.segments[id]
            side = seg.point_side(last_seg.vtxs[last_side])
            assert side is not None, "Can not connect segments: {} {}".format(last_seg, seg)
            seg_side = (seg, 1 - side)
            last_seg.next[last_side] = seg_side
            last_seg.wire[last_side] = wire
            last_seg, last_side = seg_side
        last_seg.next[last_side] = start_seg_side
        last_seg.wire[last_side] = wire
        wire.segment = start_seg_side
        wire.polygon = polygon
        return wire

    def make_polygon(self, outer_segments, holes, free_points):
        """
        Used in  deep_copy and deserialize.

        :param outer_segments:
        :param holes:
        :param free_points:
        :return:
        """
        if len(outer_segments) != 0:
            p = self.decomp.polygons.append(decomp.Polygon(None))
            p.outer_wire = self.make_wire_from_segments(outer_segments, p)
        else:
            p = self.decomp.outer_polygon

        for hole in holes:
            wire = self.make_wire_from_segments(hole, p)
            wire.set_parent(p.outer_wire)
        for free_pt_id in free_points:
            pt = self.decomp.points[free_pt_id]
            pt.set_polygon(p)
        return p



    def set_wire_parents(self):
        """
        Used in  deep_copy and deserialize.

        Set parent wire links from holes.
        """
        for poly in self.decomp.polygons.values():
            for hole in poly.outer_wire.childs:
                child_queue = hole.neighbors()
                # BFS for inner wires of the hole
                while child_queue:
                    inner_wire = child_queue.pop(0)
                    if inner_wire.parent == inner_wire:
                        inner_wire.set_parent(hole)
                        for wire in inner_wire.neighbors():
                            child_queue.append(wire)

