import enum
import geomop.idmap as idmap
from .point import Point
from .segment import Segment, right_side, left_side, out_vtx, in_vtx
from .polygon import Polygon, Wire
import numpy.linalg as la


class PolygonChange(enum.Enum):
    none = 0
    shape = 1
    add = 2
    remove = 3
    split = 4
    join = 5



class Decomposition:
    """
    Decomposition of a plane into (non-convex) polygonal subsets (not necessarily domains).
    - should contain only topological operations (with exception of checking point in wire, which
      has to be made as robust as possible)
    - all snapping of raw cooridinates should be done in frontend class PolygonDecomposition
    - all operations have its inverse.
    - all elementary operations are marked into history, history should be general enough to
      contain messages from different classes and groups of operations. Operations on this class
      should be atomic.

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
        PUBLIC: outer_polygon_id
        """
        self.points = idmap.IdMap()
        # Points dictionary ID -> Point
        self.segments = idmap.IdMap()
        # Segmants dictionary ID - > Segmant
        self.pt_to_seg = {}
        # dict (a.id, b.id) -> segment
        self.wires = idmap.IdMap()
        # Closed loops possibly degenerated) of segment sides. Single wire can be tracked through segment.next links.
        self.polygons = idmap.IdMap()
        # Polygon dictionary ID -> Polygon
        self.shapes = [self.points, self.segments, self.polygons]
        # Common access to shapes of various dim.

        # Most outer wire of whole decomposition
        outer_wire = self.wires.append(Wire())
        outer_wire.parent = None

        # Outer polygon - extending to infinity
        self.outer_polygon = Polygon(outer_wire)
        self.polygons.append(self.outer_polygon)
        outer_wire.polygon = self.outer_polygon

        self.last_polygon_change = (PolygonChange.add, self.outer_polygon, self.outer_polygon)
        # Last polygon operation.
        # TODO: make full undo/redo history.
        #
        #self.tolerance = 0.01

    def __repr__(self):
        stream = ""
        for label, objs in [("Polygons:", self.polygons), ("Wires:", self.wires), ("Segments:", self.segments)]:
            stream += label + "\n"
            for obj in objs.values():
                stream += str(obj) + "\n"
        return stream

    def __eq__(self, other):
        return len(self.points) == len(other.points) \
               and len(self.segments) == len(other.segments) \
               and len(self.polygons) == len(other.polygons)


    def check_consistency(self):
        # print(self)
        for p in self.polygons.values():
            # print(p)
            # print(p.free_points)
            assert p.outer_wire.id in self.wires
            assert p.outer_wire.polygon == p
            for pt in p.free_points:
                # print(pt)
                # print(pt.polygon)
                assert pt.poly.id in self.polygons
                assert pt.poly == p
                assert pt.segment == (None, None)

        for w in self.wires.values():
            for child in w.childs:
                assert child.id in self.wires
                child.parent == w
            assert w.polygon.id in self.polygons
            assert w == w.polygon.outer_wire or w in w.polygon.outer_wire.childs
            if w.is_root():
                assert w == self.outer_polygon.outer_wire
            else:
                seg, side = w.segment
                assert seg.id in self.segments
                assert seg.wire[side] == w
                assert w in w.parent.childs

        for sg in self.segments.values():
            assert seg.point_ids() in self.pt_to_seg
            for side in [right_side, left_side]:
                assert sg.vtxs[side].id in self.points
                assert sg.wire[side].id in self.wires
                assert sg.next[side][0].id in self.segments

                assert sg in [seg for seg, side in sg.vtxs[side].segments()]
                w_seg, w_side = sg.wire[side].segment
                assert sg.wire[side] == w_seg.wire[w_side]
                n_seg, n_side = sg.next[side]
                assert sg.wire[side] == n_seg.wire[n_side]

        for points, seg in self.pt_to_seg.items():
            assert seg.id in self.segments
            x_seg = self.segments[seg.id]
            assert seg.point_ids() == points

        for pt in self.points.values():
            if pt.is_free():
                assert pt.poly.id in self.polygons
                assert pt in pt.poly.free_points
            else:
                seg, side = pt.segment
                assert seg.id in self.segments
                assert seg.vtxs[side] == pt
        return True



    ###############################
    # Public invertible operations.

    def add_free_point(self, point, poly, id=None):
        """
        :param point: XY array
        :return: Point instance
        """

        pt = Point(point, poly)
        self.points.append(pt, id)
        poly.free_points.add(pt)
        return pt

    def remove_free_point(self, point):
        assert point.poly is not None
        assert point.segment[0] is None
        point.poly.free_points.remove(point)
        del self.points[point.id]


    def new_segment(self, a_pt, b_pt):
        """
        LAYERS
        Add segment between given existing points. Assumes that there is no intersection with other segment.
        Just return the segment if it already exists (possibly opposite orientation).
        :param a_pt: Start point of the segment.
        :param b_pt: End point.
        :return: new segment
        """
        assert a_pt != b_pt
        assert la.norm(a_pt.xy - b_pt.xy) >1e-10
        self.last_polygon_change = (PolygonChange.none, None, None)
        segment = self.pt_to_seg.get((a_pt.id, b_pt.id), None)
        if segment is not None:
            return segment
        segment = self.pt_to_seg.get((b_pt.id, a_pt.id), None)
        if segment is not None:
            return segment

        if a_pt.is_free() and b_pt.is_free():
            assert a_pt.poly == b_pt.poly
            return self._new_wire(a_pt.poly, a_pt, b_pt)

        vec = b_pt.xy - a_pt.xy
        a_insert = a_pt.insert_vector(vec)
        b_insert = b_pt.insert_vector(-vec)

        if a_pt.is_free():
            assert b_insert is not None
            return self._wire_add_dendrite((a_pt, b_pt), b_insert, in_vtx)
        if b_pt.is_free():
            assert a_insert is not None
            return self._wire_add_dendrite((a_pt, b_pt), a_insert, out_vtx)

        assert a_insert is not None
        assert b_insert is not None
        a_prev, a_next, a_wire = a_insert
        b_prev, b_next, b_wire = b_insert

        if a_wire != b_wire:
            return self._join_wires(a_pt, b_pt, a_insert, b_insert)
        else:
            return self._split_poly(a_pt, b_pt, a_insert, b_insert)

    def delete_segment(self, segment):
        """
        LAYERS
        Remove specified segment.
        :param segment:
        :return: None
        """
        self.last_polygon_change = (PolygonChange.none, None, None)
        left_self_ref = segment.next[left_side] == (segment, right_side)
        right_self_ref = segment.next[right_side] == (segment, left_side)
        # Lonely segment, both endpoints are free.
        if left_self_ref and right_self_ref:
            return self._rm_wire(segment)
        # At least one free endpoint.
        if left_self_ref:
            return self._wire_rm_dendrite(segment, in_vtx)
        if right_self_ref:
            return self._wire_rm_dendrite(segment, out_vtx)

        # Both endpoints connected.
        if segment.is_dendrite():
            # Same wire from both sides. Dendrite.
            self._split_wire(segment)
        else:
            # Different wires.
            self._join_poly(segment)

    def split_segment(self, seg, mid_pt):
        """
        Split a segment into two segments.
        seg = (A, B) split to       A -> seg -> mid_pt -> new_seg -> B
        Mid point is not checked that is actually lies on the segment.
        :param seg: A segment to split
        :param Mid point
        :return: new_segment
        """

        # xy_point = seg.parametric(t_point)
        # mid_pt = Point(xy_point, None)
        # self.points.append(mid_pt)

        b_seg_insert = seg.vtx_insert_info(in_vtx)
        # TODO: remove this hard wired insert info setup
        # modify point insert method to return full insert info
        # it should have treatment of the single segment pint , i.e. tip
        seg_tip_insert = ((seg, left_side), (seg, right_side), seg.wire[right_side])
        seg.disconnect_vtx(in_vtx)
        del self.pt_to_seg[seg.point_ids()]
        self.pt_to_seg[(seg.vtxs[0].id, mid_pt.id)] = seg

        new_seg = self._make_segment((mid_pt, seg.vtxs[in_vtx]))
        seg.vtxs[in_vtx] = mid_pt
        seg._vector = seg.vtxs[in_vtx].xy - seg.vtxs[out_vtx].xy
        new_seg.connect_vtx(out_vtx, seg_tip_insert)
        if b_seg_insert is None:
            assert seg.is_dendrite()
            new_seg.connect_free_vtx(in_vtx, seg.wire[left_side])
        else:
            new_seg.connect_vtx(in_vtx, b_seg_insert)

        return new_seg

    def join_segments(self, mid_point, seg0, seg1):
        """
        Join splited segment, return free mid point.
        Has to be destroyed explicitly.
        TODO: replace by del_segment and 2x new_segment
        """
        if seg0.vtxs[in_vtx] == mid_point:
            seg0_out_vtx, seg0_in_vtx = out_vtx, in_vtx
        else:
            seg0_out_vtx, seg0_in_vtx = in_vtx, out_vtx

        if seg1.vtxs[out_vtx] == mid_point:
            seg1_out_vtx, seg1_in_vtx = out_vtx, in_vtx
        else:
            seg1_out_vtx, seg1_in_vtx = in_vtx, out_vtx

        # Assert that no other segments are joined to the mid_point
        assert seg0.next[seg0_in_vtx] == (seg1, seg1_in_vtx)
        assert seg1.next[seg1_out_vtx] == (seg0, seg0_out_vtx)

        b_seg1_insert = seg1.vtx_insert_info(seg1_in_vtx)
        seg1.disconnect_vtx(seg1_in_vtx)
        seg1.disconnect_vtx(seg1_out_vtx)
        seg0.disconnect_vtx(seg0_in_vtx)
        seg0.vtxs[seg0_in_vtx] = seg1.vtxs[seg1_in_vtx]
        if b_seg1_insert is None:
            assert seg0.is_dendrite()
            seg0.connect_free_vtx(seg0_in_vtx, seg0.wire[out_vtx])
        else:
            seg0.connect_vtx(seg0_in_vtx, b_seg1_insert)

        # fix possible wire references
        for side in [left_side, right_side]:
            wire = seg1.wire[side]
            if wire.segment == (seg1, side):
                wire.segment = (seg0, side)

        self._destroy_segment(seg1)
        return mid_point




    #######################################3
    # Internal invertible operations.





    def _new_wire(self, polygon, a_pt, b_pt):
        """
        New wire containing just single segment.
        return the new_segment
        """

        wire = self.wires.append(Wire())
        wire.polygon = polygon
        wire.set_parent(polygon.outer_wire)
        seg = self._make_segment((a_pt, b_pt))
        seg.connect_free_vtx(out_vtx, wire)
        seg.connect_free_vtx(in_vtx, wire)
        wire.segment = (seg, right_side)
        return seg

    def _rm_wire(self, segment):
        """
        Remove the last segment of a wire.
        :return: None
        """
        assert segment.next[left_side] == (segment, right_side) and segment.next[right_side] == (segment, left_side)
        assert segment.is_dendrite()
        wire = segment.wire[left_side]
        polygon = wire.polygon
        polygon.outer_wire.childs.remove(wire)
        del self.wires[wire.id]
        self._destroy_segment(segment)




    def _wire_add_dendrite(self, points, r_insert, root_idx):
        """
        Add new dendrite tip segment.
        points: (out_pt, in_pt)
        r_insert: insert information for root point
        root_idx: index (0/1) of the root, i.e. non-free point.
        """
        free_pt = points[1 - root_idx]
        polygon = free_pt.poly
        r_prev, r_next, wire = r_insert
        assert wire.polygon == free_pt.poly, "point poly: {} insert: {}".format(free_pt.poly, r_insert)

        seg = self._make_segment(points)
        seg.connect_vtx(root_idx, r_insert)
        seg.connect_free_vtx(1 - root_idx, wire)
        self.last_polygon_change = (PolygonChange.shape, [polygon], None)
        return seg

    def _wire_rm_dendrite(self, segment, tip_vtx):
        """
        Remove dendrite tip segment.
        """

        root_vtx = 1 - tip_vtx
        assert segment.is_dendrite()
        polygon = segment.wire[out_vtx].polygon
        segment.disconnect_wires()
        segment.disconnect_vtx(root_vtx)

        self._destroy_segment(segment)
        self.last_polygon_change = (PolygonChange.shape, [polygon], None)




    def _join_wires(self, a_pt, b_pt, a_insert, b_insert):
        """
        Join two wires of the same polygon by new segment.
        """
        a_prev, a_next, a_wire = a_insert
        b_prev, b_next, b_wire = b_insert
        assert a_wire != b_wire
        assert a_wire.polygon == b_wire.polygon
        polygon = a_wire.polygon
        self.last_polygon_change = (PolygonChange.shape, [polygon], None)

        # set next links
        new_seg = self._make_segment((a_pt, b_pt))
        new_seg.connect_vtx(out_vtx, a_insert)
        new_seg.connect_vtx(in_vtx, b_insert)

        ############################
        keep_wire_side = None
        if polygon.outer_wire == a_wire:
            keep_wire_side = out_vtx  # a_wire
        elif polygon.outer_wire == b_wire:
            keep_wire_side = in_vtx  # b_wire

        if keep_wire_side is None:
            # connect two holes
            keep_wire_side = in_vtx
            keep_wire = new_seg.wire[keep_wire_side]
            rm_wire = new_seg.wire[1 - keep_wire_side]
            parent_wire = keep_wire
        else:
            keep_wire = new_seg.wire[keep_wire_side]
            rm_wire = new_seg.wire[1 - keep_wire_side]
            parent_wire = keep_wire.parent  # parent wire to set for childs of rm_wire
            polygon.outer_wire = keep_wire

        # update segment links to rm_wire
        for seg, side in rm_wire.segments(start=(new_seg, 1 - keep_wire_side), end=(new_seg, keep_wire_side)):
            assert seg.wire[side] == rm_wire, "wire: {} bwire: {} awire{}".format(seg.wire[side], b_wire, a_wire)
            seg.wire[side] = keep_wire
        new_seg.wire[out_vtx] = keep_wire

        # update child links to rm_wire
        for child in list(rm_wire.childs):
            child.set_parent(parent_wire)

        # update parent link to rm_wire
        rm_wire.parent.childs.remove(rm_wire)
        #####################
        del self.wires[rm_wire.id]

        return new_seg

    def _split_wire(self, segment):
        """
        Remove segment that connects two wires.
        """
        """
         Remove segment that connects two wires.
         """
        assert segment.is_dendrite()
        a_wire = segment.wire[left_side]
        polygon = a_wire.polygon
        b_wire = self.wires.append(Wire())

        # set new wire to segments (b_wire is on the segment side of the vtx[1])
        b_vtx_next_side = in_vtx
        b_vtx_prev_side = 1 - b_vtx_next_side
        b_next_seg = segment.next[b_vtx_next_side]
        for seg, side in a_wire.segments(start=b_next_seg, end=(segment, b_vtx_prev_side)):
            assert seg.wire[side] == a_wire
            seg.wire[side] = b_wire

        segment.disconnect_wires()
        segment.disconnect_vtx(out_vtx)
        segment.disconnect_vtx(in_vtx)

        # setup new b_wire
        b_wire.segment = b_next_seg
        b_wire.polygon = a_wire.polygon
        if polygon.outer_wire == a_wire:
            # one wire inside other
            outer_wire, inner_wire = b_wire, a_wire
            if a_wire.contains_wire(b_wire):
                outer_wire, inner_wire = a_wire, b_wire
            polygon.outer_wire = outer_wire
            outer_wire.set_parent(a_wire.parent)  # outer keep parent of original wire
            inner_wire.set_parent(outer_wire)
            self._update_wire_parents(a_wire.parent, a_wire.parent, inner_wire)

        else:
            # both wires are holes
            b_wire.set_parent(a_wire.parent)
            self._update_wire_parents(a_wire, a_wire, b_wire)

        # remove segment
        self.last_polygon_change = (PolygonChange.shape, [polygon], None)
        self._destroy_segment(segment)

    def _update_wire_parents(self, orig_wire, outer_wire, inner_wire):
        # Auxiliary method of _split_wires.
        # update all wires having orig wire as parent
        # TODO: use wire childs
        for wire in self.wires.values():
            if wire.parent == orig_wire:
                if inner_wire.contains_wire(wire):
                    wire.set_parent(inner_wire)
                else:
                    wire.set_parent(outer_wire)




    def _split_poly(self, a_pt, b_pt, a_insert, b_insert):
        """
        Split polygon by new segment.
        """
        a_prev, a_next, a_wire = a_insert
        b_prev, b_next, b_wire = b_insert
        assert a_wire == b_wire
        orig_wire = a_wire

        right_wire = a_wire
        left_wire = self.wires.append(Wire())

        # set next links
        new_seg = self._make_segment((a_pt, b_pt))
        new_seg.connect_vtx(out_vtx, a_insert)
        new_seg.connect_vtx(in_vtx, (b_prev, b_next, left_wire))

        # set right_wire links
        for seg, side in orig_wire.segments(start=new_seg.next[left_side], end=(new_seg, left_side)):
            assert seg.wire[side] == orig_wire
            seg.wire[side] = left_wire
        left_wire.segment = (new_seg, left_side)
        right_wire.segment = (new_seg, right_side)

        # update polygons
        orig_poly = right_poly = orig_wire.polygon
        new_poly = Polygon(left_wire)
        self.polygons.append(new_poly)
        left_wire.polygon = new_poly

        if orig_wire.polygon.outer_wire == orig_wire:
            # two disjoint polygons
            new_poly.outer_wire = left_wire
            left_wire.set_parent(orig_wire.parent)
            self.last_polygon_change = (PolygonChange.split, orig_poly, new_poly)
        else:
            assert orig_wire.parent == orig_poly.outer_wire
            # two embedded wires/polygons
            if right_wire.contains_wire(left_wire):
                inner_wire, outer_wire = left_wire, right_wire
            else:
                inner_wire, outer_wire = right_wire, left_wire

            # fix childs of orig_wire
            for child in list(orig_wire.childs):
                child.set_parent(outer_wire)

            outer_wire.polygon = orig_poly
            inner_wire.polygon = new_poly
            new_poly.outer_wire = inner_wire
            outer_wire.set_parent(orig_wire.parent)
            inner_wire.set_parent(outer_wire)
            self.last_polygon_change = (PolygonChange.add, orig_poly, new_poly)

        # split free points
        for pt in list(orig_poly.free_points):
            if new_poly.outer_wire.contains_point(pt.xy):
                pt.set_polygon(new_poly)

        # split holes
        for hole_wire in list(orig_poly.outer_wire.childs):
            if new_poly.outer_wire.contains_wire(hole_wire):
                hole_wire.set_parent(new_poly.outer_wire)
                hole_wire.polygon = new_poly
        return new_seg

    def _join_poly(self, segment):
        """
        Join polygons by removing a segment.
        """

        left_wire = segment.wire[left_side]
        right_wire = segment.wire[right_side]

        if left_wire.parent == right_wire.parent:
            assert left_wire == left_wire.polygon.outer_wire
            assert right_wire == right_wire.polygon.outer_wire
            orig_polygon = right_wire.polygon
            new_polygon = left_wire.polygon
            self.last_polygon_change = (PolygonChange.join, orig_polygon, new_polygon)
            keep_wire = right_wire
        else:
            if left_wire.parent == right_wire:
                # right is outer
                orig_polygon = right_wire.polygon
                new_polygon = left_wire.polygon
                keep_wire = right_wire
            else:
                assert right_wire.parent == left_wire
                # left is outer
                orig_polygon = left_wire.polygon
                new_polygon = right_wire.polygon
                keep_wire = left_wire
            self.last_polygon_change = (PolygonChange.remove, orig_polygon, new_polygon)

        rm_wire = new_polygon.outer_wire

        # Join holes and free points
        for child in list(rm_wire.childs):
            child.set_parent(keep_wire)

        for pt in list(new_polygon.free_points):
            pt.set_polygon(orig_polygon)

        # set parent for keeped wire
        # right_wire.set_parent(orig_polygon.outer_wire)

        rm_wire.set_parent(rm_wire)  # disconnect

        # fix wire links for
        for seg, side in rm_wire.segments():
            assert seg.wire[side] == rm_wire
            seg.wire[side] = keep_wire

        segment.disconnect_wires()
        segment.disconnect_vtx(out_vtx)
        segment.disconnect_vtx(in_vtx)

        self._destroy_segment(segment)
        del self.wires[rm_wire.id]
        del self.polygons[new_polygon.id]


    ###################################
    # Helper change operations.
    def _make_segment(self, points):
        assert points[0] != points[1]

        seg = Segment(points)
        self.segments.append(seg)
        for vtx in [out_vtx, in_vtx]:
            seg.vtxs[vtx].join_segment(seg, vtx)
        self.pt_to_seg[seg.point_ids()] = seg
        return seg

    def _destroy_segment(self, seg):
        seg.vtxs[out_vtx].rm_segment(seg, out_vtx)
        seg.vtxs[in_vtx].rm_segment(seg, in_vtx)
        a, b = seg.point_ids()
        self.pt_to_seg.pop((a, b), None)
        self.pt_to_seg.pop((b, a), None)
        del self.segments[seg.id]





