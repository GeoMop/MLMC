import geomop.idmap as idmap
from .segment import right_side, left_side, out_vtx, in_vtx
import numpy as np


class Point(idmap.IdObject):

    def __init__(self, point, poly):
        self.xy = np.array(point, dtype=float)
        self.poly = poly
        # Containing polygon for free-nodes. None for others.
        self.segment = (None, None)
        # (seg, vtx_side) One of segments joined to the Point and local idx of the segment (out_vtx, in_vtx).

    def __repr__(self):
        return "Pt({}) {}".format(self.id, self.xy)

    def __hash__(self):
        return self.id

    def is_free(self):
        """
        Indicator of a free point, not connected to eny segment.
        :return:
        """
        return self.segment[0] is None

    def segments(self, start=(None, None)):
        """
        Generator for segments joined to the point. Segments are yielded in the clock wise direction.
        :param :start = (segment, vtx_idx)
        yields: (segment, vtx_idx), where vtx_idx is index of 'point' in 'segment'
        """
        if start[0] is None:
            start = self.segment
        if start[0] is None:
            return
        seg_side = start
        while (1):
            yield seg_side
            seg, side = seg_side
            seg, other_side = seg.next[side]
            seg_side = seg, 1 - other_side
            if seg_side == start:
                return

    def insert_vector(self, vector):
        """
        Insert a vector between segments connected to the point.

        :param vector: (X, Y) ... any indexable pair.
        :return: ( (prev_seg, prev_side), (next_seg, next_side), wire )
        Previous segment side, and next segment side relative from inserted vector and the
        wire separated by the vector.
        """
        assert abs(vector[0])  > 1e-10 or abs(vector[1])  > 1e-10

        if self.segment[0] is None:
            return None
        vec_angle = np.angle(complex(vector[0], vector[1]))
        last = (4 * np.pi, None, None)
        self_segments = list(self.segments())
        self_segments.append(self.segment)
        for seg, vtx in self_segments:
            seg_vec = seg.vector
            if vtx == in_vtx:
                seg_vec *= -1.0
            angle = np.angle(complex(seg_vec[0], seg_vec[1]))
            da = angle - vec_angle
            if da < 0.0:
                da += 2 * np.pi
            if da >= last[0]:
                prev = (last[1], last[2])
                next = (seg, 1 - vtx)
                break

            last = (da, seg, vtx)
        wire = prev[0].wire[prev[1]]
        # assert wire == next[0].wire[next[1]]
        return (prev, next, wire)




    def join_segment(self, seg, vtx):
        """
        Connect a segment side to the point.
        """
        if self.is_free():
            if self.poly is not None:
                self.poly.free_points.remove(self)
            self.poly = None
            self.segment = (seg, vtx)

    def rm_segment(self, seg, vtx):
        """
        Disconnect segment side.
        """
        assert seg.vtxs[vtx] == self
        for new_seg, side in self.segments():
            if not new_seg == seg:
                self.segment = (new_seg, side)
                return
        assert seg.is_dendrite()
        self.set_polygon(seg.wire[left_side].polygon)

    def set_polygon(self, polygon):
        if self.poly is not None:
            self.poly.free_points.remove(self)
        self.poly = polygon
        polygon.free_points.add(self)
        self.segment = (None, None)

