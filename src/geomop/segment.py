import geomop.idmap as idmap
import numpy as np

in_vtx = left_side = 1
# vertex where edge comes in; side where next segment is connected through the in_vtx
out_vtx = right_side = 0
# vertex where edge comes out; side where next segment is connected through the out_vtx


class Segment(idmap.IdObject):

    def __init__(self, vtxs):
        self.vtxs = list(vtxs)
        # tuple (out_vtx, in_vtx) of point objects; segment is oriented from out_vtx to in_vtx
        self.wire = [None, None]
        # (left_wire, right_wire) - wires on left and right side
        self.next = [None, None]
        # (left_next, right_next); next edge for left and right side;
        self._vector = (self.vtxs[in_vtx].xy - self.vtxs[out_vtx].xy)


    def __repr__(self):
        next = [self._half_seg_repr(right_side), self._half_seg_repr(left_side)]
        return "Seg({}) [ {}, {} ] next: {} wire: {}".format(self.id, self.vtxs[out_vtx], self.vtxs[in_vtx], next,
                                                             self.wire)

    def _half_seg_repr(self, side):
        """
        Auxiliary method for __repr__.
        """
        if self.next[side] is None:
            return str(None)
        else:
            return (self.next[side][0].id, self.next[side][1])

    @staticmethod
    def side_to_vtx(side, side_vtx):
        tab = [[in_vtx, out_vtx], [out_vtx, in_vtx]]
        return tab[side][side_vtx]

    ##############################################
    # Data getters and generators.

    def point_ids(self):
        # Tuple of IDs of the endpoints.
        return (self.vtxs[out_vtx].id, self.vtxs[in_vtx].id)


    def polygons(self):
        """
        Return pair of polygons on the sides of the segment.
        :return: (right_side_polygon, left_side_polygon)
        """
        return (self.wire[right_side].polygon, self.wire[left_side].polygon)

    def is_dendrite(self):
        return self.wire[left_side] == self.wire[right_side]

    @property
    def vector(self):
        # Direction vector of the segment.
        #if np.any(self._vector != (self.vtxs[in_vtx].xy - self.vtxs[out_vtx].xy)):
        #    pass
        return self._vector.copy()

    def vector_(self):
        # Direction vector of the segment.
        return (self.vtxs[in_vtx].xy - self.vtxs[out_vtx].xy)

    def parametric(self, t):
        # Parametric function of the segment for t in (0, 1)
        return self.vector * t + self.vtxs[out_vtx].xy

    def previous(self, side):
        """
        Oposite of seg.next[side]. Implemented through loop around a node.
        :param seg:
        :param side:
        :return: (previous segment, previous side), i.e. prev_seg.next[prev_side] == (self, side)
        """
        vtx_idx = Segment.side_to_vtx(side, 0)
        vtx = self.vtxs[vtx_idx]
        for seg, side in vtx.segments(start=(self, vtx_idx)):
            pass
        return (seg, side)

    def point_side(self, pt):
        # Return local side of the given point or None.
        if pt == self.vtxs[out_vtx]:
            return out_vtx
        elif pt == self.vtxs[in_vtx]:
            return in_vtx
        else:
            return None


    ###############################
    # setting functions

    def _set_next_side(self, side, next_seg):
        """
        Auxiliary method for connect_* methods.
        """
        assert next_seg[0].vtxs[1 - next_seg[1]] == self.vtxs[side]
        # prev vtx of next segment == next vtx of self segment
        self.next[side] = next_seg

    def connect_vtx(self, vtx_idx, insert_info):
        """
        Connect 'self' segment to a non-free point.
        :param vtx_idx: out_idx / in_idx; specification of the segment's endpoint to connect.
        :param insert_info: (prev_side, next_side, wire)
                ... as returned by Point.insert_segment and self.vtx_insert_info
        TODO: merge with connect_free_vtx ... common notation fo insert info.
        """
        self.vtxs[vtx_idx].join_segment(self, vtx_idx)
        prev, next, wire = insert_info
        set_side = vtx_idx
        self._set_next_side(set_side, next)

        prev_seg, prev_side = prev
        prev_seg._set_next_side(prev_side, (self, 1 - set_side))
        self.wire[set_side] = wire
        # assert prev_seg.wire[prev_side] == wire    # this doesn't hold in middle of change operations

    def connect_free_vtx(self, vtx_idx, wire):
        """
        Connect 'self' segment to a free point.
        :param vtx_idx: out_idx / in_idx; specification of the segment's endpoint to connect.
        :param wire: Wire to set to the related side of the segment (in fact both sides should have same wire).
        """
        self.vtxs[vtx_idx].join_segment(self, vtx_idx)
        next_side = vtx_idx
        other_side = 1 - next_side
        self._set_next_side(next_side, (self, other_side))
        self.wire[next_side] = wire

    def vtx_insert_info(self, vtx_idx):
        """
        Return insert info for connecting after disconnect.
        """
        side_next = vtx_idx
        next = self.next[side_next]
        if next[0] == self:
            # veertex not conneected, i.e. dendrite tip
            return None
        wire = self.wire[side_next]

        side_prev = 1 - vtx_idx  # prev side is next side of oposite vertex
        prev = self.previous(side_prev)
        return (prev, next, wire)

    def disconnect_vtx(self, vtx_idx):
        """
        Disconect next links of one vtx side of self segment.
        :param vtx_idx: out_vtx or in_vtx
        """
        self.vtxs[vtx_idx].rm_segment(self, vtx_idx)
        seg_side_prev = 1 - vtx_idx
        seg_side_next = vtx_idx

        prev_seg, prev_side = self.previous(seg_side_prev)
        prev_seg.next[prev_side] = self.next[seg_side_next]
        self.next[seg_side_next] = (self, seg_side_prev)

    def disconnect_wires(self):
        """
        Remove segment -> wire and wire ->segment links.
        """
        for side in [left_side, right_side]:
            wire = self.wire[side]
            if wire.segment == (self, side):
                wire.segment = self.next[side]
                if wire.segment[0] == self:
                    wire.segment = self.next[wire.segment[1]]
                assert wire.segment[0].wire[wire.segment[1]] == wire, "wire.segment: {}, wire: {}".format(wire.segment,
                                                                                                          wire)
                assert not wire.segment[0] == self


    def is_on_x_line(self, xy):
        """
        Returns true if the segment is on the right horizontal halfline. starting in given point.

        Evaluation of condition is_on_line is unstable for xy close to the segment,
        however still the best approximation of the condition. Nevertheless,
        such case should not happen as the point would be snapped to the segment.

        Result for special cases (not clear if this is OK):
            - y_vtx_in == y_vtx_out on h-line:
                False

            - y_vtx_in > y_vtx_out:
              - vtx_in on h-line:  -> False
              - vtx_out on h-line: ->
                x_vtx_out > x -> True
                x_vtx_out < x  -> False
                x_vtx_out == x should not happen


            - y_vtx_in < y_vtx_out:
              - vtx_out on h-line:  -> False
              - vtx_in on h-line: ->
                x_vtx_in > x -> True
                x_vtx_in < x  -> False
                x_vtx_out == x should not happen

        :param xy: (x, y) left tip of the horizontal half line
        :return:
        """

        def min_max(aa):
            if aa[0] > aa[1]:
                return (aa[1], aa[0])
            return aa

        xx, yy = zip(self.vtxs[out_vtx].xy, self.vtxs[in_vtx].xy)

        min_y, max_y = min_max(yy)
        if min_y <= xy[1] < max_y:
            min_x, max_x = min_max(xx)
            x, y = xy
            if min_x > x:
                return True
            if max_x < x:
                return False
            is_on_line = (y - yy[0]) * (xx[1] - xx[0]) > (x - xx[0]) * (yy[1] - yy[0])
            if yy[1] < yy[0]:
                is_on_line = not is_on_line
            return is_on_line

        return False


