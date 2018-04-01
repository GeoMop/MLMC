import geomop.idmap as idmap
from .segment import right_side, left_side, out_vtx, in_vtx
import numpy as np

class Polygon(idmap.IdObject):

    def __init__(self, outer_wire):
        self.outer_wire = outer_wire
        # outer boundary wire
        self.free_points = set()
        # Dict ID->pt of free points inside the polygon.

    def __repr__(self):
        outer = self.outer_wire.id
        return "Poly({}) out wire: {}".format(self.id, outer)

    def is_outer_polygon(self):
        return self.outer_wire.is_root()

    def depth(self):
        """
        LAYERS
        Return depth of the polygon. That is number of other polygons it is contained in.
        :return: int
        """
        depth = 0
        wire = self.outer_wire
        while (not wire.is_root()):
            depth += 1
            wire = wire.parent
        return depth

    def vertices(self):
        """
        LAYERS
        Return list of polygon vertices (point objects) in counter clockwise direction.
        :return:
        """
        if self.outer_wire.is_root():
            return []
        return [seg.vtxs[side] for seg, side in self.outer_wire.segments()]

    def child_polygons(self):
        """
        Yield all child polygons, i.e. polygons inside the self.outer_wire.
        """
        yield self
        for wire in self.outer_wire.child_wires():
            if wire == wire.polygon.outer_wire:
                yield wire.polygon

    def contains_point(self, xy):
        """
        Returns true if polygon contains the point.
        :param xy: array [x,y]
        :return:
        """
        if not self.outer_wire.contains_point(xy):
            return False
        for wire in self.outer_wire.childs:
            if wire.contains_point(xy):
                return False
        return True



class Wire(idmap.IdObject):

    def __init__(self):
        self.parent = self
        # Wire that contains this wire. None for the global outer boundary.
        # Parent relations are independent of polygons.
        self.childs = set()

        self.polygon = None
        # Polygon of this wire
        self.segment = (None, None)
        # One segment of the wire.

    def __eq__(self, other):
        # None is wire in infinity
        if self is None or other is None:
            return False
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __repr__(self):
        if self.is_root():
            return "Wire({}) root, poly: {}, childs: {}". \
                format(self.id, self.polygon.id, [ch.id for ch in self.childs])
        return "Wire({}) seg: {} poly: {} parent: {} childs: {}". \
            format(self.id, (self.segment[0].id, self.segment[1]),
                   self.polygon.id, self.parent.id, [ch.id for ch in self.childs])

    def repr_id(self):
        return self.id

    def is_root(self):
        return self.parent is None

    def set_parent(self, parent_wire):
        if self.parent is not None:
            assert self in self.parent.childs or self.parent == self
            self.parent.childs.discard(self)
        self.parent = parent_wire
        if parent_wire is not None:
            parent_wire.childs.add(self)

    # def segments(self, start = (None, None), end = (None, None)):
    #     """
    #     DEBUG VERSION.
    #     Yields all (segmnet, side) of the same wire as the 'start' segment side,
    #     up to end segment side.
    #     """
    #     if self.is_root():
    #         return
    #     if start[0] is None:
    #         start = self.segment
    #     if end[0] is None:
    #         end = start
    #
    #     seg_side = start
    #     visited = []
    #     while (1):
    #         visited.append( (seg_side[0], seg_side[1]) )
    #         yield seg_side
    #         segment, side = seg_side
    #         seg_side = segment.next[side]
    #         if seg_side == end:
    #             break
    #         if (seg_side[0], seg_side[1]) in visited:
    #
    #             assert False, "Repeated seg: {}\nVisited: {}".format(seg_side, visited)
    #         assert not seg_side == start, "Inifinite loop."

    def segments(self, start=(None, None), end=(None, None)):
        """
        Yields all (segmnet, side) of the same wire as the 'start' segment side,
        up to end segment side.
        """
        if self.is_root():
            return
        if start[0] is None:
            start = self.segment
        if end[0] is None:
            end = start

        seg_side = start
        while (1):
            yield seg_side
            segment, side = seg_side
            seg_side = segment.next[side]
            if seg_side == end:
                break
            assert not seg_side == start, "Inifinite loop."

    # def outer_segments(self):
    #     """
    #     :return: List of boundary componencts without tails. Single component is list of segments (with orientation)
    #     that forms outer boundary, i.e. excluding internal tails, i.e. segments appearing just once.
    #     TODO: This is not reliable for dendrites with holes. We should use whole wire for plotting.
    #     Then remove this method.
    #     """
    #     for seg, side  in self.segments():
    #         if not seg.is_dendrite():
    #             yield (seg, side)

    def neighbors(self):
        """
        Return list of all neighoring wires with same depth.
        :return:
        """
        return [seg.wire[1 - side] for seg, side in self.segments()]

    def contains_point(self, xy):
        """
        Check if the wire contains given point.
        TODO: use tolerance.
        :param xy: array [x,y]
        :return:
        """
        if self.is_root():
            return True
        n_isec = 0
        for seg, side in self.segments():
            n_isec += int(seg.is_on_x_line(xy))
        if n_isec % 2 == 1:
            return True
        else:
            return False

    def contains_wire(self, wire):
        """
        Check if the 'self' wire contains other wire.
        Translates to call of 'contains_point' for carefully selected point.
        TODO: use tolerance.
        """
        if self.is_root():
            return True
        if wire.is_root():
            return False

        # test if a point of wire is inside 'self'
        seg, side = wire.segment
        tang = seg.vector
        norm = np.array([-tang[1], tang[0]])
        if side == right_side:
            norm = -norm
        eps = 1e-10  # TODO: review this value to be as close to the line as possible while keep intersection work
        inner_point = seg.vtxs[out_vtx].xy + 0.5 * tang + eps * norm
        return self.contains_point(inner_point)

    def child_wires(self):
        """
        Yields all child wires recursively.
        :return:
        """
        if self._get_child_passed:
            assert False, "Cyclic wire links."
        self._get_child_passed = True

        yield self
        for child in self.childs:
            yield from child.child_wires()


