import geomop.polygons as polygons
from geomop.decomp import PolygonChange, in_vtx, out_vtx, right_side, left_side

def deep_copy(self):
    """
    Perform deep copy of polygon decomposition without preserving object IDs.
    :return: (copy_decomp, (point_map, segment_map, polygon_map)), decomposition copy, new id to old id maps
    """
    assert False, "Should not be used"

    # TODO: use ID maps to map segment and polygon ids.
    id_maps = ({}, {}, {})
    decomp = PolygonDecomposition()

    for pt in self.points.values():
        decomp.points.append(Point(pt.xy, poly=None), id=pt.id)
        id_maps[0][pt.id] = pt.id

    seg_orig_to_new = {}
    for seg in self.segments.values():
        new_seg = decomp.make_segment(seg.point_ids())
        id_maps[1][new_seg.id] = seg.id
        seg_orig_to_new[seg.id] = new_seg.id

    # FixMe: map orig segment IDs to new IDs
    for poly in self.polygons.values():
        outer_wire = [seg_orig_to_new[seg.id] for seg, side in poly.outer_wire.segments()]
        holes = []
        for hole in poly.outer_wire.childs:
            wire = [seg_orig_to_new[seg.id] for seg, side in hole.segments()]
            holes.append(wire)
        free_points = [pt.id for pt in poly.free_points]
        new_poly = decomp.make_polygon(outer_wire, holes, free_points)
        id_maps[2][new_poly.id] = poly.id

    decomp.set_wire_parents()

    # decomp.check_consistency()
    return decomp, id_maps


def intersect_single(decomp, other, merge_tol = 1e-10):
    """
    TODO: move to separate intersection module.

    Make new decomposition that is intersection of 'self' and 'other', that is polygons of
    both 'self' and 'other' are further split into polygons of resulting decomposition.
    A copy of 'self' is used as starting point, adding incrementaly segments of 'other'.

    TODO: Add information about linked nodes.

    :param other: PolygonDecomposition.
    :return: (decomp, maps_self, maps_other)
    Returns 'decomp' the intersection decomposition and maps that
    maps objects of the new decompositioon to objects of original decomposition or to None.
    Maps are set only for new objects, i.e. in map_self we omit identities while in
    maps_other we omit None values.

    Algorithm:
    - add segments of 'other' to 'self' keeping map from segments of new_decomp to
    pairs (self_segment, other_segment), one of them may be None.
    - assign maps_other[2] for new polygons neighbouring to other's segments
    - this maps all new polygons near boundary of other's polygons
    - use DFS to assign remaining new polygons in 'interior'

    TODO: Implement clear interface to PolygonDecomposition with history of internal elementary operations
    in particular segment splitting and line splitting. Then we can remove several hacks here.
    """
    save_tol = decomp.tolerance
    decomp.tolerance = merge_tol

    maps_self = [ {}, {}, {}]
    maps_other = [ {}, {}, {}]
    # for dim in range(3):
    #     maps_self[dim] = {obj.id: obj.id for obj in decomp.decomp.shapes[dim].values()}
    #     maps_other[dim] = {obj.id: None for obj in decomp.decomp.shapes[dim].values()}

    other_point_map = {}
    for pt in other.points.values():
        n_points = len(decomp.points)
        new_pt = decomp.add_point(pt.xy)
        # Hack to identify split segment
        seg, vtx_side = new_pt.segment
        if seg is not None and n_points < len(decomp.points):
            prev_seg, prev_side = seg.next[out_vtx]
            assert prev_seg.next[in_vtx][0] == seg
            #assert seg.id not in maps_self[1]
            maps_self[1][seg.id] = maps_self[1].setdefault(prev_seg.id, prev_seg.id)
            #maps_other[1].setdefault(seg.id, None)
        maps_other[0][new_pt.id] = pt.id
        other_point_map[pt.id] = new_pt.id
        #maps_self[0].setdefault(new_pt.id, new_pt.id)

    for seg in other.segments.values():
        new_a_pt, new_b_pt = [decomp.points[other_point_map[pt.id]] for pt in seg.vtxs]
        if new_a_pt == new_b_pt:
            continue
        other_right_poly, other_left_poly = seg.polygons()
        # print(decomp)
        # print('add line {} {}'.format(a, b))
        line_div = decomp._add_line_seg_intersections(new_a_pt, new_b_pt)
        for t, (mid_pt, seg_a, seg_b) in line_div.items():
            maps_self[1][seg_b.id] = maps_self[1].get(seg_a.id, seg_a.id)
            assert seg_a.id not in maps_other[1]
            #maps_other[1][seg_b.id] = maps_other[1].get(seg_a.id, None)  # Should still be None, unless there is common edge

        for new_seg, change, side in decomp._add_line_new_segments(new_a_pt, new_b_pt, line_div):
            maps_other[1][new_seg.id] = seg.id
            #maps_self[1].setdefault(new_seg.id, None)

            # Update polygon maps for a segment which is part of segment in other decomposition.
            change_type, orig_poly, new_poly = change
            assert change_type not in [PolygonChange.remove, PolygonChange.join]
            if change_type == PolygonChange.add or change_type == PolygonChange.split:
                # Fix maps for changed polygons
                maps_self[2][new_poly.id] = maps_self[2].get(orig_poly.id, orig_poly.id)
                maps_other[2][new_poly.id] = maps_other[2].get(orig_poly.id, None)
            # Just omit changes: PolygonChange.shape, PolygonChange.none

            # Partial fill of maps_other[2]
            if side:
                r, l = right_side, left_side
            else:
                l, r = right_side, left_side
            maps_other[2][new_seg.wire[r].polygon.id] = other_right_poly.id
            maps_other[2][new_seg.wire[l].polygon.id] = other_left_poly.id

    # Fill remaining maps_other[2] items using DFS
    for poly_id in decomp.polygons.keys():
        maps_other[2].setdefault(poly_id, None)
    for new_poly, other_poly in list(maps_other[2].items()):
        if other_poly != None:
            stack = [new_poly]
            while stack:
                p_id = stack.pop(-1)
                maps_other[2][p_id] = other_poly
                for seg, side in decomp.polygons[p_id].outer_wire.segments():
                    ngh_poly = seg.wire[1 - side].polygon
                    if maps_other[2][ngh_poly.id] is None:
                        stack.append(ngh_poly.id)

    # # Check
    # for p in decomp.polygons.values():
    #     assert maps_other[2][p.id] is not None
    #     assert maps_self[2][p.id] is not None

    # assert decomp.check_consistency()
    decomp.tolerance = save_tol
    return (decomp, maps_self, maps_other)







def intersect_decompositions(decomps, merge_tol = 1e-10):
    """
    Intersection of a list of decompositions. Segments and polygons are subdivided.

    :param decomps: List of PolygonDecomposition objects to itersect.
    :return: (common_decomp, poly_maps)
    common_decomp - resulting merged/intersected decomposition.
    poly_maps - List of maps, one for every input decomposition. For single decomp the map
    consists of maps for every dimension, [map_0d, map_1d, map_2d].
    map_Nd - is a dict mapping IDs of sommon_decomp objects to IDs of decomp objects.
    Objects of common_decomp that have no preimage in decomp are omitted.

    TODO: For larger number of intersectiong decompositions, it would be better to
    use a binary tree reduction instead of linear pass to have n log(n) complexity of map updating.
    """

    common_decomp = polygons.PolygonDecomposition()
    all_maps = []
    for decomp in decomps:
        print("frac : {} #pt: {} #seg: {} #wires: {} #poly: {}".format(
            len(all_maps),
            len(common_decomp.points),
            len(common_decomp.segments),
            len(common_decomp.decomp.wires),
            len(common_decomp.polygons)))
        common_decomp, common_maps, decomp_maps = intersect_single(common_decomp, decomp, merge_tol)
        decomp_maps = [ { key: val for key,val in map.items() if val is not None} for map in decomp_maps ]
        for one_decomp_maps in all_maps:
            for one_dim_map, common_map in zip(one_decomp_maps, common_maps):
                for new_id, orig_id in common_map.items():
                    if orig_id in one_dim_map:
                        assert new_id not in one_dim_map or new_id == orig_id
                        one_dim_map[new_id] = one_dim_map[orig_id]

        all_maps.append(decomp_maps)

        # check
        for dim in range(3):
            orig_id_set = { val for val in decomp_maps[dim].values()}
            for obj_id in decomp.decomp.shapes[dim].keys():
                assert obj_id in orig_id_set, "dim:{} id:{}".format(dim, obj_id)

    return common_decomp, all_maps
