import numpy as np
import geomop.polygons as poly
from geomop.plot_polygons import plot_polygon_decomposition


def get_first_item(map):
    for map_d in map:
        for key, item in map_d.items():
            return item

def add_to_region_map(region_map, map, region_id):
    # drop None items from map
    map = [ { key:item for key, item in map_d.items() if item is not None } for map_d in map]
    unique_value = get_first_item(map)
    for reg_map_d, map_d in zip(region_map, map):
        # For single dimension
        for id, orig_id in map_d.items():
            assert orig_id == unique_value, "{} != {}".format(orig_id, unique_value)
            reg_map_d[id] = region_id


def make_decomposition(box, fractures):
    box_pd = poly.PolygonDecomposition()
    p00, p11 = box
    p01 = np.array([p00[0], p11[1]])
    p10 = np.array([p11[0], p00[1]])
    box_pd.add_line(p00, p01)
    box_pd.add_line(p01, p11)
    box_pd.add_line(p11, p10)
    box_pd.add_line(p10, p00)

    decompositions = [box_pd]
    for p0, p1 in fractures:
        pd = poly.PolygonDecomposition()
        pd.add_line(p0, p1)
        decompositions.append(pd)

    common_decomp, maps = poly.intersect_decompositions(decompositions)
    plot_polygon_decomposition(common_decomp)

    # Map common_decomp objects to region IDs
    decomp_shapes = [common_decomp.points, common_decomp.segments, common_decomp.polygons]
    reg_map = [{key: None for key in decomp_shapes[d].keys()} for d in range(3)]
    add_to_region_map(reg_map, maps[0], 0)
    for i_frac, f_map in enumerate(maps[1:]):
        add_to_region_map(reg_map, f_map, 1000 + i_frac)

    # Check that reg_map is fully defined.
    for reg_map_d in reg_map:
        for val in reg_map_d.values():
            assert val is not None

    return common_decomp, reg_map


def fill_lg(decomp, reg_map):
    """
    Create LayerGeometry object.
    """
    


def make_frac_mesh(box, fractures):
    """
    Make geometry and mesh for given 2d box and set of fractures.
    :param box: [min_point, max_point]; points are np.arrays
    :param fractures: Array Nx2x2, one row for every fracture given by endpoints: [p0, p1]
    :return: GmshIO object with physical groups:
        box: 0,
        fractures: 1000 + i, i = 0, ... , N-1
    """
    decomp, reg_map = make_decomposition(box, fractures)
    fill_lg(decomp, reg_map)
