import numpy as np
import geomop.polygons as poly
import geomop.merge as merge
import geomop.polygons_io as poly_io
import geomop.format_last as lg
import geomop.layers_io
import geomop.geometry
from geomop.plot_polygons import plot_polygon_decomposition









def make_frac_mesh(root_polygon, mesh_step:float, fractures, frac_step:float, mesh_base="fractured_2d"):
    """
    :param root_polygon: List[Point2d]
    :param fractures: List[(Point2d, Point2d)]

    Make geometry and mesh for given 2d box and set of fractures.
    :param box: [min_point, max_point]; points are np.arrays
    :param fractures: Array Nx2x2, one row for every fracture given by endpoints: [p0, p1]
    :return: GmshIO object with physical groups:
        "bulk": 1
        "side_<n>", n + 1
        "frac_<n>", 1000 + n
    """
    regions = []
    add_reg(regions, "NONE", -1, not_used=True)
    i_r_bulk = add_reg(regions, "bulk", 2, mesh_step)
    i_r_side = [
        add_reg(regions, "side_{}".format(s_id), 1, bc=True)
        for s_id in range(len(root_polygon))
    ]
    i_r_frac = [
        add_reg(regions, "frac_{}".format(f_id), 1, frac_step)
        for f_id in range(len(fractures))
    ]
    decomp, reg_map = make_decomposition(root_polygon, fractures, regions, i_r_bulk, i_r_side, i_r_frac)
    geom = fill_lg(decomp, reg_map, regions, mesh_base=mesh_base)
    return make_mesh(geom)


def add_reg(regions, name, dim, step=0.0, bc=False, not_used =False):
    reg = lg.Region(dict(name=name, dim=dim, mesh_step=step, boundary=bc, not_used=not_used))
    reg._id = len(regions)
    regions.append(reg)
    return reg._id



def make_decomposition(root_polygon_points, fractures, regions, i_r_bulk, i_r_side, i_r_frac):
    # Create boundary polygon
    box_pd = poly.PolygonDecomposition()
    box_pd.tolerance = 1
    last_pt = root_polygon_points[-1]
    side_segments = {}
    for i_side, pt in enumerate(root_polygon_points):
        sub_segments = box_pd.add_line(last_pt, pt, attr=regions[i_r_side[i_side]])
        last_pt = pt
        assert type(sub_segments) == list and len(sub_segments) == 1
        seg = sub_segments[0]
        side_segments[seg.id] = i_side
    assert len(box_pd.polygons) == 2
    box_pd.polygons[1].attr = regions[i_r_bulk]

    # Add fractures
    for i_fr, (p0, p1) in enumerate(fractures):
        segments = box_pd.add_line(p0, p1, attr=regions[i_r_frac[i_fr]])


    #common_decomp, maps = merge.intersect_decompositions(decompositions)
    plot_polygon_decomposition(box_pd)
    #print(maps)

    # Map common_decomp objects to regions.
    none_region_id = 0
    decomp_shapes = [common_decomp.points, common_decomp.segments, common_decomp.polygons]
    reg_map = [{key: regions[none_region_id] for key in decomp_shapes[d].keys()} for d in range(3)]

    for i_frac, f_map in enumerate(maps[1:]):
        for id, orig_seg_id in f_map[1].items():
            reg_map[1][id] = regions[i_r_frac[i_frac]]

    for id, orig_poly_id in maps[0][2].items():
        if orig_poly_id == 0:
            continue
        reg_map[2][id] = regions[i_r_bulk]

    for id, orig_seg_id in maps[0][1].items():
        if orig_seg_id in side_segments:
            reg_map[1][id] = regions[i_r_side[side_segments[orig_seg_id]]]


    return common_decomp, reg_map


def fill_lg(decomp, reg_map, regions, mesh_base="fractured_2d"):
    """
    Create LayerGeometry object.
    """
    nodes, topology = poly_io.serialize(decomp)

    geom = lg.LayerGeometry()
    geom.version
    geom.regions = regions



    iface_ns = lg.InterfaceNodeSet(dict(
        nodeset_id = 0,
        interface_id = 0
    ))
    layer = lg.FractureLayer(dict(
        name = "layer",
        top = iface_ns,
        polygon_region_ids = [ reg_map[2][poly.id]._id for poly in decomp.polygons.values() ],
        segment_region_ids = [ reg_map[1][seg.id]._id for seg in decomp.segments.values() ],
        node_region_ids = [ reg_map[0][node.id]._id for node in decomp.points.values() ]
    ))
    geom.layers = [ layer ]
    #geom.surfaces = [ClassFactory(Surface)]

    iface = lg.Interface(dict(
        surface_id = None,
        elevation = 0.0
    ))
    geom.interfaces = [ iface ]
    #geom.curves = [ClassFactory(Curve)]
    geom.topologies = [ topology ]

    nodeset = lg.NodeSet(dict(
        topology_id = 0,
        nodes = nodes
    ))
    geom.node_sets = [ nodeset ]
    geomop.layers_io.write_geometry(mesh_base + ".json", geom)
    return geom


def make_mesh(geometry, mesh_base="fractured_2d"):
    return geomop.geometry.make_geometry(geometry=geometry, layers_file=mesh_base + ".json", mesh_step=1.0)