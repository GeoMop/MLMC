import numpy as np
import geomop.polygons as poly
import geomop.merge as merge
import geomop.polygons_io as poly_io
import geomop.format_last as lg
import geomop.layers_io
import geomop.geometry
#from geomop.plot_polygons import plot_polygon_decomposition









def make_frac_mesh(box, mesh_step, fractures, frac_step):
    """
    Make geometry and mesh for given 2d box and set of fractures.
    :param box: [min_point, max_point]; points are np.arrays
    :param fractures: Array Nx2x2, one row for every fracture given by endpoints: [p0, p1]
    :return: GmshIO object with physical groups:
        box: 1,
        fractures: 1000 + i, i = 0, ... , N-1
    """
    regions = make_regions(mesh_step, fractures, frac_step)
    decomp, reg_map = make_decomposition(box, fractures, regions)
    geom = fill_lg(decomp, reg_map, regions)
    return make_mesh(geom)


def add_reg(regions, name, dim, step=0.0, bc=False, not_used =False):
    reg = lg.Region(dict(name=name, dim=dim, mesh_step=step, boundary=bc, not_used=not_used))
    reg._id = len(regions)
    regions.append(reg)

def make_regions(mesh_step, fractures, frac_step):
    regions = []
    add_reg(regions, "NONE", -1, not_used=True)
    add_reg(regions, "bulk_0", 2, mesh_step)
    add_reg(regions, ".bc_inflow", 1, bc=True)
    add_reg(regions, ".bc_outflow", 1, bc=True)
    for f_id in range(len(fractures)):
        add_reg(regions, "frac_{}".format(f_id), 1, frac_step)
    return regions


def make_decomposition(box, fractures, regions):
    box_pd = poly.PolygonDecomposition()
    p00, p11 = box
    p01 = np.array([p00[0], p11[1]])
    p10 = np.array([p11[0], p00[1]])
    box_pd.add_line(p00, p01)
    seg_outflow, = box_pd.add_line(p01, p11)
    box_pd.add_line(p11, p10)
    seg_inflow, = box_pd.add_line(p10, p00)

    decompositions = [box_pd]
    for p0, p1 in fractures:
        pd = poly.PolygonDecomposition()
        pd.add_line(p0, p1)
        decompositions.append(pd)

    common_decomp, maps = merge.intersect_decompositions(decompositions)
    #plot_polygon_decomposition(common_decomp)
    #print(maps)

    # Map common_decomp objects to regions.
    none_region_id = 0
    box_reg_id = 1
    bc_inflow_id = 2
    bc_outflow_id = 3
    frac_id_shift = 4
    decomp_shapes = [common_decomp.points, common_decomp.segments, common_decomp.polygons]
    reg_map = [{key: regions[none_region_id] for key in decomp_shapes[d].keys()} for d in range(3)]
    for i_frac, f_map in enumerate(maps[1:]):
        for id, orig_seg_id in f_map[1].items():
            reg_map[1][id] = regions[frac_id_shift + i_frac]

    for id, orig_poly_id in maps[0][2].items():
        if orig_poly_id == 0:
            continue
        reg_map[2][id] = regions[box_reg_id]

    for id, orig_seg_id in maps[0][1].items():
        if orig_seg_id == seg_inflow.id:
            reg_map[1][id] = regions[bc_inflow_id]
        if orig_seg_id == seg_outflow.id:
            reg_map[1][id] = regions[bc_outflow_id]


    return common_decomp, reg_map


def fill_lg(decomp, reg_map, regions):
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
    geomop.layers_io.write_geometry("fractured_2d.json", geom)
    return geom


def make_mesh(geometry):
    return geomop.geometry.make_geometry(geometry=geometry, layers_file="fractured_2d.json", mesh_step=1.0)