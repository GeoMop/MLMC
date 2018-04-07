import geomop.bspline as bs
import geomop.format_last as gs



def bs_zsurface_read(z_surface_io):
    io = z_surface_io
    u_basis = bs.SplineBasis(io.u_degree, io.u_knots)
    v_basis = bs.SplineBasis(io.v_degree, io.v_knots)

    z_surf = bs.Surface( (u_basis, v_basis), io.poles, io.rational)
    surf = bs.Z_Surface(io.orig_quad, z_surf)
    surf.transform(io.xy_map, io.z_map)
    return surf

def bs_zsurface_write(z_surf):


    xy_map, z_map = z_surf.get_transform()
    config = dict(
        u_degree = z_surf.u_basis.degree,
        u_knots = z_surf.u_basis.knots.tolist(),
        v_degree = z_surf.v_basis.degree,
        v_knots = z_surf.v_basis.knots.tolist(),
        rational = z_surf.z_surface.rational,
        poles = z_surf.z_surface.poles.tolist(),
        orig_quad = z_surf.orig_quad.tolist(),
        xy_map = xy_map.tolist(),
        z_map = z_map.tolist()
    )
    return gs.SurfaceApproximation(config)
