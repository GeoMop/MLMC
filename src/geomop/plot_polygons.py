#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt
#from matplotlib import collections  as mc
#from matplotlib import patches as mp




def _plot_polygon(polygon):
    import plotly.graph_objs as go

    if polygon is None or polygon.displayed or polygon.outer_wire.is_root():
        return []

    # recursion
    assert polygon.outer_wire.parent.polygon != polygon, polygon.id
    patches = _plot_polygon(polygon.outer_wire.parent.polygon)
    pts = [pt.xy for pt in polygon.vertices()]
    X, Y = zip(*pts)
    plot_poly = go.Scatter(
        x=X,
        y=Y,
        #mode='markers',
        mode='none',
        fill='toself',
        opacity=0.7
    )
    ## patches.append(mp.Polygon(pts))
    patches.append(plot_poly)
    return patches


def plot_polygon_decomposition(decomp, points=None):
    import plotly.offline as pl
    import plotly.graph_objs as go

    ## fig, ax = plt.subplots()

    # polygons
    for poly in decomp.polygons.values():
        poly.displayed = False

    patches = []
    for poly in decomp.polygons.values():
        patches.extend(_plot_polygon(poly))
    ## p = mc.PatchCollection(patches, color='blue', alpha=0.2)

    ## ax.add_collection(p)

    for s in decomp.segments.values():
        ## ax.plot((s.vtxs[0].xy[0], s.vtxs[1].xy[0]), (s.vtxs[0].xy[1], s.vtxs[1].xy[1]), color='green')
        patches.append(go.Scatter(
            x=[s.vtxs[0].xy[0], s.vtxs[1].xy[0]],
            y=[s.vtxs[0].xy[1], s.vtxs[1].xy[1]],
            line = {
                'color': 'green',
            }
        ))

    x_pts = []
    y_pts = []
    if points is None:
        points = decomp.points.values()
    for pt in points:
        x_pts.append(pt.xy[0])
        y_pts.append(pt.xy[1])
    ## ax.plot(x_pts, y_pts, 'bo', color='red')
    patches.append(go.Scatter(
        x=x_pts,
        y=y_pts,
        mode='markers',
        marker=dict(color='red')))
    ## plt.show()
    fig = go.Figure(data=patches)
    fig.update_layout(width=1600, height=1600)
    pl.plot(fig, filename='polygons.html')


def plot_decomp_segments(decomp, points_a=[], points_b=[]):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import collections  as mc

    lines = [[seg.vtxs[0].xy, seg.vtxs[1].xy] for seg in decomp.segments.values()]
    lc = mc.LineCollection(lines, linewidths=1)

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    Point = next(iter(decomp.points.values())).__class__
    for pt_list in [decomp.points.values(), points_a, points_b]:
        points = np.array([pt.xy if type(pt) is Point else pt for pt in pt_list])
        if len(points) > 0 :
            ax.scatter(points[:, 0], points[:, 1], s=1)

    ax.autoscale()
    ax.margins(0.1)
    fig.savefig("fractures.pdf")
    plt.show()