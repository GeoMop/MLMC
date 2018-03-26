#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as plt
#from matplotlib import collections  as mc
#from matplotlib import patches as mp

import plotly.offline as pl
import plotly.graph_objs as go



def _plot_polygon(polygon):
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


def plot_polygon_decomposition(decomp):
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
    for pt in decomp.points.values():
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
    pl.plot(fig, filename='polygons.html')
