#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

logger = logging.getLogger(__name__)

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    From https://gist.github.com/dwyerk/10561690
    and
    https://github.com/dwyerk/boundaries/blob/master/concave_hulls.ipynb

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller numbers don't fall inward as much as
                    larger numbers. Too large, and you lose everything!
    """
    import shapely.geometry as geometry
    from shapely.ops import cascaded_union, polygonize
    from scipy.spatial import Delaunay
    # import fiona

    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]

    # Lengths of sides of triangle
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5

    # Semiperimeter of triangle
    s = ( a + b + c ) / 2.0

    # Area of triangle by Heron's formula
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)

    # Here's the radius filter.
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))

    return cascaded_union(triangles), edge_points

def plot_polygon(polygon, ax=None, show=False):
    from descartes import PolygonPatch
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    margin = .3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999', ec='#000000', fill=True, zorder=-1)
    ax.add_patch(patch)

    if show:
        plt.show()
    return fig

if __name__ == '__main__':
    import fiona

    import shapely.geometry as geometry

    def plot_points(x, y, fig=None, show=True):
        if fig is None:
            fig = plt.figure(figsize=(10, 10))
        _ = plt.plot(x, y, 'o', color='#f16824')
        if show:
            plt.show()

    if False:
        input_shapefile = './boundaries-master/data/concave_demo_points.shp'
        shapefile = fiona.open(input_shapefile)
        points = [geometry.shape(point['geometry']) for point in shapefile]
        print("We have {0} points!".format(len(points)))

        # The following proves that I don't know numpy in the slightest
        x = [p.coords.xy[0] for p in points]
        y = [p.coords.xy[1] for p in points]

        point_collection = geometry.MultiPoint(list(points))

        # Plot points
        # plot_points(x, y, show=True)

        # Plot envelope
        fig = plot_polygon(point_collection.envelope)
        plot_points(x, y, fig=fig, show=True)

        # Convex hull
        convex_hull_polygon = point_collection.convex_hull
        fig = plot_polygon(convex_hull_polygon)
        plot_points(x, y, fig=fig, show=True)

        coords = np.array([point.coords[0] for point in points])

        # Concave hull
        concave_hull, edge_points = alpha_shape(points, alpha=1.87)
        fig = plot_polygon(concave_hull)
        plot_points(x, y, fig=fig, show=True)

        # Concave hull tighter
        alpha_range = [0.1, 10]
        for alpha in np.linspace(*alpha_range, 5):
            concave_hull, edge_points = alpha_shape(points, alpha=alpha)
            fig = plot_polygon(concave_hull)
            plot_points(x, y, fig=fig, show=True)

    if True:
        input_shapefile = './boundaries-master/data/demo_poly_scaled_points_join.shp'
        new_shapefile = fiona.open(input_shapefile)
        new_points = [geometry.shape(point['geometry']) for point in new_shapefile]
        print("We have {0} points!".format(len(new_points)))

        x = [p.coords.xy[0] for p in new_points]
        y = [p.coords.xy[1] for p in new_points]
        # fig = plt.figure(figsize=(10, 10))
        # plot_points(x, y, fig=fig, show=True)


        for alpha in np.arange(0.1, 1, 0.1):
            concave_hull, edge_points = alpha_shape(new_points, alpha=alpha)

            # print concave_hull
            lines = LineCollection(edge_points)
            fig = plot_polygon(concave_hull.buffer(1), ax=None)
            fig = plot_polygon(concave_hull, ax=plt.gca())
            plot_points(x, y, fig=fig, show=False)
            # plt.figure(figsize=(10, 10))
            plt.title('Alpha={0:0.2g} Delaunay triangulation'.format(alpha))
            ax = plt.gca()
            ax.add_collection(lines)

            delaunay_points = np.array([point.coords[0] for point in new_points])
            plt.plot(delaunay_points[:, 0], delaunay_points[:, 1], 'o', color='#f16824')

            plt.show()


    pass