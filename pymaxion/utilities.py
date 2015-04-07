import fiona
import shapely
import shapely.ops
import pkg_resources
import numpy
import os

from .conversions import cartesian_to_spherical

SHAPEFILE = pkg_resources.resource_filename('pymaxion', 'resources/ne_110m_land.shp')

def get_land(shapefile=None, bbox=None):
    """ Get land from a shapefile
    """
    # Load up shape data from shapefile
    with fiona.drivers():
        with fiona.open(SHAPEFILE) as source:
            if bbox:
                source = source.filter(bbox=bbox)
            shapes = shapely.ops.cascaded_union([
                      shapely.geometry.asShape(s['geometry'])
                      for s in source])

    # There's a small issue with part of the polygon for North
    # America, we'll just clip it out for now
    shapes = shapes.difference(shapely.geometry.box(-132.71, 54.04, -132.72, 54.05))

    # Clip if we've got something to clip
    if bbox:
        bbox_shape = shapely.geometry.box(*bbox)
        return shapes.intersection(bbox_shape)
    else:
        return shapes

def geodesic_linspace(a, b, npoints=50, inclusive=True):
    """ Construct a 'linspace' along a geodesic for a spherical earth between two points
    """
    # Make sure that everything is set up right
    long_a, lat_a = numpy.radians(a)
    long_b, lat_b = numpy.radians(b)
    if inclusive:
        fraction = numpy.linspace(0, 1, npoints)
    else:
        fraction = numpy.linspace(0, 1, npoints+1)[:-1]

    # Parameterize arc between the points
    d = numpy.arccos(numpy.sin(lat_a) * numpy.sin(lat_b) + numpy.cos(lat_a) * numpy.cos(lat_b) * numpy.cos(long_a - long_b))
    A = numpy.sin((1 - fraction) * d) / numpy.sin(d)
    B = numpy.sin(fraction * d) / numpy.sin(d)

    # Calculate cartesian points
    points = \
        numpy.vstack([A * numpy.cos(lat_a) * numpy.cos(long_a)
                        + B * numpy.cos(lat_b) * numpy.cos(long_b),
                      A * numpy.cos(lat_a) * numpy.sin(long_a)
                        + B * numpy.cos(lat_b) * numpy.sin(long_b),
                      A * numpy.sin(lat_a) + B * numpy.sin(lat_b)])

    # Convert back to latitude and longitude and return
    return numpy.degrees(
         cartesian_to_spherical(points[0], points[1], points[2]))
