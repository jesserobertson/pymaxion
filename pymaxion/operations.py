#!/usr/bin/env python
""" file: operations.py
    author: Jess Robertson
            CSIRO Mineral Resources Flagship
    date:   Tuesday March 31, 2015

    description: Various functions to perform operations on polygons etc
"""

from __future__ import division, print_function

import numpy
import shapely.ops, shapely.geometry


def median_location(geom):
    """ Return a representative median location for a given geometry
        
        The representative location is defined for a LineString as the
        median longitude and median latitude values for the line's 
        coordinates. For a polygon, it's defined as the represenative 
        location of its boundary. For a multigeometry instance, it's
        defined as the median of the representative locations of its parts.
        
        Note that in the numpy implementation of median, when N is even, 
        the mean of the two middle points is taken.
        
        Parameters:
            geom - a Shapely geometry instance
        
        Returns:
            longitude, latitude - the representative location, as a tuple
    """
    # Deal with multigeometry instances first
    if geom.type.startswith('Multi'):
        result = numpy.asarray([median_location(g) for g in geom])
        if len(result.shape) == 1:
            return tuple(result.ravel())
        else:
            return tuple(numpy.median(result, axis=0).ravel())
    elif geom.type == 'Polygon':
        return median_location(geom.boundary)
    elif geom.type == 'LineString':
        return tuple(numpy.median(geom.coords, axis=0).ravel())
    else:
        raise ValueError(
            "Don't know what to do with geometry {0}".format(geom))


def fix_longitude(geom, max_distance=180, degrees=True):
    """ Rework longitude locations so that polygon is simple
    
        Parameters:
            geom - a Shapely geometry
            max_distance - the maximum distance away that points are 
                expected to be, given as a latitude, longitude tuple. 
                Optional, defaults to (180, 90). 
            degrees - whether the geometry locations are in degrees or not
        
        Returns:
            a new geometry instance which has everything close together
    """
    # Get a representative point
    rlong, rlat = median_location(geom)
    if rlong > 180 and degrees:
        rlong -= 360
    elif rlong < -180 and degrees:
        rlong += 360
    elif rlong > numpy.pi and not degrees:
        rlong -= 2 * numpy.pi
    elif rlong < -numpy.pi and not degrees:
        rlong += 2 * numpy.pi
    
    # Function to transform points
    def _transform_fn(x, y):
        """ If the points are more than long_box away from the median 
            in longitude, or lat_box degrees in latitude, then we probably 
            need to reshuffle them
        """
        # Assume we've just got individual points first
        try:
            if abs(x - rlong) > max_distance:
                x -= 2 * max_distance * numpy.sign(x - rlong)
            return x, y
        except ValueError:
            pass
        
        # We've probably got numpy arrays so try that instead
        x_move = numpy.abs(x - rlong) > max_distance
        x[x_move] -= 2 * max_distance * numpy.sign(x[x_move] - rlong)
        return x, y
    
    # Return transformed shape
    return shapely.ops.transform(_transform_fn, geom)
