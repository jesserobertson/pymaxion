#!/usr/bin/env python
""" file: coordinate_conversions.py
    author: Jess Robertson
            CSIRO Mineral Resources Flagship
    date:   Tuesday March 31, 2015

    description: An implementation of the Dymaxion projection, stolen from
        Mike Bostok's javascript implementation, and numpyified and generally
        Pythonized for use with Shapely objects
"""

import numpy
from numpy import sin, cos, degrees, radians, arcsin, arctan, arctan2, sqrt, pi, \
    vstack, asarray

def spherical_to_cartesian(points):
    """ Convert speherical polar to cartesian coordinates

        Parameters:
            points - a 2 by N array of spherical polar coordinates (theta,
                phi), given in radians (phi = inclination, theta = rotation
                about polar axis)

        Returns:
            a 3 by N array containing (x, y, z) cartesian points
    """
    return vstack([
        sin(points[0]) * cos(points[1]),
        cos(points[0]) * cos(points[1]),
        sin(points[1])])

def cartesian_to_spherical(points):
    """ Convert cartesian to spherical polar coordinates.

        Parameters:
            points - an 3 by N array of (x, y, z) cartesian coordinates,
                given in radians

        Returns:
            a 2 by N numpy array of (theta, phi) spherical coordinates for the
                points given (phi = inclination, theta = rotation about polar
                axis)
    """
    return vstack([
        arctan2(points[0], points[1]), 
        arctan2(points[2], sqrt(points[0] ** 2 + points[1] ** 2))])
    
def longlat_to_spherical(points):
    """ Convert longitude and latitude into spherical polar
        coordinates (theta, phi) with radius unity.
        
        Longitudes and latitudes are assumed to be long in [0, 360] 
        and lat in [-90, 90] degrees.

        Parameters:
            a 2 by N array of longitude/latitude points,
                given in degrees N and degrees E

        Returns:
            a tuple containing (theta, phi) arrays for the points given
                (theta = rotation about polar axis ~~ 'longitude',
                 phi = rotation from north pole ~~ 'latitude')
    """
    # Convert longitude and latitude to theta and phi
    return radians(points)

def sterographic_projection(centre):
    """ Transform function for points using a stereographic projection
        centred on the given centre
    """
    theta_c, phi_c = longlat_to_spherical(centre.x, centre.y)
    def _transform_fn(longitude, latitude):
        theta, phi = longlat_to_spherical(longitude, latitude)
        scale = 2 / (1 + sin(phi_c) * sin(phi) + cos(phi_c) * cos(phi) * cos(theta - theta_c))
        x = scale * cos(phi) * sin(theta - theta_c)
        y = scale * (cos(phi_c) * sin(phi) - sin(phi_c) * cos(phi) * cos(theta - theta_c))
        return x, y
    return _transform_fn

def gnomonic_projection(centre):
    """ Transform points using a gnomonic projection with centre point centre.
        
        All points should be given in degrees longitude and degrees latitude

        Arguments:
            centre - the centre point of the projection - this is 
                where the gnominic plane touches the surface of the 
                earth. Given as a tuple of longitude and latitude in degrees
        
        Returns:
            a function to transform a point into the projective plane
    """
    theta_c, phi_c = longlat_to_spherical(centre.x, centre.y)
    def _transform_fn(longitude, latitude):
        theta, phi = longlat_to_spherical(longitude, latitude)
        scale = 1 / (sin(phi_c) * sin(phi) + cos(phi_c) * cos(phi) * cos(theta - theta_c))
        x = scale * cos(phi) * sin(theta - theta_c)
        y = scale * (cos(phi_c) * sin(phi) - sin(phi_c) * cos(phi) * cos(theta - theta_c))
        return x, y
    return _transform_fn
