#!/usr/bin/env python
""" file: coordinate_conversions.py
    author: Jess Robertson
            CSIRO Mineral Resources Flagship
    date:   Tuesday March 31, 2015

    description: An implementation of the Dymaxion projection, stolen from
        Mike Bostok's javascript implementation, and numpyified and generally
        Pythonized for use with Shapely objects
"""

from __future__ import division, print_function

import numpy
from numpy import sin, cos, degrees, radians, arcsin, arctan, arctan2, sqrt, pi, \
    vstack, asarray, pi

def spherical_to_cartesian(theta, phi):
    """ Convert speherical polar to cartesian coordinates

        Parameters:
            points - a 2 by N array of spherical polar coordinates (theta,
                phi), given in radians (phi = inclination, theta = rotation
                about polar axis)

        Returns:
            a 3 tuple array containing (x, y, z) cartesian points
    """
    return (sin(theta) * cos(phi), cos(theta) * cos(phi), sin(phi))

def cartesian_to_spherical(x, y, z):
    """ Convert cartesian to spherical polar coordinates.

        Parameters:
            points - an 3 by N array of (x, y, z) cartesian coordinates,
                given in radians

        Returns:
            a 2 by N numpy array of (theta, phi) spherical coordinates for the
                points given (phi = inclination, theta = rotation about polar
                axis)
    """
    return (arctan2(x, y), arctan2(z, sqrt(x ** 2 + y ** 2)))

def longlat_to_spherical(longitude, latitude):
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
    return (radians(longitude), radians(latitude))

def spherical_to_longlat(theta, phi):
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
    return (degrees(theta), degrees(phi))
