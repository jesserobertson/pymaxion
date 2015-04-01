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

def spherical_to_cartesian(points):
    """ Convert speherical polar to cartesian coordinates

        Parameters:
            points - a 2 by N array of spherical polar coordinates (theta,
                phi), given in radians (phi = inclination, theta = rotation
                about polar axis)

        Returns:
            a 3 by N array containing (x, y, z) cartesian points
    """
    theta, phi = points
    return numpy.vstack([
        numpy.sin(theta) * numpy.cos(phi),
        numpy.sin(theta) * numpy.sin(phi),
        numpy.cos(theta)])

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
    theta = numpy.arccos(points[2])
    phi = numpy.arctan2(points[1], points[0])
    return numpy.vstack([theta, phi])

def latlong_to_spherical(points):
    """ Convert latitude and longitude into spherical polar
        coordinates (theta, phi) with radius unity.

        Parameters:
            a 2 by N array of latitude/longitude points,
                given in degrees N and degrees E

        Returns:
            a tuple containing (theta, phi) arrays for the points given
                (phi = inclination, theta = rotation about polar axis)
    """
    latitude, longitude = points
    theta, phi = 90. - latitude, longitude
    phi[longitude < 0.] += 360.
    return numpy.radians(numpy.vstack([theta, phi]))
