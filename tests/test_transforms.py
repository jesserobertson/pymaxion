#!/usr/bin/env python

import unittest

from pymaxion import DymaxionProjection
from pymaxion.utilities import get_land
from pymaxion.conversions import *

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fiona
import shapely
import pkg_resources

class TestConversions(unittest.TestCase):

    def setUp(self):
        land = get_land()
        area_idx = numpy.argsort([l.area for l in land])
        self.shapes = shapely.geometry.MultiPolygon(
            [land[int(i)] for i in area_idx[-1:-20:-1]])

    def test_cartesian_to_spherical(self):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.set_zlim(-1, 1)

        for shape in self.shapes:
            shape = shapely.ops.transform(longlat_to_spherical,
                                          shape)
            shape = shapely.ops.transform(spherical_to_cartesian,
                                          shape)
            if shape.boundary.type.startswith('Multi'):
                for boundary in shape.boundary:
                    ax1.plot(*numpy.asarray(boundary.coords).transpose())
            else:
                ax1.plot(*numpy.asarray(boundary.coords).transpose())
        fig1.savefig('tests/cartesian_test.png')

    def test_forwards_and_backwards(self):
    	# Get Americas as test shape
        longitude, latitude = self.shapes[2].boundary.xy

        # Carry out conversions
        theta, phi = longlat_to_spherical(longitude, latitude)
        x, y, z = spherical_to_cartesian(theta, phi)
        theta_b, phi_b = cartesian_to_spherical(x, y, z)

        # Generate plot just to check
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(theta, phi, 'r')
        ax.plot(theta_b, phi_b, 'b')
        fig.savefig('tests/testfig.png')

       	# Check that everything comes back ok
        self.assertTrue(numpy.allclose(
            theta_b, theta))
        self.assertTrue(numpy.allclose(
            phi_b, phi))


