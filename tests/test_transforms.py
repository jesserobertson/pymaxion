#!/usr/bin/env python

import unittest

from pymaxion import DymaxionProjection
from pymaxion.utilities import get_land

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
		self.shapes = shapely.geometry.MultiPolygon([land[int(i)] for i in area_idx[-1:-7:-1]])

	def test_cartesian_to_spherical(self):
		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111, projection='3d')
		ax1.set_zlim(-1, 1)
		fig2 = plt.figure()
		ax2 = fig2.gca()

		for boundary in self.shapes.boundary:
		    points = numpy.radians(boundary.coords).transpose()
		    points = spherical_to_cartesian(points)
		    ax1.plot(*points)
		    points = cartesian_to_spherical(points)
		    ax2.plot(*points)
		    assert(numpy.allclose(numpy.radians(boundary.coords).transpose(),
		                   points))

		fig1.savefig('cartesian_test.png')
		fig2.savefig('conversion_test.png')