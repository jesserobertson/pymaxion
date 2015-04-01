#!/usr/bin/env python

import unittest

from pymaxion import DymaxionProjection

import numpy
import matplotlib.pyplot as plt

class TestDymaxion(unittest.TestCase):

    def test_plotting_polygon(self):
        """ Check that we can plot the polygon used for DymaxionProjection
        """
        fig = plt.figure()
        proj = DymaxionProjection()
        proj.plot_polygon()
        fig.savefig('tests/poly.png')

    def test_plotting_unfold(self):
        """ Check that we can plot the polygon used for DymaxionProjection
        """
        fig = plt.figure()
        proj = DymaxionProjection()
        proj.plot_unfolded()
        fig.savefig('tests/poly_unfolded.png')

    def test_transform(self):
        """ Test that transformation works
        """
        proj = DymaxionProjection()

        # Untransformed points
        plt.figure()
        npoints = 10000
        latitudes = numpy.random.uniform(0, 180, size=npoints)
        longitudes = numpy.random.uniform(-180, 180, size=npoints)
        axes = plt.subplot(2, 1, 1)
        axes.plot(longitudes, latitudes, '.')
        axes.set_aspect('equal')

        # Transformed points
        transformed = proj(latitudes=latitudes, longitudes=longitudes)
        axes = plt.subplot(2, 1, 2)
        plt.plot(*transformed, marker='.')
        plt.savefig('tests/transformed.png')
