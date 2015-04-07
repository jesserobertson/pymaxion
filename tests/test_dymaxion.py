#!/usr/bin/env python

import unittest
import numpy
import matplotlib.pyplot as plt
import shapely, shapely.ops

from pymaxion import DymaxionProjection, dymaxion_transform


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

    def test_plotting_grid(self):
        """ Check we can plot a dymaxion grid
        """
        fig = plt.figure()
        proj = DymaxionProjection()
        proj.plot_grid()
        fig.savefig('tests/grid.png')

    def test_transform(self):
        """ Test that transformation works
        """
        # Untransformed points
        plt.figure()
        npoints = 1000
        points = numpy.vstack([
            numpy.random.uniform(0, 180, size=npoints),
            numpy.random.uniform(-180, 180, size=npoints)]).transpose()
        line = shapely.geometry.LineString(points)
        axes = plt.subplot(2, 1, 1)
        axes.plot(*line.xy, marker='.')
        axes.set_aspect('equal')

        # Transformed points
        transformed = shapely.ops.transform(dymaxion_transform,
                                            line)
        axes = plt.subplot(2, 1, 2)
        plt.plot(*transformed.xy, marker='.')
        plt.savefig('tests/transformed.png')
