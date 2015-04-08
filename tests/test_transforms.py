#!/usr/bin/env python

import unittest

from pymaxion import DymaxionProjection
from pymaxion.utilities import *
from pymaxion.conversions import *
from pymaxion.operations import * 
from pymaxion.plotting import *

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import fiona
import shapely, shapely.geometry
import descartes

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
        shape = self.shapes[2]

        # Carry out conversions
        theta, phi = shapely.ops.transform(longlat_to_spherical,
                                          shape).boundary.xy
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

    def test_spherical_plots_mpl3d(self):
        pass

    def test_plot_face_locations(self):
        fig = plt.figure(figsize=(21, 11))
        axes = plt.gca()
        proj = DymaxionProjection()
        face_idxs = (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        land = get_land()
        area_idx = numpy.argsort([l.area for l in land])

        for shape in self.shapes:
            if shape.type.startswith('Multi'):
                for poly in shape:
                    axes.add_patch(descartes.PolygonPatch(
                        poly, facecolor='gray', edgecolor='none', alpha=0.25))
            else:
                axes.add_patch(descartes.PolygonPatch(
                        shape, facecolor='gray', edgecolor='none', alpha=0.25))

        colorlist = alternate_colors('RdYlBu_r', len(face_idxs))
        for face_idx in face_idxs:
            face = fix_longitude(proj.latlong_faces[face_idx])
            color = colorlist[face_idx]
            axes.add_patch(descartes.PolygonPatch(
                    face, facecolor=color, edgecolor='none', alpha=0.4))
            x, y = face.centroid.xy
            axes.text(x[0], y[0], 
                      s='Face {0}'.format(face_idx), color=color)
                    
        axes.set_xlim(-220, 220)
        axes.set_ylim(-100, 100)
        axes.set_aspect('equal')
        axes.set_axis_off()
        fig.savefig('tests/locations.png')

    def test_subpoly(self):
        fig = plt.figure(figsize=(21, 11))
        axes = plt.gca()
        face_idxs = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18)
        problems = (8, 13)
        colorlist = alternate_colors('RdYlBu_r', 19)
        proj = DymaxionProjection()
        shape = self.shapes[0]

        for face_idx in face_idxs:
            # Define transform functions
            centre = numpy.degrees(cartesian_to_spherical(*proj.face_centres[face_idx]))
            transform = lambda s: shapely.ops.transform(
                sterographic_projection(centre[0], centre[1], longlat=True), s)
            itransform = lambda s: shapely.ops.transform(
                inverse_sterographic_projection(centre[0], centre[1], longlat=True), s)

            # Get intersection of poly with face
            intersect = proj.get_poly_intersection(shape, face_idx)
            if intersect.is_empty:
                continue
            else:
                intersect = fix_longitude(intersect)

            # Plot part of poly
            color = colorlist[face_idx]
            if intersect.type.startswith('Multi'):
                for poly in intersect:
                    axes.add_patch(descartes.PolygonPatch(
                        poly, facecolor=color, edgecolor='none', alpha=0.8))
            else:
                axes.add_patch(descartes.PolygonPatch(
                        intersect, facecolor=color, edgecolor='none', alpha=0.8))

        set_bounds(shape.bounds)
        axes.set_axis_off()
        fig.savefig('tests/subpoly.png')

