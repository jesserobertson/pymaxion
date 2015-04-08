#!/usr/bin/env python

import unittest

from pymaxion import *
from pymaxion.utilities import *
from pymaxion.conversions import *
from pymaxion.operations import * 
from pymaxion.plotting import *

import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TestRotations(unittest.TestCase):

    def setUp(self):
        bbox = (-180, -90, 180, 90)
        bbox_shape = shapely.geometry.box(*bbox)
        self.graticules = make_grid(bbox=bbox, npoints=200, graticule_spacing=20)
        land = get_land()
        area_idx = numpy.argsort([l.area for l in land])
        self.shapes = shapely.geometry.MultiPolygon(
            [land[int(i)] for i in area_idx[-1:-20:-1]])

    def test_face_rotations(self):
        """ Check that the face rotations work ok
        """
        fig1 = plt.figure(figsize=(11,11))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.set_zlim(-1, 1)
        fig2 = plt.figure(figsize=(11,11))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.set_zlim(-1, 1)

        proj = DymaxionProjection()
        nfaces = len(proj.faces)
        cmap = plt.get_cmap('coolwarm')
        final_faces = []
        for face_idx in range(nfaces):
            color = cmap(face_idx / nfaces)
            rotation = proj.face_rotation_matrices[face_idx]
            transform = lambda pt: numpy.dot(rotation, pt)

            # Now we can plot these up
            vertex = proj.vertices[proj.faces[face_idx][0]] 
            center = proj.face_centres[face_idx]
            face_points = proj.vertices[proj.faces[face_idx]]
            face_points = numpy.vstack([face_points, face_points[0]]).transpose()
            ax1.plot(*center[numpy.newaxis].transpose(), color=color, marker='o')
            ax1.plot(*face_points, color=color, linewidth=2)
            ax1.plot(*vertex[numpy.newaxis].transpose(), color=color, marker='s')
            center = transform(center)
            vertex = transform(vertex)
            face_points = transform(face_points)
            ax2.plot(*center[numpy.newaxis].transpose(), color=color, marker='o')
            ax2.plot(*face_points, color=color, linewidth=2)
            ax2.plot(*vertex[numpy.newaxis].transpose(), color=color, marker='s')
            final_faces.append(face_points)

        # Check that we have all our values in the right place
        for a, b in itertools.combinations(final_faces, 2):
            # We can have two possible orderings for the faces (cw or ccw)
            # so we check both here
            assert(numpy.allclose(a, b) 
               or numpy.allclose(a, b[:, (0, 2, 1, 3)]))
            
        gfmt = dict(color='gray', dashes=(4, 2), alpha=0.5)
        sfmt = dict(color='black', linewidth=2)
        for collection, fmt in [(self.graticules, gfmt), (self.shapes.boundary, sfmt)]:
            for line in collection:
                points = numpy.radians(line.coords).transpose()
                points = spherical_to_cartesian(*points)
                ax1.plot(*points, **fmt)
                ax2.plot(*points, **fmt)
        fig1.savefig('tests/initial_faces.png')
        fig2.savefig('tests/final_faces_after_rotation.png')