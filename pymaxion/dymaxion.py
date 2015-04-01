#!/usr/bin/env python
""" file: dymaxion.py
    author: Jess Robertson
            CSIRO Mineral Resources Flagship
    date:   Tuesday March 31, 2015

    description: An implementation of the Dymaxion projection, stolen from
        Mike Bostok's javascript implementation, and numpyified and generally
        Pythonized for use with Shapely objects
"""

import numpy
import shapely.geometry
import descartes
import matplotlib.pyplot as plt
import scipy.spatial

from .coordinate_conversions import spherical_to_cartesian, \
    cartesian_to_spherical, latlong_to_spherical
from .rotations import rotate

# CONSTANTS
SQRT3 = numpy.sqrt(3)
SQRT5 = numpy.sqrt(5)

## DYMAXION PROJECTION CLASS
class DymaxionProjection(object):

    """ Class for implementing a Dymaxion Projection
    """

    # Cartesian coordinates for the 12 vertices of icosahedron
    # Array indices are [node_idx, axis_idx]
    vertices = numpy.array([
        [0.420152426708710003, 0.078145249402782959, 0.904082550615019298],
        [0.995009439436241649, -0.091347795276427931, 0.040147175877166645],
        [0.518836730327364437, 0.835420380378235850, 0.181331837557262454],
        [-0.414682225320335218, 0.655962405434800777, 0.630675807891475371],
        [-0.515455959944041808, -0.381716898287133011, 0.767200992517747538],
        [0.355781402532944713, -0.843580002466178147, 0.402234226602925571],
        [0.414682225320335218, -0.655962405434800777, -0.630675807891475371],
        [0.515455959944041808, 0.381716898287133011, -0.767200992517747538],
        [-0.355781402532944713, 0.843580002466178147, -0.402234226602925571],
        [-0.995009439436241649, 0.091347795276427931, -0.040147175877166645],
        [-0.518836730327364437, -0.835420380378235850, -0.181331837557262454],
        [-0.420152426708710003, -0.078145249402782959, -0.904082550615019298]
    ])

    # Vertices defining each face, each row gives the three vertices of the
    # face
    faces = numpy.array([
        [ 0,  1,  2], [ 0,  2,  3], [ 0,  3,  4], [ 0,  4,  5], [ 0,  1,  5],
        [ 1,  2,  7], [ 7,  2,  8], [ 8,  2,  3], [ 9,  8,  3], [ 4,  9,  3],
        [ 4, 10,  9], [ 4,  5, 10], [10,  5,  6], [ 6,  5,  1], [ 7,  6,  1],
        [11,  8,  7], [11,  8,  9], [11, 10,  9], [11, 10,  6], [11,  7,  6]],
        dtype=numpy.int)

    # Here's the list of rotations and scalings to move a face into
    # position on a 2D plane. Note that faces 8 and 15 have bits
    # moved around and need to be handled seperately
    transformations = {
        0:  { 'rotation': 240, 'translation': (2,   7 / (2 * SQRT3)) },
        1:  { 'rotation': 300, 'translation': (2,   5 / (2 * SQRT3)) },
        2:  { 'rotation': 0,   'translation': (2.5, 2 / SQRT3)       },
        3:  { 'rotation': 60,  'translation': (3,   5 / (2 * SQRT3)) },
        4:  { 'rotation': 180, 'translation': (2.5, 4 * SQRT3 / 3)   },
        5:  { 'rotation': 300, 'translation': (1.5, 4 * SQRT3 / 3)   },
        6:  { 'rotation': 300, 'translation': (1,   5 / (2 * SQRT3)) },
        7:  { 'rotation': 0,   'translation': (1.5, 2 / SQRT3)       },
        8:  None,
        9:  { 'rotation': 60,  'translation': (2.5, 1 / SQRT3)       },
        10: { 'rotation': 60,  'translation': (3.5, 1 / SQRT3)       },
        11: { 'rotation': 120, 'translation': (3.5, 2 / SQRT3)       },
        12: { 'rotation': 60,  'translation': (4,   5 / (2 * SQRT3)) },
        13: { 'rotation': 0,   'translation': (4,   7 / (2 * SQRT3)) },
        14: { 'rotation': 0,   'translation': (5,   7 / (2 * SQRT3)) },
        15: None,
        16: { 'rotation': 0,   'translation': (1,   1 / (2 * SQRT3)) },
        17: { 'rotation': 120, 'translation': (4,   1 / (2 * SQRT3)) },
        18: { 'rotation': 120, 'translation': (4.5, 2 / (SQRT3))     },
        19: { 'rotation': 300, 'translation': (5,   5 / (2 * SQRT3)) },
    }

    # These have split faces so need to handled differently
    conditional_transformations = {
        8: {
            'condition': lambda pt, face: \
                self._which_subregion(pt, face) < 2,
            True:  { 'rotation': 300, 'translation': (1.5, 1 / SQRT3) },
            False: { 'rotation': 0, 'translation': (2, 1 / (2 * SQRT3)) }
        },
        15: {
            'condition': lambda pt, face: \
                self._which_subregion(pt, face) < 4,
            True: { 'rotation': 60, 'translation': (0.5, 1 / SQRT3) },
            False: { 'rotation': 0, 'translation': (5.5, 2 / SQRT3) }
        }
    }

    def __init__(self):
        # We need to cache the node centers on initialization
        # We also need to cache the vertex array in a KD tree for
        # nearest-neighbour lookup later
        self.vertex_kd_tree = scipy.spatial.cKDTree(self.face_centres)

    def __call__(self, latitudes, longitudes):
        """ Converts the given latitudes and longitudes to Dymaxion points
        """
        # Convert the coordinates into cartesian coordinates
        points = spherical_to_cartesian(
                     latlong_to_spherical(
                         numpy.vstack([latitudes, longitudes])))

        # Determine which face the points are in
        faces = self._which_face(points)

        # Convert points to dymaxion locations
        transformed = numpy.empty((2, len(faces)))
        for face in range(len(self.faces)):
            mask = (faces == face)
            transformed[:, mask] = self._dymax(face, points[:, mask])
        return transformed

    def _dymax(self, face, points):
        """ Construct projected location given a face, a set of cartesian
            points in that face, and subregions for those points.
        """
        # In order to rotate the given point into the template spherical
        # face, we need the spherical polar coordinates of the center
        # of the face and one of the face vertices.
        h0 = face_vertex = self.vertices[self.faces[face][0]]
        face_centre = self.face_centres[face].transpose()
        theta, phi = cartesian_to_spherical(face_centre)

        # Rotate the point onto the template face

        # Project the point onto the template face using gnomonic projection

        # Rotate and scale points back onto 2D plane
        def _make_transform(points, rotation, translation, **kwargs):
            points = rotate(points, rotation) + translation
        transform = self.transformations[face]
        if transform:
            _make_transform(points, **transform)
        else:
            transform = self.conditional_transformations[face]
            flags = transform['condition'](points, face)
            flag_unique = set(transform.keys()) - 'condition'
            for flag in flag_unique:
                _make_transform(points[flags == flag], **transform[flag])
        return points

    def _which_face(self, points):
        """ Determine which face center is closest to the given point

            Parameters:
                points - a 3 by N array containing the cartesian coordinates of
                    the points
        """
        # Determine nearest face centre using the vertex kd tree
        _, faces = self.vertex_kd_tree.query(points.transpose())
        return faces

    def _which_subregions(self, points, faces):
        """ Determine which subregion of a face these points fall into
        """
        # Determine the distance to the face vertices
        if len(faces) == 1:
            faces = [faces]
        dist = numpy.sqrt([
            ((points[:, i] - self.vertices[self.faces[t]]) ** 2).sum(axis=-1)
            for i, t in enumerate(faces)]).transpose()

        # We also need to know about subregions since some of the faces
        # are cut up into smaller sections
        # Here we label subregion regions based on which segment we're in
        # and create a mask over points with the given ordering of distances
        label_and_order = {
            1: (0, 1, 2), 2: (0, 2, 1), 3: (1, 0, 2),
            4: (1, 2, 0), 5: (2, 0, 1), 6: (2, 1, 0)
        }
        mask = lambda o: numpy.logical_and(dist[o[0]] <= dist[o[1]],
                                           dist[o[1]] <= dist[o[2]])

        # Figure out which subregion we're in based on distance to face
        # vertices
        subregions = numpy.zeros(len(faces), dtype=int)
        for label, order in label_and_order.items():
            subregions[mask(order)] = label

    @property
    def face_centres(self):
        """ Return the centre of each face
        """
        if hasattr(self, '_face_centres'):
            return self._face_centres
        else:
            self._face_centres = numpy.empty((self.faces.shape))
            for idx, node in enumerate(self.faces):
                center = numpy.sum(self.vertices[node], axis=0) / 3
                magnitude = numpy.sqrt(numpy.sum(center ** 2))
                self._face_centres[idx] = center / magnitude
            return self._face_centres

    def plot_polygon(self, axes=None):
        """ Plots the polygon used for the Dymaxion projection

            This is really just to check that I've got my vertex ordering
            all sorted
        """
        axes = axes or plt.gca()
        cmap = plt.get_cmap('coolwarm')
        for idx, face in enumerate(self.faces):
            color = cmap(idx / len(self.faces))
            shape = shapely.geometry.Polygon(self.vertices[face][:, :2])
            axes.add_patch(descartes.PolygonPatch(
                shape, alpha=0.5, linewidth=2,
                facecolor=color, edgecolor='black'))
            face_axis = numpy.vstack(([0, 0], self.face_centres[idx][:2])).T
            axes.plot(*face_axis,
                      marker='o', linewidth=3, color=color,
                      markerfacecolor=color, markeredgecolor=color,
                      alpha=0.4)
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)
        axes.set_aspect("equal")
        axes.set_axis_off()

    def plot_unfolded(self, axes=None):
        """ Plots the unfolded polygon
        """
        # Set up & plot initial shape
        sqrt3 = numpy.sqrt(3)
        shape = shapely.geometry.Polygon([[ 0,    1/SQRT3     ],
                                          [ 1/2, -1/(2*SQRT3) ],
                                          [-1/2, -1/(2*SQRT3) ]])
        axes = axes or plt.gca()
        axes.add_patch(descartes.PolygonPatch(
                shape, facecolor='white', edgecolor='black', alpha=0.7))
        axes.plot(*shape.centroid.xy, color='red', marker='o')

        # Generate translated polygons for map
        points = numpy.vstack(shape.boundary.xy)
        cmap = plt.get_cmap('coolwarm')
        for idx, transform in self.transformations.items():
            # Get conditional transform
            if not transform:
                transform = self.conditional_transformations[idx][True]

            # Plot results
            color = cmap(idx / len(self.transformations))
            poly = shapely.geometry.Polygon(
                rotate(points, transform['rotation']).transpose()
                + transform['translation'])
            axes.add_patch(descartes.PolygonPatch(
                poly, facecolor=color, edgecolor='black', alpha=0.7))
            axes.text(poly.centroid.x, poly.centroid.y, s=idx)

        # Fix up axes
        axes.set_xlim(-1, 6)
        axes.set_ylim(-1, 3)
        axes.set_aspect('equal')
        axes.set_axis_off()
