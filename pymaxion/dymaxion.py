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

from .conversions import spherical_to_cartesian, \
    cartesian_to_spherical, longlat_to_spherical, gnomonic_projection
from .rotations import rotate

# CONSTANTS
SQRT3 = numpy.sqrt(3)
SQRT5 = numpy.sqrt(5)

def geodesic_linspace(a, b, npoints=50):
    """ Construct a 'linspace' along a geodesic for a spherical earth between two points
    """
    # Make sure that everything is set up right
    long_a, lat_a = numpy.radians(a)
    long_b, lat_b = numpy.radians(b)
    fraction = numpy.linspace(0, 1, npoints)
    
    # Parameterize arc between the points
    d = numpy.arccos(numpy.sin(lat_a) * numpy.sin(lat_b) + numpy.cos(lat_a) * numpy.cos(lat_b) * numpy.cos(long_a - long_b))
    A = numpy.sin((1 - fraction) * d) / numpy.sin(d)
    B = numpy.sin(fraction * d) / numpy.sin(d)

    # Calculate cartesian points
    points = \
        numpy.vstack([A * numpy.cos(lat_a) * numpy.cos(long_a) 
                        + B * numpy.cos(lat_b) * numpy.cos(long_b), 
                      A * numpy.cos(lat_a) * numpy.sin(long_a) 
                        + B * numpy.cos(lat_b) * numpy.sin(long_b), 
                      A * numpy.sin(lat_a) + B * numpy.sin(lat_b)])

    # Convert back to latitude and longitude and return
    return numpy.degrees(cartesian_to_spherical(points))

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

    # This is the template shape for the faces
    template = shapely.geometry.Polygon([[ 0,    1/SQRT3     ],
                                         [ 1/2, -1/(2*SQRT3) ],
                                         [-1/2, -1/(2*SQRT3) ]])

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
            'condition': lambda self, pt, face: \
                self._which_subregions(pt, face) < 2,
            True:  { 'rotation': 300, 'translation': (1.5, 1 / SQRT3) },
            False: { 'rotation': 0, 'translation': (2, 1 / (2 * SQRT3)) }
        },
        15: {
            'condition': lambda self, pt, face: \
                self._which_subregions(pt, face) < 4,
            True: { 'rotation': 60, 'translation': (0.5, 1 / SQRT3) },
            False: { 'rotation': 0, 'translation': (5.5, 2 / SQRT3) }
        }
    }

    def __init__(self):
        # We need to cache the node centers on initialization
        # We also need to cache the vertex array in a KD tree for
        # nearest-neighbour lookup later
        self.vertex_kd_tree = scipy.spatial.cKDTree(self.face_centres)

    def __call__(self, longitudes, latitudes):
        """ Converts the given latitudes and longitudes to Dymaxion points
        """
        # Convert the coordinates into cartesian coordinates
        theta, phi = longlat_to_spherical(longitudes, latitudes)
        x, y, z = spherical_to_cartesian(theta, phi)

        # Determine which face the points are in
        faces = self._which_face(x, y, z)

        # Plot this result
        reds = [plt.get_cmap('Reds')(3 * face / len(self.faces)) 
                for face in range(len(self.faces))]
        blues = [plt.get_cmap('Blues')(3 * face / len(self.faces)) 
                for face in range(len(self.faces))]
        greens = [plt.get_cmap('Greens')(3 * face / len(self.faces)) 
                for face in range(len(self.faces))]
        colors = []
        for rgb in zip(reds, blues, greens):
            colors.extend(rgb)
        plt.figure()
        axes = plt.gca()
        for face in self.transformations.keys():
            axes.plot(longitudes[faces == face],
                      latitudes[faces == face],
                      marker='.', color=colors[face],
                      linewidth=0)

        # Convert points to dymaxion locations
        plt.figure()
        axes = plt.gca()
        dymax_points = numpy.empty((2, len(points)))
        for face in range(len(self.faces)):
            mask = (faces == face)
            result = self._dymax(face, points[:, mask])
            axes.plot(result[:, 0], result[:, 1], 
                      color=colors[face], linewidth=0, marker='.')
        axes.set_xlim(0, 6)
        axes.set_ylim(0, 4)
        plt.show()
        return dymax_points

    def _dymax(self, face, points):
        """ Construct projected location given a face, a set of cartesian
            points in that face, and subregions for those points.
        """
        # In order to rotate the given point into the template spherical
        # face, we need the spherical polar coordinates of the center
        # of the face and one of the face vertices.
        h0 = face_vertex = self.vertices[self.faces[face][0]]
        face_centre = self.face_centres[face].transpose()

        # Project the point onto the template face using gnomonic projection
        proj_points = project(cartesian_to_spherical(face_centre), 
                              cartesian_to_spherical(points))
        print(proj_points.shape, points.shape)

        # Rotate and scale points back onto 2D plane
        trans_info = self.transformations[face]
        if trans_info:
            proj_points = (rotate(proj_points, trans_info['rotation']).transpose() 
                           + trans_info['translation'])
        else:
            proj_points = numpy.zeros_like(proj_points.transpose())
            # # We're doing something funky here, get out which transform should be used
            # # for which point
            # flags = self.conditional_transformations[face]['condition'](self, points, [face])

            # # Loop through flags, transform relevant points
            # flag_unique = [k for k in self.conditional_transformations[face].keys() 
            #                if k != 'condition']
            # for flag in flag_unique:
            #     trans_info = self.conditional_transformations[face][flag]
            #     proj_points = (rotate(proj_points, trans_info['rotation'])
            #                    + trans_info['translation'])
        return proj_points

    def _which_face(self, points):
        """ Determine which face center is closest to the given point

            Parameters:
                points - a 2 by N array containing the spherical coordinates of
                    the points
        """
        # Determine nearest face centre using the vertex kd tree
        c_points = spherical_to_cartesian(points)
        _, faces = self.vertex_kd_tree.query(c_points.transpose())
        return faces

    def _which_subregions(self, points, faces):
        """ Determine which subregion of a face these points fall into
        """
        # Determine the distance to the face vertices
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
        return subregions

    @property
    def face_rotation_matrices(self):
        """ Constructs matrices to rotate all the faces and the data
            into a standard position (vertically oriented, with the centre point
            on the x-axis).
            
            This means we can define the later translation and rotation to form
            the net from a single starting orientation
        """
        # Return the cached version if we have it
        if hasattr(self, _rotation_matrices):
            return self._rotation_matrices

        # Otherwise, calculate the rotation matrices using the face data
        nfaces = len(self.faces)
        self._rotation_matrices = []
        for face_idx in range(nfaces):
            # We first need to rotate all the face centers so that they all overlap
            # We'll use two rotation matrices to define this because it's easier
            # than trying to calculate the relevant Euler angles
            centre = self.face_centres[face_idx]
            theta, phi = cartesian_to_spherical(centre).ravel()
            rotation = numpy.dot(
                rotation_matrix([-pi/2, 0, -phi]), 
                rotation_matrix([theta, 0, 0]))

            # We also need to rotate the faces so they are all aligned the same way
            # So we rotate a single vertex to sit directly above the face centre 
            vertex = numpy.dot(rotation, 
                               self.vertices[self.faces[face_idx][0]])
            angle = numpy.arctan2(vertex[1], vertex[2])
            self._rotation_matrices.append(
                numpy.dot(rotation_matrix([0, 0, angle]), 
                          rotation))

        return self._rotation_matrices

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

    @property
    def fuller_faces(self):
        """ Return the polygon faces as Shapely Polygons in the Fuller plane
        """
        # Return cached value if already calculated
        if hasattr(self, '_fuller_faces'):
            return self._fuller_faces

        # Generate translated polygons for map
        self._fuller_faces = []
        points = numpy.vstack(self.template.boundary.xy)
        for idx, trans_info in self.transformations.items():
            # Get conditional transform
            if not trans_info:
                trans_info = self.conditional_transformations[idx]
                first_key = list(trans_info.keys())[1]
                trans_info = trans_info[first_key]
            rotation = trans_info['rotation']
            translation = trans_info['translation']

            # Plot results
            self._fuller_faces.append(
                shapely.geometry.Polygon(
                    rotate(points, rotation).transpose() + translation))

        return self._fuller_faces

    @property
    def latlong_faces(self):
        """ Return the polygon faces as Shapely Polygons in the lat-long plane
        """
        # Return cached value if already calculated
        if hasattr(self, '_latlong_faces'):
            return self._latlong_faces

        # Construct polygons for faces by using geodesic arcs between vertices
        self._latlong_faces = []
        for idx, vidxs in enumerate(self.faces):
            vertices = [numpy.degrees(cartesian_to_spherical(v)) 
                        for v in self.vertices[vidxs]]
            vertex_list = [(vertices[0], vertices[1]), 
                           (vertices[1], vertices[2]), 
                           (vertices[2], vertices[0])]
            edges = numpy.hstack([
                numpy.vstack(geodesic_linspace(v0, v1))
                for v0, v1 in vertex_list]).transpose()
            self._latlong_faces.append(shapely.geometry.Polygon(edges))
        return self._latlong_faces

    def plot_unfolded(self, axes=None):
        """ Plots the unfolded polygon
        """
        # Set up & plot initial shape
        axes = axes or plt.gca()
        axes.add_patch(descartes.PolygonPatch(
                self.template, facecolor='white', edgecolor='black', alpha=0.7))
        axes.plot(*self.template.centroid.xy, color='red', marker='o')

        # Plot translated polygons for map
        cmap = plt.get_cmap('coolwarm')
        for idx, poly in enumerate(self.fuller_faces):
            color = cmap(idx / len(self.transformations))
            axes.add_patch(descartes.PolygonPatch(
                poly, facecolor=color, edgecolor='black', alpha=0.7))
            axes.text(poly.centroid.x, poly.centroid.y, s=idx)

        # Fix up axes
        axes.set_xlim(-1, 6)
        axes.set_ylim(-1, 3)
        axes.set_aspect('equal')
        axes.set_axis_off()

    def plot_grid(self, axes=None, **kwargs):
        """ Plot the background grid for a dymaxion map
        """
        axes = axes or plt.gca()
        format_dict = dict(
            facecolor='white', edgecolor='gray', linewidth=1,
            alpha=1, zorder=0)
        format_dict.update(kwargs)
        for poly in self.fuller_faces:
            axes.add_patch(descartes.PolygonPatch(poly, **format_dict))
    
        # Fix up axes
        axes.set_xlim(-1, 6)
        axes.set_ylim(-1, 3)
        axes.set_aspect('equal')
        axes.set_axis_off()

