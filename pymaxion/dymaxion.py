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
import warnings

from .conversions import spherical_to_cartesian, \
    cartesian_to_spherical, longlat_to_spherical
from .projections import gnomonic_projection, \
    sterographic_projection, inverse_sterographic_projection
from .rotations import rotation_matrix, rotate_translate
from .utilities import geodesic_linspace

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
        # 8:  None,
        8:  { 'rotation': 300, 'translation': (1.5, 1 / SQRT3) },
        9:  { 'rotation': 60,  'translation': (2.5, 1 / SQRT3)       },
        10: { 'rotation': 60,  'translation': (3.5, 1 / SQRT3)       },
        11: { 'rotation': 120, 'translation': (3.5, 2 / SQRT3)       },
        12: { 'rotation': 60,  'translation': (4,   5 / (2 * SQRT3)) },
        13: { 'rotation': 0,   'translation': (4,   7 / (2 * SQRT3)) },
        14: { 'rotation': 0,   'translation': (5,   7 / (2 * SQRT3)) },
        # 15: None,
        15: { 'rotation': 60, 'translation': (0.5, 1 / SQRT3) },
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

    def __call__(self, longitude, latitude):
        """ Converts the given latitudes and longitudes to Dymaxion points
        """
        # Convert the coordinates into cartesian coordinates, determine
        # which face the points are in & rotate the given point
        # onto the template spherical face
        theta, phi = longlat_to_spherical(longitude, latitude)
        x, y, z = spherical_to_cartesian(theta, phi)
        face_idx = self._which_face(x, y, z)
        x, y, z = numpy.dot(self.face_rotation_matrices[face_idx],
                            numpy.vstack([x, y, z]))
        rlong, rlat = cartesian_to_spherical(x, y, z)

        # Project points through gnomonic projection, then rotate back into
        # postition on the Fuller plane
        x_tmpl, y_tmpl = gnomonic_projection(numpy.pi/2, 0)(rlong, rlat)
        transf = self.transformations[face_idx]
        return rotate_translate(**transf)(x_tmpl, y_tmpl)

    def get_poly_intersection(self, polygon, face_idx):
        """ Get the intersection of a polygon with a given face
        
            This carries out the intersection using a stereographic transform
            to get around issues with wrapping in a spherical geometry
        
            Parameters:
                polygon - a shapely.geometry.Polygon instance
                face_idx - the face to get the intersection with
                
            Returns:
                intersection - a shapely.geometry.Polygon instance containing
                    the intersection
        """
        # Define transform functions
        centre = numpy.degrees(cartesian_to_spherical(*self.face_centres[face_idx]))
        transform = lambda s: shapely.ops.transform(
            sterographic_projection(centre[0], centre[1], longlat=True), s)
        itransform = lambda s: shapely.ops.transform(
            inverse_sterographic_projection(centre[0], centre[1], longlat=True), s)

        # Transform the face and the polygon
        with warnings.catch_warnings():
            # Suppress some warnings - we can end up with invalid polygons but
            # we deal with this ourselves
            warnings.simplefilter("ignore")
            face = transform(self.latlong_faces[face_idx])
            polygon_tr = transform(polygon)
            if not polygon_tr.is_valid:
                # We have a self-intersecting polygon so use the
                # .buffer(0) trick to remove the intersections
                polygon_tr = polygon_tr.buffer(0)

        # We also need to track the interior of the polygons as the projection
        # can turn them inside out. Because we're projecting through the centre
        # of the face we don't need to worry about this happening for it.
        interior_point = transform(polygon.representative_point())

        # Get intersection
        if polygon_tr.contains(interior_point):
            # We're ok - the inside and outside are in the right locations
            intersect = itransform(face.intersection(polygon_tr))
        else:
            # We need to switch from intersection to difference because 
            # the interior and exterior of our polygon have been switched 
            # by the projection
            intersect = itransform(face.difference(polygon_tr))
        
        return intersect

    def _which_face(self, x, y, z):
        """ Determine which face center is closest to the given point

            Parameters:
                x, y, z - arrays containing the cartesian coordinates of
                    the points
        """
        # Determine nearest face centre using the vertex kd tree
        points = numpy.vstack([x, y, z]).transpose()
        _, faces = self.vertex_kd_tree.query(points)
        if len(faces) == 1:
            return faces[0]
        else:
            return faces

    def _which_subregions(self, face_idx, x, y, z):
        """ Determine which subregion of a face these points fall into
        """
        return 1
        # Determine the distance to the face vertices
        points = numpy.vstack([x, y, z]).transpose()
        vertices = self.vertices[self.faces[face_idx]]
        dist = numpy.sqrt(sum((points - vertices) ** 2))

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
        subregions = numpy.zeros(len(x), dtype=int)
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
        if hasattr(self, '_rotation_matrices'):
            return self._rotation_matrices

        # Otherwise, calculate the rotation matrices using the face data
        nfaces = len(self.faces)
        self._rotation_matrices = []
        for face_idx in range(nfaces):
            # We first need to rotate all the face centers so that they all overlap
            # We'll use two rotation matrices to define this because it's easier
            # than trying to calculate the relevant Euler angles
            x, y, z = self.face_centres[face_idx]
            theta, phi = cartesian_to_spherical(x, y, z)
            rotation = numpy.dot(
                rotation_matrix([-numpy.pi/2, 0, -phi]),
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
                vertices = self.vertices[node]
                center = numpy.sum(vertices, axis=0) / 3
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
        for idx, trans in self.transformations.items():
            # Plot results
            self._fuller_faces.append(
                shapely.ops.transform(
                    rotate_translate(**trans),
                    self.template))

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
            vertices = [numpy.degrees(cartesian_to_spherical(*v))
                        for v in self.vertices[vidxs]]
            vertex_list = [(vertices[0], vertices[1]),
                           (vertices[1], vertices[2]),
                           (vertices[2], vertices[0])]
            edges = numpy.hstack([
                numpy.vstack(geodesic_linspace(v0, v1, inclusive=False))
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


PROJ = DymaxionProjection()
def dymaxion_transform(latitudes, longitudes):
    """ Transform the given latitudes and longitudes into x, y points in
        the Fuller (dymaxion) plane
    """
    return PROJ(latitudes, longitudes)
