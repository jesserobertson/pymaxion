#!/usr/bin/env python
""" file:  rotation.py
    author: Jess Robertson
            CSIRO Mineral Resources Flagship
    date:   Wednesday April 01, 2015

    description: rotation functions
"""

import numpy
import scipy.sparse

def rotation_matrix(angles_list=None, angles_array=None):
    """ Returns a rotation matrix in n dimensions

        The combined rotation array is build up by left-multiplying the
        preexisting rotation array by the rotation around a given axis.
        For a $d$-dimensional array, this is given by:

        $$ C(\theta) = R(\theta_{d, d-1})R(\theta_{d, d-2})\times\ldots\times
            R(\theta_{i, j})\times\ldots R(\theta_{1, 2})$$

        where $i$ and $j$ are positive integers ranging from 1 to $d$, and
        satisfy $i \leq j$.
    """
    # Check inputs
    if angles_list is not None and angles_array is not None:
        raise ValueError('You should only supply one of the angles_list'
                         ' or angles_array arguments to rotation_matrix')

    elif angles_list is not None:
        # Make sure that we have the right number of angles supplied,
        # guess the dimension required
        dimension_estimate = int(1 + numpy.sqrt(1 + 8 * len(angles_list))) // 2
        checks = [int(dimension_estimate) - 1, int(dimension_estimate)]
        allowed_angles = map(lambda d: d * (d - 1) / 2, checks)
        if len(angles_list) not in allowed_angles:
            err_string = (
                'Wrong number of angles ({0}) supplied to rotation_matrix - '
                'you should specify d*(d-1)/2 angles for a d-dimensional '
                'rotation matrix (i.e. {1[0]} angles for d={2[0]} or {1[1]} '
                'angles for d={2[1]})'
            ).format(len(angles_list), checks, allowed_angles)
            raise ValueError(err_string)
        else:
            dim = dimension_estimate

        # Generate angles array from list
        angles_array = numpy.zeros((dim, dim))
        angles_gen = (a for a in angles_list)
        for idx, _ in numpy.ndenumerate(angles_array):
            if idx[0] > idx[1]:
                angles_array[idx] = next(angles_gen)

    elif angles_array is not None:
        angles_array = numpy.asarray(angles_array)
        dim = angles_array.shape[0]

    # Generate rotation matrix
    identity = scipy.sparse.identity(dim, format='lil')
    combined = identity.copy()
    for idx, angle in numpy.ndenumerate(angles_array):
        # Make sure we're on the lower-diagonal part of the angles array
        if idx[0] <= idx[1]:
            continue

        # Build non-zero elements of rotation matrix using Givens rotations
        # see: https://en.wikipedia.org/wiki/Givens_rotation
        rotation = identity.copy()
        rotation[idx[0], idx[0]] = numpy.cos(angle)
        rotation[idx[1], idx[1]] = numpy.cos(angle)
        rotation[idx[0], idx[1]] = numpy.sin(angle)
        rotation[idx[1], idx[0]] = -numpy.sin(angle)

        # Build combined rotation matrix
        combined = combined.dot(rotation)

    return numpy.asarray(combined.todense())


def rotate(points, angle_x, angle_y=None, angle_z=None):
    """ Rotate points through some angle about the given axes

        Parameters:
            points - 2 or 3 by N array of points to rotate
            angle_x, angle_y, angle_z - rotation angle about x/y/z axis in
                degrees. y and z rotations are optional, should only be
                specified for three-dimensional points.
    """
    if points.shape[0] == 3:
        if any(a is None for a in (angle_y, angle_z)):
            raise ValueError('You have to specify three rotations for 3D data')
        else:
            angles = numpy.radians([angle_z, angle_y, angle_x])
            return numpy.dot(points.T, rotation_matrix(angles).T).T

    elif points.shape[0] == 2:
        angle = numpy.radians([angle_x])
        return numpy.dot(points.T, rotation_matrix(angle).T).T


def test():
    import matplotlib.pyplot as plt

    points = numpy.array([[1, 0], [0, 0], [0, 1]]).transpose()
    rotpoints = rotate(points, 45)
    print(type(rotpoints))
    axes = plt.gca()
    axes.plot(points[0], points[1], color='red')
    axes.plot(rotpoints[0], rotpoints[1], color='blue')
    axes.set_xlim(-1.1, 1.1)
    axes.set_ylim(-1.1, 1.1)
    axes.set_aspect("equal")
    plt.show()

if __name__ == '__main__':
    test()
