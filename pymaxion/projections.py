from numpy import sin, cos, sqrt, arctan2, arcsin, degrees, pi

from .conversions import longlat_to_spherical, spherical_to_longlat

def sterographic_projection(clong, clat, longlat=False):
    """ Transform function for points using a stereographic projection
        centred on the given centre

        Arguments:
            clong, clat - the centre point of the projection - this is
                where the gnominic plane touches the surface of the
                earth. Given as a tuple of longitude and latitude in degrees
            longlat - whether everythign is given in terms of degrees or not.
                Optional, defaults to no (i.e. angles given in radians)

        Returns:
            a function to transform a point into the projective plane
    """
    # Determine whether we have to transform to radians or not
    if longlat:
        theta_c, phi_c = longlat_to_spherical(clong, clat)
    else:
        theta_c, phi_c = clong, clat

    # Construct transform function
    def _transform_fn(longs, lats):
        if longlat:
            theta, phi = longlat_to_spherical(longs, lats)
        else:
            theta, phi = longs, lats

        # Construct projective point
        scale = 2 / (1 + sin(phi_c) * sin(phi)
                       + cos(phi_c) * cos(phi) * cos(theta - theta_c))
        x = scale * cos(phi) * sin(theta - theta_c)
        y = scale * (cos(phi_c) * sin(phi)
                     - sin(phi_c) * cos(phi) * cos(theta - theta_c))
        return x, y

    # Return function for use in shapely.ops.transform
    return _transform_fn

def inverse_sterographic_projection(clong, clat, longlat=False):
    """ Inverse transform function for points using a stereographic projection
        centred on the given centre

        Arguments:
            clong, clat - the centre point of the projection - this is
                where the gnominic plane touches the surface of the
                earth. Given as a tuple of longitude and latitude in degrees
            longlat - whether everythign is given in terms of degrees or not.
                Optional, defaults to no (i.e. angles given in radians)

        Returns:
            a function to transform a point into the projective plane
    """
    # Determine whether we have to transform to radians or not
    if longlat:
        theta_c, phi_c = longlat_to_spherical(clong, clat)
    else:
        theta_c, phi_c = clong, clat

    # Construct transform function
    def _transform_fn(x, y):
        # Construct projective point
        radius = sqrt(x ** 2 + y ** 2)
        C = 2 * arctan2(radius, 2)
        theta = theta_c + arctan2(x * sin(C),
                                  radius * cos(phi_c) * cos(C)
                                  - y * sin(phi_c) * sin(C))
        if theta < - pi / 2 and clong > 0:
            theta += 2 * pi
        if theta > pi / 2 and clong < 0:
            theta -= 2 * pi
        phi = arcsin(cos(C) * sin(phi_c)
                     + (y / radius) * (sin(C) * cos(phi_c)))
        if longlat:
            return spherical_to_longlat(theta, phi)
        else:
            return theta, phi

    # Return function for use in shapely.ops.transform
    return _transform_fn


def gnomonic_projection(clong, clat, longlat=False):
    """ Transform points using a gnomonic projection with centre point centre.

        Arguments:
            clong, clat - the centre point of the projection - this is
                where the gnominic plane touches the surface of the
                earth. Given as a tuple of longitude and latitude in degrees
            longlat - whether everythign is given in terms of degrees or not.
                Optional, defaults to no (i.e. angles given in radians)

        Returns:
            a function to transform a point into the projective plane
    """
    # Determine whether we have to transform to radians or not
    if longlat:
        theta_c, phi_c = longlat_to_spherical(clong, clat)
    else:
        theta_c, phi_c = clong, clat

    # Construct transform function
    def _transform_fn(longs, lats):
        if longlat:
            theta, phi = longlat_to_spherical(longs, lats)
        else:
            theta, phi = longs, lats

        # Construct projective point
        scale = 1 / (sin(phi_c) * sin(phi)
                     + cos(phi_c) * cos(phi) * cos(theta - theta_c))
        x = scale * cos(phi) * sin(theta - theta_c)
        y = scale * (cos(phi_c) * sin(phi)
                     - sin(phi_c) * cos(phi) * cos(theta - theta_c))
        return x, y

    # Return function for use in shapely.ops.transform
    return _transform_fn
