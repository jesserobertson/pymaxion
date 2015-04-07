from numpy import sin, cos

from .conversions import longlat_to_spherical

def sterographic_projection(central_longitude, central_latitude):
    """ Transform function for points using a stereographic projection
        centred on the given centre
    """
    theta_c, phi_c = longlat_to_spherical(central_longitude, central_latitude)
    def _transform_fn(longitude, latitude):
        theta, phi = longlat_to_spherical(longitude, latitude)
        scale = 2 / (1 + sin(phi_c) * sin(phi)
                       + cos(phi_c) * cos(phi) * cos(theta - theta_c))
        x = scale * cos(phi) * sin(theta - theta_c)
        y = scale * (cos(phi_c) * sin(phi)
                     - sin(phi_c) * cos(phi) * cos(theta - theta_c))
        return x, y
    return _transform_fn

def gnomonic_projection(central_longitude, central_latitude):
    """ Transform points using a gnomonic projection with centre point centre.

        All points should be given in degrees longitude and degrees latitude

        Arguments:
            centre - the centre point of the projection - this is
                where the gnominic plane touches the surface of the
                earth. Given as a tuple of longitude and latitude in degrees

        Returns:
            a function to transform a point into the projective plane
    """
    theta_c, phi_c = longlat_to_spherical(central_longitude, central_latitude)
    def _transform_fn(longitude, latitude):
        theta, phi = longlat_to_spherical(longitude, latitude)
        scale = 1 / (sin(phi_c) * sin(phi)
                     + cos(phi_c) * cos(phi) * cos(theta - theta_c))
        x = scale * cos(phi) * sin(theta - theta_c)
        y = scale * (cos(phi_c) * sin(phi)
                     - sin(phi_c) * cos(phi) * cos(theta - theta_c))
        return x, y
    return _transform_fn
