import fiona
import shapely
import pkg_resources

SHAPEFILE = pkg_resources.resource_filename('pymaxion.resources', 'ne_110m_land.shp')

def get_land(shapefile=None, bbox=None): 
    """ Get land from a shapefile
    """
    # Load up shape data from shapefile
    with fiona.drivers():
        with fiona.open(SHAPEFILE) as source:
            if bbox:
                source = source.filter(bbox=bbox)
            shapes = shapely.ops.cascaded_union([
                      shapely.geometry.asShape(s['geometry'])
                      for s in source])
    
    # There's a small issue with part of the polygon for North 
    # America, we'll just clip it out for now
    shapes = shapes.difference(shapely.geometry.box(-132.71, 54.04, -132.72, 54.05))
            
    # Clip if we've got something to clip
    if bbox:
        bbox_shape = shapely.geometry.box(*bbox)
        return shapes.intersection(bbox_shape)
    else:
        return shapes