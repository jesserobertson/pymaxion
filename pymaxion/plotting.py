import matplotlib.pyplot as plt
from collections import deque

def set_bounds(bbox, buffer=(0.1, 0.1), axes=None):
    """ Set bounds to the given bounding box with a given buffer
        in the x and y direction.
        
        Useful for setting axis bounds in conjunction with descartes:

            axes = gca()
            axes.add_patch(descartes.PolygonPatch(poly))
            set_limits(poly.bounds)
            
        Parameters:
            bbox - the bounding box to use for setting limits, given as
                a (minx, miny, maxx, maxy) tuple
            buffer - the buffer in the x and y direction, given as a
                fraction of the relevant bounding box direction. Optional, 
                defaults to 0.1 (10%) in each direction
            axes - the axes instance to set the limits on. Optional, if None
                defaults to the result of matplotlib.pyplot.gca().
    """
    axes = axes or plt.gca()
    
    # Set up bounds from bounding box
    minx, miny, maxx, maxy = bbox
    bx = buffer[0] * (maxx - minx)
    by = buffer[1] * (maxy - miny)
    
    # Set limits
    axes.set_xlim(minx - bx, maxx + bx)
    axes.set_ylim(miny - by, maxy + by)
    axes.set_aspect('equal')


def alternate_colors(colormap, length):
    cmap = plt.get_cmap(colormap)
    cdeque = deque([cmap(i / length) for i in range(length + 1)])
    colors = []
    while len(cdeque):
        colors.append(cdeque.popleft())
        try:
            colors.append(cdeque.pop())
        except IndexError:
            pass
    return colors