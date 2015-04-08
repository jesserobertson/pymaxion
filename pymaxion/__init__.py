from .dymaxion import DymaxionProjection, dymaxion_transform
from . import rotations, conversions, utilities, operations, plotting

# Load git autogenerated version - update with setup.py update_version
from ._version import __version__

__all__ = ['DymaxionProjection', 'dymaxion_transform',
    '__version__', 'rotations', 'conversions', 'utilities', 
    'operations', 'plotting']
