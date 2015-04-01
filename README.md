# pymaxion - polygon projections for Shapely

Jess Robertson - jesse.robertson@csiro.au

```python
from pymaxion import DymaxionProjection
from shapely.geometry import Polygon

proj = DymaxionProjection
shape = Polygon([[0, 1], [0, 0], [1, 0]])
projected_shape = proj(shape)
```
