# pymaxion - polygon projections for Shapely

Jess Robertson - jesse.robertson@csiro.au

[![Build Status](https://travis-ci.org/jesserobertson/pymaxion.svg?branch=develop)](https://travis-ci.org/jesserobertson/pymaxion)

A library to create projections of the globe onto polygons. Some examples include the [Dymaxion projection](http://en.wikipedia.org/wiki/Dymaxion_map) of Buckminster Fuller, the [Cahill Butterfly projection](http://en.wikipedia.org/wiki/Bernard_J._S._Cahill) and the [Waterman Butterfly]().

### Examples 

```python
from pymaxion import DymaxionProjection
from shapely.geometry import Polygon

proj = DymaxionProjection
shape = Polygon([[0, 1], [0, 0], [1, 0]])
projected_shape = proj(shape)
```

### Install

Requirements are in `requirements.txt`. Install using [Anaconda] and my binstar builds:

```bash
conda config --add channels jesserobertson
conda install pymaxion
```

Install from source:

```bash
# cd to-source-directory
conda install --file requirements.txt
python setup.py test
python setup.py install
```
