# Non linear cmap
"""
nlcmap - a nonlinear cmap from specified levels

Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
Release under MIT license.

Some hacks added 2012 noted in code (@MRR)
"""

from pylab import *
from numpy import *
from matplotlib.colors import LinearSegmentedColormap


class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""

    name = 'nlcmap'

    def __init__(self, cmap, levels):
        self.cmap = cmap
        # @MRR: Need to add N for backend
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = asarray(levels, dtype='float64')
        self._x = self.levels / self.levels.max()
        self._y = linspace(0.0, 1.0, len(self.levels))

    # @MRR Need to add **kw for 'bytes'
    def __call__(self, xi, alpha=1.0, **kw):
        """docstring for fname"""
        # @MRR: Appears broken?
        # It appears something's wrong with the
        # dimensionality of a calculation intermediate
        # yi = stineman_interp(xi, self._x, self._y)
        yi = interp(xi, self._x, self._y)
        return self.cmap(yi, alpha)

