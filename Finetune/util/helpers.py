"""Layer/module helpers (from timm / Ross Wightman)."""

from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes)):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_3tuple = _ntuple(3)
