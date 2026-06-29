"""Compatibility patch for matplotlib 3.10.x on Python 3.14.

matplotlib/path.py:285 implements __deepcopy__ as:
    p = copy.deepcopy(super(), memo)

In Python 3.14, deepcopy-ing a bound super() object causes it to deepcopy the
underlying instance, which calls __deepcopy__ again — infinite recursion. The fix
replaces __deepcopy__ with a direct attribute copy that bypasses super() entirely.

Call patch_matplotlib() once at process start before any matplotlib import occurs.
Importing this module applies the patch automatically.
"""
from __future__ import annotations

import copy


def patch_matplotlib() -> None:
    from matplotlib.path import Path

    if getattr(Path, "_py314_patched", False):
        return

    def _deepcopy(self, memo=None):
        if memo is None:
            memo = {}
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    Path.__deepcopy__ = _deepcopy
    Path._py314_patched = True


patch_matplotlib()
