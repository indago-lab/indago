#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Indago
Python framework for numerical optimization
https://indago.readthedocs.io/
https://pypi.org/project/Indago/

Description: Indago contains several modern methods for real fitness function optimization over a real parameter domain
and supports multiple objectives and constraints. It was developed at the University of Rijeka, Faculty of Engineering.
Authors: Stefan Ivić, Siniša Družeta, Luka Grbčić
Contact: stefan.ivic@riteh.uniri.hr
License: MIT

File content: Definition of top-level Engine class.
Usage: from indago.core._engine import Engine
"""

import indago
import numpy as np
from numpy.typing import NDArray

class Engine:
    """Base class for posteriori multi-objective optimization methods.

    Attributes
    ----------
    variables : VariablesDictType
        A dictionary of optimization variables.
    dimensions : int
        Number of dimensions of the search space i.e. number of optimization variables.
    _dimensions : int | None
        Private attribute used when user explicitly sets optimizer.dimensions. Do not use it, use optimizer.dimensions
        instead.
    lb : ndarray or list of float or float
        Lower bounds for a real-variable problem. If of type float, it will be expanded to ndarray of float of size
        **dimensions**. If None, defaults to -1e100. Non finite (np.inf, np.nan) values (or members of ndarray or list)
        default to -1e100. If (all members) not float, Optimizer.X0 must be provided. Default: ``None``.
        TODO: Last sentence is confusing.
    ub : ndarray or list of float or float
        Upper bounds for a real-variable problem. If of type float, it will be expanded to ndarray of float of size
        **dimensions**. If None, defaults to 1e100. Non finite (np.inf, np.nan) values (or members of ndarray or list)
        default to 1e100. If (all members) not float, Optimizer.X0 must be provided. Default: ``None``.
        TODO: Last sentence is confusing.
    _all_real : bool
        Private flag indicating whether all variables are real (real-valued). Used for better performances of some
         optimization algorithms when dealing with real-variable problems.
    """

    @property
    def dimensions(self) -> int:
        return len(self.variables)

    @dimensions.setter
    def dimensions(self, value: int):
        self._dimensions = value

    def __init__(self):

        self.variables: indago.VariableDictType = dict()
        self.dimensions: int = 0 # Just to remove mypy warning (not having dimensions before initialization)
        self._dimensions: int | None = None
        self._all_real = False
        self.lb = None
        self.ub = None

        self._x_format = indago.XFormat.Tuple

    def _init_variables(self) -> None:
        """Private method for validating variables dictionary and initializing related attributes."""

        self._all_real: bool = all([var_type == indago.VariableType.Real for var_name, (var_type, *_) \
                                    in self.variables.items()])

        # TODO check types, ordering, etc.

        if self._all_real:
            lb: list[float] = []
            ub: list[float] = []
            for var_name, (var_type, *var_options) in self.variables.items():
                _lb = var_options[0]
                _ub = var_options[1]
                if _lb is None or not np.isfinite(_lb): _lb = -1e100
                if _ub is None or not np.isfinite(_ub): _ub = 1e100
                lb.append(_lb)
                ub.append(_ub)
            self.lb = np.asarray(lb)
            self.ub = np.asarray(ub)

        else:
            self.lb = None
            self.ub = None

    def _init_from_bounds(self) -> None:
        """Private method for validating bounds in real-variable optimization and initializing variables dictionary."""

        # Check for no bounds
        if self.lb is None:
            self.lb = -np.inf
        if self.ub is None:
            self.ub = np.inf
        if not np.isfinite(self.lb).all() or not np.isfinite(self.ub).all():
            assert self.X0 is not None, \
                "(some of the) bounds are not provided or are given as +/-np.inf or np.nan, optimizer.X0 needed"
        self.lb = np.nan_to_num(self.lb, nan=-1e100, posinf=1e100, neginf=-1e100).astype(float)
        self.ub = np.nan_to_num(self.ub, nan=1e100, posinf=1e100, neginf=-1e100).astype(float)

        # Check dimensions or get it from lb/ub
        if self._dimensions is not None:
            assert isinstance(self._dimensions, int) and self._dimensions > 0, \
                "optimizer.dimensions should be positive integer"
        else:
            self._dimensions = max(np.size(self.lb), np.size(self.ub))
            assert self._dimensions > 1, \
                "optimizer.lb and optimizer.ub both of size 1, missing optimizer.dimensions"

        # Expand scalar lb/ub
        if np.size(self.lb) == 1:
            self.lb = np.full(self._dimensions, self.lb)
        if np.size(self.ub) == 1:
            self.ub = np.full(self._dimensions, self.ub)

        # in case lb/ub is a list or tuple
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)

        assert np.size(self.lb) == np.size(self.ub) == self._dimensions, \
            "optimizer.lb and optimizer.ub should be of equal size or scalar"

        assert (self.lb < self.ub).all(), \
            "optimizer.lb should be strictly lower than optimizer.ub"

        self._all_real = True

        for i, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            self.variables[f'x{i}'] = indago.VariableType.Real, lb, ub