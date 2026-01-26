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

File content: Definition of Candidate classes.
Usage: from indago import Candidate
"""
# from __future__ import annotations
# import typing
# if typing.TYPE_CHECKING:
#     from ._optimizer import Optimizer

import numpy as np
from typing import TypeAlias
from numpy.typing import NDArray

from enum import Enum

class VariableType(Enum):
    """Enum class for variable types."""

    Real = 'R'
    Integer = 'I'
    Discrete = 'D'
    Category = 'C'

    def __str__(self):
        return self.name + ': ' + self.value

X_Type: TypeAlias = tuple[int | float | str]


class XFormat(Enum):
    """Enum class for """

    Ndarray = 'ndarray'
    List = 'list'
    # Discrete = 'D'
    # Category = 'C'

    def __str__(self):
        return self.name + ': ' + self.value

class Candidate:
    """Base class for search agents in all optimization methods.
    Candidate solution for the optimization problem.

    Attributes
    ----------
    X : list[float | int | str]
        Design vector.
    O : ndarray
        Objectives' values.
    C : ndarray
        Constraints' values.
    f : float
        Fitness.

    Returns
    -------
    Candidate
        CandidateState instance.

    """

    def __init__(self, variables: dict, n_objectives: int = 1, n_constraints: int = 0,
                 x_format: XFormat = XFormat.Ndarray) -> None:
        """Candidate constructor."""
        # if optimizer is None:
        #     return

        X = []
        for var_name, (var_type, *var_params) in variables.items():
            if var_type == VariableType.Real:
                X.append(np.nan)
            elif var_type == VariableType.Integer:
                X.append(0)
            elif var_type == VariableType.Discrete:
                X.append(np.nan)
            elif var_type == VariableType.Category:
                X.append('X')
            else:
                raise ValueError(f'Unknown variable type {var_type} for variable {var_name}')

        self._X: X_Type = tuple(X)

        if x_format == XFormat.List:
            self._get_x = self.get_x_as_list
        elif x_format == XFormat.Ndarray:
            self._get_x = self.get_x_as_ndarray
        else:
            raise NotImplementedError

        self.O: NDArray[np.float64] = np.full(n_objectives, np.nan)
        self.C: NDArray[np.float64] = np.full(n_constraints, np.nan)
        self.f: np.float64 = np.float64(np.nan)

        # Comparison operators
        if n_objectives == 1 and n_constraints == 0:
            self._eq_fn = self._eq_fast
            self._lt_fn = self._lt_fast
            # self.__gt__ = self._gt_fast
        else:
            self._eq_fn = self._eq_full
            self._lt_fn = self._lt_full
            # self.__gt__ = self._gt_full

        # if optimizer.forward_unique_str:
        self.unique_str = None

    @property
    def X(self):
        return self._get_x()

    @X.setter
    def X(self, value):
        self._set_x(value)

    def get_x_as_list(self):
        return self._X
    def get_x_as_ndarray(self):
        return np.asarray(self._X, dtype=np.float64)
    def get_x_as_xy(self):
        X = []
        Y = []
        for x in self._X:
            if type(x) in [float, np.float64]:
                X.append(x)
            elif type(x) in [int, np.int32]:
                Y.append(x)
            else:
                raise NotImplementedError
        return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.int32)

    def _set_x(self, values):
        X = []
        if type(values) == list:
            for i, (x, v) in enumerate(zip(self._X, values)):
                assert type(x) == type(v), f'X[{i}] value mismatch (replacing {type(x)} with {type(v)})'
                X.append(v)
            self._X = tuple(X)
        elif type(values) == np.ndarray:
            self._X = values.tolist()

    def clip(self, optimizer) -> None:
        """Method for clipping (trimming) the design vector (Candidate.X)
        values to lower and upper bounds.

        Returns
        -------
        None
            Nothing

        """

        self.X = np.clip(self.X, optimizer.lb, optimizer.ub)

    def copy(self):
        """Method for creating a copy of the CandidateState.

        Returns
        -------
        Candidate
            CandidateState instance

        """

        candidate = Candidate(None)
        candidate.X = np.copy(self.X)
        candidate.O = np.copy(self.O)
        candidate.C = np.copy(self.C)
        candidate.f = self.f

        # Comparison operators
        if self.O.size == 1 and self.C.size == 0:
            candidate._eq_fn = candidate._eq_fast
            candidate._lt_fn = candidate._lt_fast
            # self.__gt__ = self._gt_fast
        else:
            candidate._eq_fn = candidate._eq_full
            candidate._lt_fn = candidate._lt_full
            # self.__gt__ = self._gt_full

        # if optimizer.forward_unique_str:
        candidate.unique_str = self.unique_str

        return candidate

        # previous solution (much slower)
        # cP = copy.deepcopy(self)
        # return cP

    def __str__(self):
        """Method for a useful printout of Candidate properties.

        Returns
        -------
        printout : str
            String of the fancy table of Candidate properties.

        """

        title = f'Indago {type(self).__name__}'
        if type(self) != Candidate:
            title += ' (subclass of Candidate)'
        table = Table(title=title)

        table.add_column('Property', justify='left', style='magenta')
        table.add_column('Value', justify='left', style='cyan')

        for var, value in vars(self).items():
            if not var.startswith('_'):
                if isinstance(value, (int, float, str, bool)):
                    table.add_row(var, str(value))
                elif isinstance(value, (list, dict)) and len(value) > 0:
                    table.add_row(var, str(value))
                elif isinstance(value, np.ndarray) and np.size(value) > 0:
                    if isinstance(value[0], (int, float)):
                        table.add_row(var, np.array_str(value, max_line_width=np.inf))

        Console().print(table)
        return ''

    def __eq__(self, other):
        """Equality operator wrapper.

        Parameters
        ----------
        other : Candidate
            The other of the two candidate solutions.

        Returns
        -------
        equal : bool
            ``True`` if candidate solutions are equal, ``False`` otherwise.

        """

        return self._eq_fn(self, other)

    @staticmethod
    def _eq_fast(a, b):
        """Private method for fast candidate solution equality check.
        Used in single objective, unconstrained optimization.

        Parameters
        ----------
        a : Candidate
            The first of the two candidate solutions.
        b : Candidate
            The second of the two candidate solutions.

        Returns
        -------
        equal : bool
            ``True`` if candidate solutions are equal, ``False`` otherwise.

        """

        return a.f == b.f

    @staticmethod
    def _eq_full(a, b):
        """Private method for full candidate solution equality check.
        Used in multiobjective and/or constrained optimization.

        Parameters
        ----------
        a : Candidate
            The first of the two candidate solutions.
        b : Candidate
            The second of the two candidate solutions.

        Returns
        -------
        equal : bool
            ``True`` if candidate solutions are equal, ``False`` otherwise.

        """

        # return np.sum(np.abs(a.X - b.X)) + np.sum(np.abs(a.O - b.O)) + np.sum(np.abs(a.C - b.C)) == 0.0
        return (a.X == b.X).all() and (a.O == b.O).all() and (a.C == b.C).all() and a.f == b.f

    def __ne__(self, other):
        """Inequality operator.

        Parameters
        ----------
        other : Candidate
            The other of the two candidate solutions.

        Returns
        -------
        not_equal : bool
            ``True`` if candidate solutions are not equal, ``False`` otherwise.

        """

        return self.f != other.f

    def __lt__(self, other):
        """Less-then operator wrapper.

        Parameters
        ----------
        other : Candidate
            The other of the two candidate solutions.

        Returns
        -------
        lower_than : bool
            ``True`` if the candidate solution is better than **other**, ``False`` otherwise.

        """

        return self._lt_fn(self, other)

    @staticmethod
    def _lt_fast(a, b):
        """Fast less-than operator.
        Used in single objective, unconstrained optimization.

        Parameters
        ----------
        a : Candidate
            The first of the two candidate solutions.
        b : Candidate
            The second of the two candidate solutions.

        Returns
        -------
        lower_than : bool
            ``True`` if **a** is better than **b**, ``False`` otherwise.

        """

        if np.isnan(a.f):
            return False
        if np.isnan(b.f):
            return True
        return a.f < b.f

    @staticmethod
    def _lt_full(a, b):
        """Private method for full less-than check.
        Used in multiobjective and/or constrained optimization.

        Parameters
        ----------
        a : Candidate
            The first of the two candidate solutions.
        b : Candidate
            The second of the two candidate solutions.

        Returns
        -------
        lower_than : bool
            ``True`` if **a** is better than **b**, ``False`` otherwise.

        """

        if np.isnan([*a.O, *a.C]).any():
            return False
        if np.isnan([*b.O, *b.C]).any():
            return True
        if np.sum(a.C > 0) == 0 and np.sum(b.C > 0) == 0:
            # Both are feasible
            # Better candidate is the one with smaller fitness
            return a.f < b.f

        elif np.sum(a.C > 0) == np.sum(b.C > 0):
            # Both are unfeasible and break same number of constraints
            # Better candidate is the one with smaller sum of unfeasible (positive) constraint values
            return np.sum(a.C[a.C > 0]) < np.sum(b.C[b.C > 0])

        else:
            # The number of unsatisfied constraints is not the same
            # Better candidate is the one which breaks fewer constraints
            return np.sum(a.C > 0) < np.sum(b.C > 0)

    def __gt__(self, other):
        """Greater-then operator wrapper.

        Parameters
        ----------
        other : Candidate
            The other of the two candidate solutions.

        Returns
        -------
        greater_than : bool
            ``True`` if the candidate solution is worse than **other**, ``False`` otherwise.

        """

        return not (self._lt_fn(self, other) or self._eq_fn(self, other))

    # are these two necessary?
    """
    def _gt_fast(self, other):
        return self.f > other.f
    def _gt_full(self, other):
        return not (self.__eq__(other) or self.__lt__(other))
    """

    def __le__(self, other):
        """Less-than-or-equal operator wrapper.

        Parameters
        ----------
        other : Candidate
            The other of the two candidate solutions.

        Returns
        -------
        lower_than_or_equal : bool
            ``True`` if the candidate solution is better or equal to **other**, ``False`` otherwise.

        """

        return self._lt_fn(self, other) or self.__eq__(other)

    def __ge__(self, other):
        """Greater-than-or-equal operator wrapper.

        Parameters
        ----------
        other : Candidate
            The other of the two candidate solutions.

        Returns
        -------
        greater_than_or_equal : bool
            ``True`` if the candidate solution is worse or equal to **other**, ``False`` otherwise.

        """

        return self.__gt__(other) or self.__eq__(other)

    def is_feasible(self):
        """
        Determines whether the design vector of a Candidate is a feasible solution. A feasible solution is
        a solution which satisfies all constraints.

        Returns
        -------
        is_feasible : bool
            ``True`` if the Candidate design vector is feasible, ``False`` otherwise.

        """

        return np.all(self.C <= 0)
