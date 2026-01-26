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

from rich.console import Console
from rich.table import Table
from enum import Enum


class VariableType(Enum):
    """Enum class for variable types."""

    Real = 'R'
    Integer = 'I'
    Discrete = 'D'
    Category = 'C'

    def __str__(self):
        return self.name + ': ' + self.value


# Define types
X_Content_Type: TypeAlias = int | float | str
X_All_Containers = tuple[X_Content_Type] | list[X_Content_Type] | dict[str, X_Content_Type] | NDArray[np.float64]
X_Storage_Type: TypeAlias = tuple[X_Content_Type]


class XFormat(Enum):
    """Enum class for """

    Tuple = 'tuple'
    List = 'list'
    Dict = 'dict'
    Ndarray = 'ndarray'
    TypeSplit = 'type_split'

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
                 x_format: XFormat = XFormat.Tuple) -> None:
        """Candidate constructor."""

        X: list[X_Content_Type] = []
        type_count = {k: 0 for k in VariableType}
        for var_name, (var_type, *var_params) in variables.items():
            match var_type:
                case VariableType.Real:
                    X.append(np.nan)
                case VariableType.Integer:
                    X.append(0)
                case VariableType.Discrete:
                    X.append(np.nan)
                case VariableType.Category:
                    X.append('X')
                case _:
                    raise ValueError(f'Unknown variable type {var_type} for variable {var_name}')
            type_count[var_type] += 1

        var_float = type_count[VariableType.Real] + type_count[VariableType.Discrete]
        var_int = type_count[VariableType.Integer]
        var_str = type_count[VariableType.Category]
        x_is_homogenous = np.sum(np.array([var_float, var_int, var_str]) > 0) == 1

        self._X: X_Storage_Type = tuple[X_Content_Type](X)
        self._variables = variables

        match x_format:
            case XFormat.Tuple:
                self._get_x = self.get_x_as_tuple
            case XFormat.List:
                self._get_x = self.get_x_as_list
            case XFormat.Dict:
                self._get_x = self.get_x_as_dict
            case XFormat.Ndarray:
                assert x_is_homogenous, f'Cant use x_format {x_format} for heterogeneous variables'
                self._get_x = self.get_x_as_ndarray
            case _:
                raise NotImplementedError

        self._x_format = x_format

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
    def X(self) -> X_All_Containers:
        return self._get_x()

    @X.setter
    def X(self, value: X_All_Containers) -> None:
        self._set_x(value)

    def get_x_as_tuple(self) -> tuple[X_Content_Type]:
        return self._X

    def get_x_as_list(self) -> list[X_Content_Type]:
        return list(self._X)

    def get_x_as_dict(self) -> dict[str, X_Content_Type]:
        X: dict[str, X_Content_Type] = {}
        for (var_name, var), x in zip(self._variables.items(), self._X):
            X[var_name] = x
        return X

    def get_x_as_ndarray(self) -> NDArray[np.float64]:
        return np.asarray(self._X, dtype=np.float64)

    def get_x_as_xy(self) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.str_]]:
        X_float = []
        X_int = []
        X_str = []
        for x in self._X:
            if type(x) in [float, np.float64]:
                X_float.append(x)
            elif type(x) in [int, np.int32]:
                X_int.append(x)
            elif type(x) in [str]:
                X_str.append(x)
            else:
                raise NotImplementedError
        return np.asarray(X_float, dtype=np.float64), np.asarray(X_int, dtype=np.int32), np.asarray(X_str, dtype=np.str_)

    def _set_x(self, values: X_All_Containers):

        X = []
        x_format: type = type(values)
        if x_format in [list, tuple, np.ndarray]:
            for i, (v, x) in enumerate(zip(values, self._X)):
                assert type(x) == type(v), f'X[{i}] value mismatch (replacing {x} of type {type(x)} with {v} of type {type(v)})'
                X.append(v)
        elif x_format == dict:
            for i, ((var_name, v), x) in enumerate(zip(values.items(), self._X)):
                assert type(x) == type(v), f'X[{i}] value mismatch (replacing {type(x)} with {type(v)})'
                X.append(v)
        elif x_format == np.ndarray:
            self._X = values.tolist()
        else:
            raise NotImplementedError(f'Unsupported x_format {x_format}')
        self._X = tuple(X)


    def adjust(self) -> bool:
        X: list[X_Content_Type] = []
        for (var_name, options), x in zip(self._variables.items(), self._X):

            # Real variable
            if options[0] == VariableType.Real:
                _x = 0.0 if np.isnan(x) or np.isinf(x) else x # nan or inf values
                if options[1] is not None and x < float(options[1]): # lower bound
                    _x = float(options[1])
                if options[2] is not None and x > float(options[2]): # upper bound
                    _x = float(options[2])
                X.append(_x)

            # Discrete variable
            if options[0] == VariableType.Discrete:
                if x in options[1]:
                    X.append(x)
                elif np.isnan(x) or np.isinf(x): # nan or inf values
                    X.append(options[1][0])
                else:
                    j = np.argmin(np.abs(np.asarray(options[1]) - _x))
                    X.append(options[1][j])

            # Integer variable
            if options[0] == VariableType.Integer:
                _x = x
                if x< options[1]:
                    _x = options[1]
                elif x > options[2]:
                    _x = options[2]
                X.append(_x)

            # Category variable
            if options[0] == VariableType.Category:
                if x in options[1]:
                    X.append(x)
                else:
                    X.append(options[1][0])


        X = tuple[X_Content_Type](X)
        changed = not (X == self._X)
        # print(f'{self._X=}  ==>>  {X=}  {changed=}')
        if changed:
            self.X = X
        return changed


    def clip(self, optimizer) -> None:
        """Method for clipping (trimming) the design vector (Candidate.X)
        values to lower and upper bounds.

        Returns
        -------
        None
            Nothing

        """
        # TODO this needs to be reimplemented to use Candidate.variables instead of Optimizer
        self.X = np.clip(self.X, optimizer.lb, optimizer.ub)

    def copy(self):
        """Method for creating a copy of the CandidateState.

        Returns
        -------
        Candidate
            CandidateState instance

        """

        candidate: Candidate = Candidate(self._variables, self.O.size, self.C.size, self._x_format)
        candidate.X = self.X  #np.copy(self.X)
        candidate.O = np.copy(self.O)
        candidate.C = np.copy(self.C)
        candidate.f = self.f

        # # Comparison operators
        # if self.O.size == 1 and self.C.size == 0:
        #     candidate._eq_fn = candidate._eq_fast
        #     candidate._lt_fn = candidate._lt_fast
        #     # self.__gt__ = self._gt_fast
        # else:
        #     candidate._eq_fn = candidate._eq_full
        #     candidate._lt_fn = candidate._lt_full
        #     # self.__gt__ = self._gt_full

        # if optimizer.forward_unique_str:
        # candidate.unique_str = self.unique_str

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

        attributes = list(vars(self).items())
        attributes.append(('X', self.X))
        print(attributes)
        for var, value in attributes:
            print(f'{var}: {value}')
            if not var.startswith('_'):
                if isinstance(value, (int, float, str, bool)):
                    table.add_row(var, str(value))
                elif isinstance(value, (list, tuple, dict)) and len(value) > 0:
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
