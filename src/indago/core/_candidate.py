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

from __future__ import annotations # To support using Candidate type annotation inside Candidate class code
from typing import TypeAlias, Any
from unicodedata import numeric

import numpy as np
from numpy.typing import NDArray

from rich.console import Console
from rich.table import Table
from enum import Enum


class VariableType(Enum):
    """Enum class for design variable types. Supported variable types are ``VariableType.Real``,
    ``VariableType.Integer``, ``VariableType.RealDiscrete``, ``VariableType.Categorical``"""

    Real = 'R'
    Integer = 'I'
    RealDiscrete = 'D'
    Categorical = 'C'

    def __str__(self):
        """String representation for design variable type"""
        return self.name


X_Content_Type: TypeAlias = int | float | str
"""Type for possible content of design vector ``X``"""

X_All_Containers = tuple[X_Content_Type] | list[X_Content_Type] | dict[str, X_Content_Type] | NDArray[np.float64]
"""All possible (container) types for the design vector ``X``"""

X_Storage_Type: TypeAlias = tuple[X_Content_Type]
"""Container type which is used for storing design vector ``X`` (``Candidate._X``)"""

from numbers import Real
VariableDictRealType = tuple[VariableType, Real | None, Real | None]
VariableDictDiscreteType = tuple[VariableType, list[Real | str]]
VariableDictType = dict[str, VariableDictRealType | VariableDictDiscreteType]
"""A (container) type ``Optimizer.variables`` dictionary uses for variable definitions"""

class XFormat(Enum):
    """Enum class for """

    Tuple = 'tuple'
    List = 'list'
    Dict = 'dict'
    Ndarray = 'ndarray'
    Grouped = 'grouped'

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
    _variables : VariableDictType
        A hidden attribute for accessing variables definitions dictionary.
    _x_format : XFormat:
        A hidden attribute for attribute X's return value format.
    """

    def __init__(self, variables: VariableDictType, n_objectives: int = 1, n_constraints: int = 0,
                 x_format: XFormat = XFormat.Tuple) -> None:
        """Candidate constructor.

        Parameters
        ----------
        variables : VariableDictType
            A dictionary containing definitions of all design variables.
        n_objectives : int
            Number of objectives.
        n_constraints : int
            Number of constraints.
        x_format : XFormat:
            A format for attribute X's return value.

        Returns
        -------
        Candidate
            Candidate instance.
        """

        X: list[X_Content_Type] = []
        type_count = {k: 0 for k in VariableType}
        for var_name, (var_type, *var_params) in variables.items():
            match var_type:
                case VariableType.Real:
                    X.append(np.nan)
                case VariableType.Integer:
                    X.append(0)
                case VariableType.RealDiscrete:
                    X.append(np.nan)
                case VariableType.Categorical:
                    X.append('X')
                case _:
                    raise ValueError(f'Unknown variable type {var_type} for variable {var_name}')
            type_count[var_type] += 1

        var_float = type_count[VariableType.Real] + type_count[VariableType.RealDiscrete]
        var_int = type_count[VariableType.Integer]
        var_str = type_count[VariableType.Categorical]
        x_is_homogenous = np.sum(np.array([var_float, var_int, var_str]) > 0) == 1

        self._variables = variables
        self.X: X_Storage_Type = tuple[X_Content_Type](X)

        match x_format:
            case XFormat.Tuple:
                self._get_x = self._get_X_as_tuple
            case XFormat.List:
                self._get_x = self._get_X_as_list
            case XFormat.Dict:
                self._get_x = self._get_X_as_dict
            case XFormat.Ndarray:
                assert x_is_homogenous, f'Cant use x_format {x_format} for heterogeneous variables'
                self._get_x = self._get_X_as_ndarray
            case _:
                raise NotImplementedError

        self._x_format: XFormat = x_format

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
        """A property for the design vector X.

        Returns
        -------
        X: X_All_Containers
            A container of all design variables. The type of the container is determined by ``x_format`` argument in
            the Candidate constructor.
        """

        return self._get_x()

    @X.setter
    def X(self, design: X_All_Containers) -> None:
        """Set value for property X.

        Parameters
        ----------
        design : X_All_Containers
            The design vector in any of supported formats (list, tuple, dict or numpy.ndarray)
        """

        X = []
        x_format: type = type(design)
        if x_format in [list, tuple, np.ndarray]:
            for i, (val, (var_name, (var_type, *_))) in enumerate(zip(design, self._variables.items())):
                if var_type == VariableType.Real or var_type == VariableType.RealDiscrete:
                    assert isinstance(val, (float, np.floating)), f'Invalid value type (value={val}, type={type(val)}) for X[{i}], expected {var_type}'
                    X.append(float(val))
                elif var_type == VariableType.Integer:
                    assert isinstance(val, (int, np.integer)), f'Invalid value type (value={val}, type={type(val)}) for X[{i}], expected {var_type}'
                    X.append(int(val))
                elif var_type == VariableType.Categorical:
                    assert isinstance(val, (str, np.str_)),f'Invalid value type (value={val}, type={type(val)}) for X[{i}], expected {var_type}'
                    X.append(val)
                else:
                    raise NotImplementedError(f'Unknown variable type {var_type} for variable {var_name}')

        elif x_format == dict:
            for i, ((var_name, v), x) in enumerate(zip(design.items(), self._X)):
                assert type(x) == type(v), f'X[{i}] value mismatch (replacing {type(x)} with {type(v)})'
                X.append(v)
        else:
            raise NotImplementedError(f'Unsupported x_format {x_format}')
        self._X = tuple(X)

    def _get_X_as_tuple(self) -> tuple[X_Content_Type]:
        """Utility function of getting a design vector X as a tuple.

        Returns
        -------
        X: tuple[X_Content_Type]
            A tuple of all design variables.
        """

        return tuple[X_Content_Type](self._X)

    def _get_X_as_list(self) -> list[X_Content_Type]:
        """Utility function of getting a design vector X as a list.

        Returns
        -------
        X: list[X_Content_Type]
            A list of all design variables.
        """

        return list(self._X)

    def _get_X_as_dict(self) -> dict[str, X_Content_Type]:
        """Utility function of getting a design vector X as a dictionary.

        Returns
        -------
        X: dict[str, X_Content_Type]
            A dictionary of all design variables in a form  ``'var_name': var_value``.
        """

        X: dict[str, X_Content_Type] = {}
        for (var_name, var), x in zip(self._variables.items(), self._X):
            X[var_name] = x
        return X

    def _get_X_as_ndarray(self) -> NDArray[np.float64]:
        """Utility function of getting a design vector X as a numpy.ndarray. It can be used only when all variables
        are real (``VariableType.Real`` or ``VariableType.RealDiscrete``).
        TODO this raises an error if called for mixed problems. Decide whether to remove/check/assert.

        Returns
        -------
        X: NDArray[np.float64]
            A list of all design variables.
        """

        return np.asarray(self._X, dtype=np.float64)

    def _get_X_as_grouped(self) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.str_]]:
        """Utility function of getting a design vector X as a tuple of numpy.ndarrays grouped by type.

        Returns
        -------
        X: tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.str_]]
            A tuple of grouped design variables.
        """

        design_float = []
        design_int = []
        design_str = []
        for x in self._X:
            if isinstance(x, (float, np.floating)):
                design_float.append(x)
            elif isinstance(x, (int, np.integer)):
                design_int.append(x)
            elif isinstance(x, (str, np.str_)):
                design_str.append(x)
            else:
                raise NotImplementedError
        return np.asarray(design_float, dtype=np.float64), np.asarray(design_int, dtype=np.int32), np.asarray(design_str, dtype=np.str_)

    def _set_X_rel(self, R: NDArray[float] | float) -> None:
        """Sets the design vector using ndarray or float of relative values [0, 1]. Correctly sets the values
        for all variable types. Raises an error if relative values are outside of range [0, 1].

        Parameters
        -------
        R: ndarray[float] or float
            Array of relative values in range [0, 1].
        """

        # expand scalar to array
        R = np.asarray(R, dtype=float)
        if np.size(R) == 1:
            R = np.full(len(self._variables), R)

        X: list[X_Content_Type] = []
        for (var_name, (var_type, *var_options)), r in zip(self._variables.items(), R):
            if r < 0 or r > 1:
                raise ValueError(f'Relative value {r} is out of [0, 1] range for variable {var_name}')
            match var_type:
                case VariableType.Real:
                    X.append(var_options[0] + r * (var_options[1] - var_options[0]))
                case VariableType.RealDiscrete:
                    i = int(round(r * len(var_options[0]) - 0.5))
                    X.append(var_options[0][i])
                case VariableType.Integer:
                    i = int(round(var_options[0] + r * (var_options[1] - var_options[0] + 0.5)))
                    X.append(i)
                case VariableType.Categorical:
                    i = int(round(r * len(var_options[0]) - 0.5))
                    X.append(var_options[0][i])
                case _:
                    raise NotImplementedError(f'Unknown variable type {var_type} for variable {var_name}')
        self._X = tuple(X)

    def _get_X_rel(self) -> NDArray[float]:
        """Gets the relative design vector using ndarray of values [0, 1].

        Returns
        -------
        R: ndarray[float]
            Array of relative values in range [0, 1].
        """

        R = []
        for (var_name, (var_type, *var_options)), x in zip(self._variables.items(), self._X):
            match var_type:
                case VariableType.Real:
                    R.append((x - var_options[0]) / (var_options[1] - var_options[0]))
                case VariableType.RealDiscrete:
                    R.append((var_options[0].index(x) + 0.5) / len(var_options[0]))
                case VariableType.Integer:
                    R.append((x - var_options[0] + 0.5) / (var_options[1] - var_options[0] + 1))
                case VariableType.Categorical:
                    R.append((var_options[0].index(x) + 0.5) / len(var_options[0]))
                case _:
                    raise NotImplementedError(f'Unknown variable type {var_type} for variable {var_name}')
        return np.array(R)

    def adjust(self) -> bool:
        """Checks the values of the design vector X and adjusts them to valid values defined by ``variables`` dict
        provide in the ``Candidate`` constructor.

        Returns
        -------
        changed: bool
            Return whether a design vector X is changed or not, as a ``bool`` value.
        """

        X: list[X_Content_Type] = []
        for (var_name, (var_type, *var_options)), x in zip(self._variables.items(), self._X):
            # print(f'{var_name=}  {var_type=}  {var_options=}  {x=}')

            # Real variable
            if var_type == VariableType.Real:
                # TODO: If x is nan or inf (how is this even possible?) set it to zero? Me not like.
                _x = 0.0 if np.isnan(x) or np.isinf(x) else x # nan or inf values
                if var_options[0] is not None and _x < float(var_options[0]): # lower bound
                    _x = float(var_options[0])
                if var_options[1] is not None and _x > float(var_options[1]): # upper bound
                    _x = float(var_options[1])
                X.append(_x)

            # Discrete variable
            if var_type == VariableType.RealDiscrete:
                if x in var_options[0]:
                    X.append(x)
                elif np.isnan(x) or np.isinf(x): # nan or inf values
                    X.append(var_options[0][0])
                else:
                    j = np.argmin(np.abs(np.asarray(var_options[0]) - x))
                    X.append(var_options[0][j])

            # Integer variable
            if var_type == VariableType.Integer:
                _x = x
                if x < var_options[0]:
                    _x = var_options[0]
                elif x > var_options[1]:
                    _x = var_options[1]
                X.append(_x)

            # Category variable
            if var_type == VariableType.Categorical:
                if x in var_options[0]:
                    X.append(x)
                else:
                    X.append(var_options[0][0])

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


    def copy(self) -> Candidate:
        """Method for creating a copy of the candidate.

        Returns
        -------
        Candidate
            Candidate instance

        """

        candidate: Candidate = self.__class__(self._variables, self.O.size, self.C.size, self._x_format)
        candidate.X = self.X
        candidate.O = np.copy(self.O)
        candidate.C = np.copy(self.C)
        candidate.f = self.f
        candidate.unique_str = self.unique_str

        return candidate

    def __str__(self) -> str:
        """Method for a useful printout of Candidate properties.
        TODO this needs to return a clean str. Move rich printout to dedicated method e.g. Candidate.rich_print(), consider __repr__ too

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

        for var, value in attributes:
            if not var.startswith('_'):
                if isinstance(value, (int, float, str, bool)):
                    table.add_row(var, str(value))
                elif isinstance(value, (list, tuple, dict)) and len(value) > 0:
                    table.add_row(var, str(value))
                elif isinstance(value, np.ndarray) and np.size(value) > 0:
                    if isinstance(value[0], (int, float)):
                        table.add_row(var, np.array_str(value))

        Console().print(table)
        return ''

    def __eq__(self, other: Candidate) -> bool:
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
    def _eq_fast(a: Candidate, b: Candidate) -> bool:
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
    def _eq_full(a: Candidate, b: Candidate) -> bool:
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

    def __ne__(self, other: Candidate) -> bool:
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

    def __lt__(self, other: Candidate) -> bool:
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
    def _lt_fast(a: Candidate, b: Candidate) -> bool:
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
    def _lt_full(a: Candidate, b: Candidate) -> bool:
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

    def __gt__(self, other: Candidate) -> bool:
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

    # TODO are these two necessary?
    """
    def _gt_fast(self, other):
        return self.f > other.f
    def _gt_full(self, other):
        return not (self.__eq__(other) or self.__lt__(other))
    """

    def __le__(self, other: Candidate) -> bool:
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

    def __ge__(self, other: Candidate) -> bool:
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

    def is_feasible(self) -> bool:
        """
        Determines whether the design vector of a Candidate is a feasible solution. A feasible solution is
        a solution which satisfies all constraints.

        Returns
        -------
        is_feasible : bool
            ``True`` if the Candidate design vector is feasible, ``False`` otherwise.

        """

        return np.all(self.C <= 0)
