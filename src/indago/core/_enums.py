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

from enum import Enum


class VariableType(Enum):
    """Enum class for design variable types. Supported variable types are ``VariableType.REAL``,
    ``VariableType.REAL_DISCRETE``, ``VariableType.REAL_PERIODIC``, ``VariableType.REAL_DISCRETE_PERIODIC``,
    ``VariableType.INTEGER``, ``VariableType.INTEGER_PERIODIC``, ``VariableType.CATEGORICAL``."""

    REAL = 'R'
    REAL_PERIODIC = 'RP'
    REAL_DISCRETE = 'RD'
    REAL_DISCRETE_PERIODIC = 'RDP'
    INTEGER = 'I'
    INTEGER_PERIODIC = 'IP'
    CATEGORICAL = 'C'

    def is_real(self) -> bool:
        """Return whether this is a real-valued variable type."""

        return self in (VariableType.REAL,
                        VariableType.REAL_PERIODIC,
                        VariableType.REAL_DISCRETE,
                        VariableType.REAL_DISCRETE_PERIODIC,
                        )

    def is_integer(self) -> bool:
        """Return whether this is an integer-valued variable type."""

        return self in (VariableType.INTEGER,
                        VariableType.INTEGER_PERIODIC,
                        )

    def is_discrete(self) -> bool:
        """Return whether this is a discrete variable type."""

        return self in (VariableType.REAL_DISCRETE,
                        VariableType.REAL_DISCRETE_PERIODIC,
                        VariableType.INTEGER,
                        VariableType.INTEGER_PERIODIC,
                        VariableType.CATEGORICAL,
                        )

    def is_periodic(self) -> bool:
        """Return whether this is a periodic variable type."""

        return self in (VariableType.REAL_PERIODIC,
                        VariableType.REAL_DISCRETE_PERIODIC,
                        VariableType.INTEGER_PERIODIC,
                        )

    def __str__(self) -> str:
        """String representation for design variable type."""

        return self.name


class XFormat(Enum):
    """Enum class for the formats of the design vector Candidate.X."""

    TUPLE = 'tuple'
    LIST = 'list'
    DICT = 'dict'
    NDARRAY = 'ndarray'
    GROUPED = 'grouped'

    def __str__(self) -> str:
        """String representation for design vector format."""

        return self.name + ': ' + self.value
