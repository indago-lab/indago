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
    """Enum class for design variable types. Supported variable types are ``VariableType.Real``,
    ``VariableType.Integer``, ``VariableType.RealDiscrete``, ``VariableType.Categorical``."""

    Real = 'R'
    Integer = 'I'
    RealDiscrete = 'D'
    Categorical = 'C'

    def __str__(self):
        """String representation for design variable type."""
        return self.name


class XFormat(Enum):
    """Enum class for the formats of the design vector Candidate.X."""

    Tuple = 'tuple'
    List = 'list'
    Dict = 'dict'
    Ndarray = 'ndarray'
    Grouped = 'grouped'

    def __str__(self):
        return self.name + ': ' + self.value