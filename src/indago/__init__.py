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

File content: Indago init file.
Usage: import indago

"""


__version__ = '0.7.0'

from indago.core._candidate import Candidate, VariableType, VariableDictType, XFormat, X_Content_Type
from indago.core._optimizer import Optimizer, Status

# from indago._utility import *
# from indago._utility import _round_smooth

from indago.optimizers._rs import RS
from indago.optimizers._pso import PSO
from indago.optimizers._fwa import FWA
# from indago.optimizers._ssa import SSA
from indago.optimizers._de import DE
# from indago.optimizers._ba import BA
# from indago.optimizers._efo import EFO
# from indago.optimizers._mrfo import MRFO
from indago.optimizers._abc import ABC
from indago.optimizers._nm import NM
# from indago.optimizers._msgd import MSGD
from indago.optimizers._gwo import GWO
from indago.optimizers._hbo import HBO
from indago.optimizers._crs import CRS

optimizers: list[Optimizer] = [PSO, FWA, DE, ABC, NM, RS, GWO, HBO, CRS]
"""A list of all available Indago optimizer classes."""

optimizers_name_list: list[str] = [o.__name__ for o in optimizers]
"""A list of all available Indago method names (abbreviations)."""

optimizers_dict: dict = {o.__name__: o for o in optimizers}
"""A dict of all available Indago optimizers, in the form of method name (abbreviation, type: str) 
as key, and optimizer class (type: Optimizer) as value."""

# Backward compatibility aliases
NelderMead = NM

# Undocumented optimizers
from indago.optimizers._eeeo import EEEO
# from indago.optimizers._sa import SA
# from indago.optimizers._gd import GD
# from indago.optimizers._rbs import RBS
# # from indago.optimizers._bo import BO
# from indago.optimizers._esc import ESC
# from indago.optimizers._ga import GA
# from indago.optimizers._cmaes import CMAES
# from indago.optimizers._dgs import DGS
