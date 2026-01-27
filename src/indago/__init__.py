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


__version__ = '0.5.4'

from indago.core._candidate import Candidate, VariableType, XFormat, VariableDictType
# from indago.core._optimizer import Optimizer

# from indago._utility import *
# from indago._utility import _round_smooth
#
# from indago._pso import PSO
# from indago._fwa import FWA
# from indago._ssa import SSA
# from indago._de import DE
# from indago._ba import BA
# from indago._efo import EFO
# from indago._mrfo import MRFO
# from indago._abc import ABC
# from indago._nm import NM
# from indago._msgd import MSGD
# from indago._rs import RS
# from indago._gwo import GWO
# from indago._hbo import HBO
# from indago._crs import CRS

# optimizers = [PSO, FWA, SSA, DE, BA, EFO, MRFO, ABC, NM, MSGD, RS, GWO, HBO, CRS]
"""list of Optimizer : A list of all available Indago optimizer classes."""

# optimizers_name_list = [o.__name__ for o in optimizers]
"""list of str : A list of all available Indago method names (abbreviations)."""

# optimizers_dict = {o.__name__: o for o in optimizers}
"""dict : A dict of all available Indago optimizers, in the form of method
    name (abbreviation, type: str) as key, and optimizer class (type: Optimizer) 
    as value."""

# # Backward compatibility aliases
# NelderMead = NM
#
# # Undocumented optimizers
# from indago._eeeo import EEEO
# from indago._sa import SA
# from indago._gd import GD
# from indago._rbs import RBS
# # from indago._bo import BO
# from indago._esc import ESC
# from indago._ga import GA
# from indago._cmaes import CMAES
# from indago._dgs import DGS
