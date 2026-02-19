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

File content: Validation utility functions.
Usage: from indago.utils._validation import validate_variables
"""
import numpy as np

import indago
from numbers import Real

def validate_variables(variables: indago.VariableDictType) -> tuple[bool, list[tuple[Exception, str]]]:

    validation_log = []
    if not isinstance(variables, dict):
        validation_log.append((TypeError, 'Variables should be a dictionary'))
    else:
        for var_name, (var_type, *var_options) in variables.items():
            if not isinstance(var_name, str): validation_log.append((TypeError, f'Variable name {var_name} is not a string'))
            if not isinstance(var_type, indago.VariableType): validation_log.append(
                (TypeError, f'Variable type {var_type} for variable {var_name} is not an indago.VariableType'))

            match var_type:
                case indago.VariableType.Real:
                    if len(var_options) != 2:
                        validation_log.append((ValueError, f'Definition of real variable {var_name} needs to be a tuple '
                                                           f'with exactly three items (indago.VariableType.Real, lb, ub)'))
                    else:
                        lb, ub = var_options
                        if not isinstance(lb, Real): validation_log.append((TypeError, f'Unsupported lb type ({type(lb)}) for {var_name}'))
                        if not isinstance(ub, Real): validation_log.append((TypeError, f'Unsupported ub type ({type(ub)}) for {var_name}'))

                        if isinstance(lb, Real) and isinstance(ub, Real):
                            if np.isnan(lb):
                                validation_log.append((ValueError, f'Definition of lb for {var_name} does not allow NaN values'))
                            if np.isnan(ub):
                                validation_log.append((ValueError, f'Definition of ub for {var_name} does not allow NaN values'))
                            if lb >= ub:
                                validation_log.append((ValueError, f'Lower bound of real variable {var_name} ({lb}) should '
                                                                   f'be strictly lower than upper bound ({ub})'))

                case indago.VariableType.RealDiscrete:
                    if len(var_options) != 1:
                        validation_log.append((ValueError, f'Definition of real discrete variable {var_name} needs to '
                                                           f'be a tuple with exactly two items '
                                                           f'(indago.VariableType.RealDiscrete, discrete_values)'))
                    else:
                        discrete_values = var_options[0]
                        if not isinstance(discrete_values, (list, tuple, np.ndarray)):
                            validation_log.append((TypeError, f'Discrete values {discrete_values} for {var_name} should be a list, tuple, or numpy array'))
                        else:
                            for v in discrete_values:
                                if not isinstance(v, Real):
                                    validation_log.append((TypeError, f'Discrete value {v} for {var_name} should be a real number'))
                                else:
                                    if not np.isfinite(v):
                                        validation_log.append((ValueError, f'Discrete value {v} for {var_name} should be a real number'))

                            if all(isinstance(v, Real) for v in discrete_values):
                                discrete_values = np.asarray(discrete_values)
                                if not np.all(discrete_values[:-1] < discrete_values[1:]):
                                    validation_log.append((ValueError, f'Discrete values {discrete_values} for {var_name} should be sorted'))

                case indago.VariableType.Integer:
                    if len(var_options) != 2:
                        validation_log.append((ValueError, f'Definition of integer variable {var_name} needs to be a tuple '
                                                           f'with exactly three items (indago.VariableType.Integer, lb, ub)'))
                    else:
                        lb, ub = var_options
                        if not isinstance(lb, (int, np.integer)): validation_log.append((TypeError, f'Unsupported lb type ({type(lb)}) for {var_name}'))
                        if not isinstance(ub, (int, np.integer)): validation_log.append((TypeError, f'Unsupported ub type ({type(ub)}) for {var_name}'))

                        if isinstance(lb, (int, np.integer)) and isinstance(ub, (int, np.integer)):
                            if lb >= ub:
                                validation_log.append((ValueError, f'Lower bound of real variable {var_name} ({lb}) should '
                                                                   f'be strictly lower than upper bound ({ub})'))

                case indago.VariableType.Categorical:
                    if len(var_options) != 1:
                        validation_log.append((ValueError, f'Definition of categorical variable {var_name} needs to be '
                                                           f'a tuple with exactly two items (indago.VariableType.Categorical, '
                                                           f'list_of_string_values)'))
                    else:
                        str_values = var_options[0]
                        if not isinstance(str_values, (list, tuple, np.ndarray)):
                            validation_log.append((TypeError, f'Values {str_values} for {var_name} should be a list, tuple, or numpy array'))
                        else:
                            for v in str_values:
                                if not isinstance(v, str):
                                    validation_log.append((TypeError, f'Categorical value {v} for {var_name} should be a string'))

    return len(validation_log) == 0, validation_log