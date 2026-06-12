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

File content: Definition of Indago utility functions.
Usage: from indago import minimize, minimize_exhaustive, inspect, inspect_optimizers, unconstrain, read_evals_db

"""


import indago
import numpy as np
import time
from rich.table import Table
from rich.console import Console
from copy import deepcopy
import os


def minimize(evaluator,
             variables=None,
             lb=None, 
             ub=None, 
             optimizer_name='PSO', 
             seed=None,
             **kwargs):
    """Shorthand one-line utility function for running an optimization.
        
    Parameters
    ----------
    evaluator : callable
        Evaluation function. Takes a design vector (ndarray) and returns fitness value (float),
        or in case of multiobjective and/or constrained optimization a tuple with
        objectives (float) and constraints (float).
    variables : dict or None
        A dictionary of optimization variables with their types, bounds, and allowed values.
    lb : list or ndarray or float or None
        Lower bounds. If ``None`` lower bounds will be taken from **evaluator.lb**.
    ub : list or ndarray or float or None
        Upper bounds. If ``None`` upper bounds will be taken from **evaluator.ub**.
    optimizer_name : str
        Name (abbreviation) of the optimization method used. Default value is ``'PSO'``.
    optimize_seed : int or None
        A random seed. Use the same value for reproducing identical stochastic procedures.
    **kwargs : kwarg
        Keyword arguments passed to the Optimizer object corresponding to the **optimizer_name**.

    Returns
    -------
    (X, f) or (X, f, O, C) : tuple
        Results of the optimization, comprising the design vector **X** (tuple)
        and the corresponding minimum fitness **f** (float).
        In case of more than one objective and/or defined constraints, results also include
        objectives **O** (ndarray), and constraints **C** (ndarray).
        :param variables:
    """
    
    assert optimizer_name in indago.optimizers_name_list, \
        f'Unknown optimizer name "{optimizer_name}". Use one of the following names: {", ".join(indago.optimizers_name_list)}.'

    # initialize optimizer
    opt = indago.optimizers_dict[optimizer_name]()
    
    # pass parameters
    opt.evaluator = evaluator
    if variables is not None:
        opt.variables = variables
    else:
        opt.lb = lb
        opt.ub = ub
    for kw, val in kwargs.items():
        setattr(opt, kw, val)
        # print(f'{kw=}: {val=}')
    
    # run
    result = opt.optimize(seed=seed)
    
    # return
    if opt.objectives == 1 and opt.constraints == 0:
        return result.X, result.f
    else:
        return result.X, result.f, result.O, result.C
