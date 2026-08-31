# -*- coding: utf-8 -*-
"""
INDAGO OPTIMIZERS PERFORMANCE TEST
a (mostly) comprehensive test of Indago optimizers performance

A TEST FOR EVERY NEW METHOD/FEATURE PERFORMANCE SHOULD BE ADDED HERE
"""

# need this for local (non-pip) install only
import sys

import indago

sys.path.append('..')
sys.path.append('../indagobench')

import numpy as np
import math
import indago
from indago import PSO, ACO


VARS = {
    'x1': (indago.VariableType.INTEGER, 0, 9),
    'x2': (indago.VariableType.INTEGER, 0, 9),
    'x3': (indago.VariableType.INTEGER, 0, 9),
    'x4': (indago.VariableType.INTEGER, 0, 9),
    'x5': (indago.VariableType.INTEGER, 0, 9),
    }

def F(x: tuple) -> float:
    """
    Fitness function for the 5-variable discrete optimization problem.
    Each x_i in {0, 1, ..., 9}
    """
    x1, x2, x3, x4, x5 = x
    Z = (7*x1 - x1**2
         + 5*x2**2 - (x2**3)/2
         + 4*x3*x4 - x3**2 - x4**2
         + 6*x5 - x5**2
         + 2*x1*x5
         - 3*x2*x3
         + x4*x5*np.sin(x2))
    return Z


MAXEVAL = 1000
RUNS = 500


pso, aco_vanilla, aco_spillover = PSO(), ACO(), ACO()
aco_spillover.variant = 'Spillover'

for optimizer in [pso, aco_vanilla, aco_spillover]:
    print(optimizer.__class__.__name__, optimizer.variant if optimizer.variant else '')
    optimizer.evaluator = F
    optimizer.variables = VARS
    optimizer.max_evaluations = MAXEVAL

    res = []
    for _ in range(RUNS):
        opt = optimizer.copy()
        opt.optimize()
        res.append(opt.best.f)

    print('f:', np.median(res))
