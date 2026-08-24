#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
testing performance of Indago optimizers on constrained problems
"""

import sys
sys.path.append('..')
import indago
import numpy as np


RUNS = 30

def f2(X):
    return np.array([np.sum(X**2), 
                     12 + np.sum(np.sin(X)),
                     7 - np.average(X),
                     ])
f2.constraints = 2

def f1_1(X):
    return np.array([np.sum(X**2),
                     12 + np.sum(np.sin(X)),
                     ])
f1_1.constraints = 1

def f1_2(X):
    return np.array([np.sum(X**2),
                     7 - np.average(X),
                     ])
f1_2.constraints = 1


for f in [f1_1, f1_2,f2]:

    print(f'testing on {f.__name__}...')

    for optimizer in indago.optimizers:

        res = []
        for _ in range(RUNS):

            opt = optimizer()

            opt.evaluator = f
            opt.processes = 1
            opt.dimensions = 20
            opt.lb = np.ones(opt.dimensions) * -10
            opt.ub = np.ones(opt.dimensions) *  10
            opt.objectives = 1
            opt.constraints = f.constraints
            opt.max_evaluations = 30000

            if optimizer == indago.DE:
                opt.variant = 'LSHADE'
                opt.params['rank_enabled'] = True

            c = opt.optimize()

            if c.is_feasible():
                res.append(c.f)
            else:
                res.append(np.inf)

        print(f'   {optimizer.__name__}... {np.median(res):.2e}')
