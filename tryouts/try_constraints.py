#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:34:28 2021

@author: stefan
"""

import sys
sys.path.append('..')
from indago import PSO, FWA, SSA, DE
import numpy as np
import matplotlib.pyplot as plt

def f(X):
    return np.array([np.sum(X**2), 
                     12 + np.sum(np.sin(X)),
                     7 - np.average(X),
                     ])

for optimizer in [SSA(),
                  PSO(), 
                  FWA(),
                  DE()
                  ]:

    optimizer.evaluator = f
    optimizer.processes = 1
    optimizer.dimensions = 20
    optimizer.lb = np.ones(optimizer.dimensions) * -10
    optimizer.ub = np.ones(optimizer.dimensions) *  10
    optimizer.objectives = 1
    optimizer.constraints = 2
    optimizer.max_evaluations = 30000

    if type(optimizer).__name__ == 'DE':
        optimizer.variant = 'LSHADE'
        optimizer.params['rank_enabled'] = True

    optimizer.optimize()
    optimizer.plot_history()

plt.show()