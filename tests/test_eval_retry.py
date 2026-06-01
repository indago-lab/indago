# -*- coding: utf-8 -*-
"""
test eval_retry
"""

# need this for local (non-pip) install only
import sys
sys.path.append('..')

import pytest

from indago import PSO
import numpy as np

def F(X):
    if np.logical_and(X > 1.95, X < 2.05).any():
        return 1/0
    else:
        return np.sum(X**2), -1

def test_eval_retry():

    opt = PSO()
    opt.constraints = 1
    opt.dimensions = 10
    opt.max_evaluations = 500 * opt.dimensions
    opt.lb = np.ones(opt.dimensions) * -5
    opt.ub = np.ones(opt.dimensions) * 5
    opt.evaluator = F

    opt.safe_evaluation = True
    opt.eval_fail_behavior = 'retry'
    opt.eval_retry_attempts = 10
    opt.eval_retry_recede = 0.05
    # opt.convergence_log_file = 'test_eval_retry.log'
    runs = 5

    with pytest.raises(AssertionError, match='TOO MANY FAILED EVALUATIONS. OPTIMIZATION ABORTED'):
        for r in range(runs):
            # print(f'\n************* run #{r} *************')
            test_results = []
            o = opt.copy()
            test_results.append(o.optimize(seed=4-r).f)  # run with seed=0 should fail
