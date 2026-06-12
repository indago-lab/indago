# -*- coding: utf-8 -*-
"""
test indago.minimize
"""

# need this for local (non-pip) install only
import sys
sys.path.append('..')

import numpy as np
import indago
from indago import minimize


def F(x):
    """CEC f3 - Discus Function"""
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)

MAXEVAL = 1000
TOL = 1e-10

params = {'swarm_size': 10,
          'inertia': 0.6,
          'cognitive_rate': 2.0,
          'social_rate': 2.0
          }

def test_minimize_old_style():
    X, f = minimize(F,
                    None,  # variables dict
                    -100*np.ones(10),  # lb
                    100*np.ones(10),  # ub
                    'PSO',
                    0,  # seed
                    variant='Vanilla',
                    max_evaluations=MAXEVAL,
                    params=params
                    )
    expected_result = 4.026320535708222
    result = np.log10(f)
    assert np.isclose(expected_result, result, atol=TOL, rtol=0), \
        f'TEST MINIMIZE FAILED, result={result}, expected={expected_result}'

def test_minimize_new_style():
    X, f = minimize(F,
                    {f'var{i}': (indago.VariableType.Real, -100, 100) for i in range(0, 10)},  # variables dict
                    None,  # lb
                    None,  # ub
                    'PSO',
                    0,  # seed
                    variant='Vanilla',
                    max_evaluations=MAXEVAL,
                    params=params
                    )
    expected_result = 4.026320535708222
    result = np.log10(f)
    assert np.isclose(expected_result, result, atol=TOL, rtol=0), \
        f'TEST MINIMIZE FAILED, result={result}, expected={expected_result}'


# stand-alone testing
if __name__ == '__main__':
    test_minimize_old_style()
    test_minimize_new_style()
