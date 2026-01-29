# import sys
# sys.path.append('../tests/')
# from test_utils import *
import copy
import timeit
from typing import Any

import indago
import numpy as np


def real_function(x):
    x = np.asarray(x)
    f = np.sum((x - np.arange(x.size)) ** 2)
    # print(f'{x=}, {f=}')
    return f

def pso_real():
    real_variables_10D: dict[str, Any] = {f'x{i + 1}': (indago.VariableType.Real, -100, 100) for i in range(14)}

    optimizer = indago.PSO()
    optimizer.variables = real_variables_10D
    optimizer.evaluator = real_function
    # optimizer.monitoring = 'basic'
    optimizer.optimize()

def pso_mixed():
    real_variables_10D: dict[str, Any] = {f'x{i + 1}': (indago.VariableType.Real, -100, 100) for i in range(10)}
    numerical_variables_10D = copy.deepcopy(real_variables_10D)
    numerical_variables_10D['x11'] = indago.VariableType.RealDiscrete, np.linspace(-100, 100, 2001)
    numerical_variables_10D['x12'] = indago.VariableType.RealDiscrete, np.linspace(-100, 100, 1001)
    numerical_variables_10D['x13'] = indago.VariableType.Integer, -100, 100
    numerical_variables_10D['x14'] = indago.VariableType.Integer, -50, 150

    optimizer = indago.PSO()
    optimizer.variables = numerical_variables_10D
    optimizer.evaluator = real_function
    # optimizer.monitoring = 'basic'
    optimizer.optimize()

if __name__ == '__main__':

    t_real = timeit.timeit(pso_real, number=10)
    print(f'PSO 14D real-variable problem, {t_real=}')

    t_mixed = timeit.timeit(pso_mixed, number=10)
    print(f'PSO 14D real-variable problem, {t_mixed=}')
