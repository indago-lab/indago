import copy
from typing import Any

import indago
from test_utils import *

def test_real_problem():
    optimizer = indago.PSO()

    optimizer.variables = real_variables_10D
    optimizer.evaluator = real_function
    # optimizer.monitoring = 'basic'
    optimizer.optimize()
    print(optimizer.best)

def test_numeric_problem():
    optimizer = indago.PSO()

    optimizer.variables = generate_variables_dict('real', 10)
    optimizer.evaluator = real_function
    # optimizer.monitoring = 'basic'
    optimizer.optimize()
    print(optimizer.best)

def real_function(x):
    f = np.sum((x - np.arange(len(x))) ** 2)
    # print(f'{x=}, {f=}')
    return f

def mixed_function(x):
    c = x[-1]
    x = np.asarray(x[:-1])
    if c == 'A':
        f = np.sum((x - np.arange(x.size)) ** 2)
    elif c == 'B':
        f = 10 + np.sum((x + np.arange(x.size)) ** 2)
    elif c == 'C':
        f = np.sum((x - np.arange(x.size)**2) ** 2)
    else:
        raise NotImplementedError(f'Invalid categorical variable value {c}')
    # print(f'{x=}, {f=}')
    return f

def test_pso_real():
    real_variables_10D: dict[str, Any] = {f'x{i + 1}': (indago.VariableType.Real, -100, 100) for i in range(14)}

    optimizer = indago.PSO()
    optimizer.variables = real_variables_10D
    optimizer.evaluator = real_function
    # optimizer.monitoring = 'basic'
    optimizer.optimize()

def test_pso_mixed_num():
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

def test_pso_mixed():
    real_variables_10D: dict[str, Any] = {f'x{i + 1}': (indago.VariableType.Real, -100, 100) for i in range(10)}
    numerical_variables_10D = copy.deepcopy(real_variables_10D)
    numerical_variables_10D['x11'] = indago.VariableType.RealDiscrete, np.linspace(-100, 100, 2001)
    numerical_variables_10D['x12'] = indago.VariableType.RealDiscrete, np.linspace(-100, 100, 1001)
    numerical_variables_10D['x13'] = indago.VariableType.Integer, -100, 100
    numerical_variables_10D['x14'] = indago.VariableType.Integer, -50, 150
    numerical_variables_10D['x15'] = indago.VariableType.Categorical, 'A B C'.split()

    optimizer = indago.PSO()
    optimizer.variables = numerical_variables_10D
    optimizer.evaluator = mixed_function
    # optimizer.monitoring = 'basic'
    optimizer.optimize()

