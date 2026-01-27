import numpy as np

import indago
from indago.core._optimizer import Optimizer

def test_variables_initialization():

    optimizer = Optimizer()
    optimizer.variables['a'] = indago.VariableType.Real, -1, 5
    optimizer.variables['b'] = indago.VariableType.Real, 1, None
    optimizer.variables['c'] = indago.VariableType.Real, None, 0
    optimizer.variables['d'] = indago.VariableType.Real, None, None
    for i in range(10):
        optimizer.variables[f'theta{1 + i}'] = indago.VariableType.Real, -20, 20
    optimizer.variables['n1'] = indago.VariableType.Integer, 2, 10
    optimizer.variables['n2'] = indago.VariableType.Integer, -2, 5
    optimizer.variables['h'] = indago.VariableType.RealDiscrete, np.linspace(1, 2, 21)
    optimizer.variables['s'] = indago.VariableType.RealDiscrete, np.linspace(0, 0.5, 51)
    optimizer.variables['PumpType'] = indago.VariableType.Categorical, 'A B C'.split()

    candidate = indago.Candidate(optimizer.variables)
    print(candidate.X)

    candidate.adjust()
    print(candidate.X)
