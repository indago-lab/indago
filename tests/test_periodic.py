import indago
import numpy as np
from copy import deepcopy


real_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealPeriodic, 0, 360) for i in range (0, 10)}
real_discrete_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealDiscretePeriodic, [float(_) for _ in range(0, 361)]) for i in range (0, 10)}
integer_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.IntegerPeriodic, 0, 360) for i in range (0, 10)}

def goalfun(x):
    x = np.asarray(x)
    offset = np.arange(np.size(x))
    return np.sum(np.sin(np.deg2rad(x - offset - 90))) + np.size(x)

# import matplotlib.pyplot as plt
# goalfun = np.vectorize(goalfun)
# plt.plot(np.arange(0, 361), goalfun(np.arange(0, 361)))
# plt.show()

def test_periodic():

    for vars in [real_periodic, real_discrete_periodic, integer_periodic]:

        optimizer = indago.PSO()
        optimizer.variables = vars
        optimizer.evaluator = goalfun
        optimizer.max_evaluations = 100
        optimizer.optimize(seed=0)
        print(f"{vars['var0'][0]} solution: {optimizer.best.f}")
        assert optimizer.eval == 100


def test_periodic():
    vars = {'x1': (indago.VariableType.Real, -10, 10),
            'x2': (indago.VariableType.RealPeriodic, -20, 20),
            'x3': (indago.VariableType.RealDiscrete, np.linspace(-30, 30, 31)),
            'x4': (indago.VariableType.RealDiscretePeriodic, np.linspace(-40, 40)),
            'x5': (indago.VariableType.Integer, -50, 50),
            'x6': (indago.VariableType.IntegerPeriodic, -60, 60),
            }
    optimizer = indago.PSO()
    optimizer.variables = vars
    candidate = indago.Candidate(variables=vars)
    print()

    X = np.arange(len(vars)) *10
    candidate._R = 1.05
    print(f'{candidate._R=}')
    print(f'{candidate.X=}')
    candidate._X = X + 2.345
    print(f'{candidate._R=}')
    print(f'{candidate.X=}')
    candidate.adjust()
    print(f'{candidate._R=}')
    print(f'{candidate.X=}')

    candidate._R = -0.5
    print()
    print(f'{candidate._R=}')
    print(f'{candidate.X=}')
    candidate.adjust()
    print(f'{candidate._R=}')
    print(f'{candidate.X=}')