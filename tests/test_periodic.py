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
        print(f'{vars['var0'][0]} solution: {optimizer.best.f}')
        assert optimizer.eval == 100
