import indago
import numpy as np
from copy import deepcopy


real: indago.VariableDictType = {f'var{i}': (indago.VariableType.Real, 0, 360) for i in range (0, 10)}
real_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealPeriodic, 0, 360) for i in range (0, 10)}
real_discrete_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealDiscretePeriodic, [float(_) for _ in range(0, 361)]) for i in range (0, 10)}
integer_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.IntegerPeriodic, 0, 360) for i in range (0, 10)}

optimizers = [indago.PSO]

def goalfun(x):
    x = np.asarray(x)
    offset = np.arange(np.size(x))
    return np.sum(np.sin(np.deg2rad(x - offset - 90))) + np.size(x)

# import matplotlib.pyplot as plt
# goalfun = np.vectorize(goalfun)
# plt.plot(np.arange(0, 361), goalfun(np.arange(0, 361)))
# plt.show()

for vars in [real, real_periodic, real_discrete_periodic, integer_periodic]:

    for opt_class in optimizers:

        res = []
        for i in range(20):
            optimizer = opt_class()
            optimizer.variables = vars
            optimizer.evaluator = goalfun
            optimizer.optimize()
            res.append(optimizer.best.f)
            # print(f'{vars['var0'][0]} solution #{i+1}: {optimizer.best.f}')

        print(f'*** {opt_class.__name__} {vars['var0'][0]} median solution: {np.median(res)}')

