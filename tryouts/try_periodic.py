import indago
import numpy as np
from copy import deepcopy

dims = 10
real: indago.VariableDictType = {f'var{i}': (indago.VariableType.Real, 0, 360) for i in range(0, dims)}
real_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealPeriodic, 0, 360) for i in range(0, dims)}
real_discrete: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealDiscrete, [float(_) for _ in range(0, 361)]) for i in range (0, dims)}
real_discrete_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealDiscretePeriodic, [float(_) for _ in range(0, 361)]) for i in range (0, dims)}
integer: indago.VariableDictType = {f'var{i}': (indago.VariableType.Integer, 0, 360) for i in range(0, dims)}
integer_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.IntegerPeriodic, 0, 360) for i in range(0, dims)}

optimizers = [indago.PSO,
              indago.NM,
              indago.CRS,
              indago.HBO,
              indago.DE,
              ]

def goalfun(x):
    x = np.asarray(x)
    offset = (1 + np.arange(np.size(x))) * 1.5
    # return np.sum(np.sin(np.deg2rad(x - offset + 90))) + np.size(x) # Periodic with minimum at the middle
    return np.sum(np.sin(np.deg2rad(x - offset - 90))) + np.size(x) # Periodic with minimum near the bounds

# import matplotlib.pyplot as plt
# f = np.vectorize(goalfun)
# plt.plot(np.arange(0, 361), f(np.arange(0, 361)))
# plt.show()

for vars in [real, real_periodic,
             real_discrete, real_discrete_periodic,
             integer, integer_periodic,
             ]:

    for opt_class in optimizers:

        res = []
        for i in range(50):
            optimizer = opt_class()
            optimizer.variables = vars
            optimizer.evaluator = goalfun  # minimum on bounds
            # optimizer.evaluator = lambda x: -goalfun(x) + 2*len(vars)  # minimum in the middle
            optimizer.max_evaluations = 2000
            optimizer.optimize()
            res.append(optimizer.best.f)
            # print(f"{vars['var0'][0]} solution #{i+1}: {optimizer.best.f}")

            # fig, axes = optimizer.plot_history()
            # fig.show()
            # print(optimizer.best.X, optimizer.best.f)

        print(f"*** {opt_class.__name__} {vars['var0'][0]} median solution: {np.median(res)}")

