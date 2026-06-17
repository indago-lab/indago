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

optimizers = [indago.ABC,
              indago.CRS,
              indago.DE,
              indago.EFO,
              indago.FWA,
              indago.GWO,
              indago.HBO,
              indago.NM,
              indago.PSO,
              indago.SSA
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

for vars_pair in [(real, real_periodic),
             (real_discrete, real_discrete_periodic),
             (integer, integer_periodic),
             ]:

    for opt_class in optimizers:

        res = [None, None]

        for r, vars in enumerate(vars_pair):

            res[r] = []
            for i in range(100):
                optimizer = opt_class()
                optimizer.variables = vars
                optimizer.evaluator = goalfun  # minimum on bounds
                # optimizer.evaluator = lambda x: -goalfun(x) + 2*len(vars)  # minimum in the middle
                optimizer.max_evaluations = 100 * dims**2
                optimizer.optimize()
                res[r].append(optimizer.best.f)
                # print(f"{vars['var0'][0]} solution #{i+1}: {optimizer.best.f}")

        imp = np.median(res[0]) - np.median(res[1])
        if imp > 0:
            print(f"*** {opt_class.__name__} {vars_pair[0]['var0'][0]}Periodic improvement: {imp}")


"""
*** ABC RealPeriodic improvement: 0.0017245661873372953
*** DE RealPeriodic improvement: 0.003355630779067198
*** FWA RealPeriodic improvement: 0.07699413382982812
*** GWO RealPeriodic improvement: 0.03908262134487295
*** HBO RealPeriodic improvement: 0.00034329579795233656
*** PSO RealPeriodic improvement: 0.0973233380240206
*** SSA RealPeriodic improvement: 0.00013338682671992785
*** ABC RealDiscretePeriodic improvement: 0.0021319372706374295
*** FWA RealDiscretePeriodic improvement: 0.000304598088614938
*** GWO RealDiscretePeriodic improvement: 0.017705707236527424
*** HBO RealDiscretePeriodic improvement: 0.0003807563097195654
*** PSO RealDiscretePeriodic improvement: 0.09682834945696062
*** SSA RealDiscretePeriodic improvement: 0.013927492310052081
*** ABC IntegerPeriodic improvement: 0.002664407023693194
*** DE IntegerPeriodic improvement: 0.00020320478323299085
*** FWA IntegerPeriodic improvement: 0.000304598088614938
*** GWO IntegerPeriodic improvement: 0.0178582320637366
*** HBO IntegerPeriodic improvement: 0.000304598088614938
*** PSO IntegerPeriodic improvement: 0.09104380937589873
*** SSA IntegerPeriodic improvement: 0.015067518257371404
"""