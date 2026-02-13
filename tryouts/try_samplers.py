# import sys
# sys.path.append('../tests/')
# from test_utils import *
from typing import Any

import indago
import numpy as np
import matplotlib.pyplot as plt


def fun(x):
    x = np.asarray(x)
    f = np.sum((x - np.arange(x.size)) ** 2)
    return f

SAMPLER = 'random halton sobol lhs'.split()

for sampler in SAMPLER:

    real_variables_2D: dict[str, Any] = {f'x{i + 1}': (indago.VariableType.Real, -100, 100) for i in range(2)}

    optimizer = indago.PSO()
    optimizer.params = {'swarm_size': 100}
    optimizer.variables = real_variables_2D
    optimizer.evaluator = fun
    optimizer.sampler = sampler
    optimizer._init_optimizer()
    optimizer.params['inertia'] = None
    optimizer._init_method()
    # print(optimizer)

    plt.figure()
    for c in optimizer._swarm:
        plt.plot(c.X[0], c.X[1], 'o')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.title(sampler)

plt.show()
