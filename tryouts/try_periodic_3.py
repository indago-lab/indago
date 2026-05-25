from random import seed

import indago
import numpy as np
import matplotlib.pyplot as plt

dims = 5
vars_real = {f'x{i}': (indago.VariableType.Real, -100, 100) for i in range(dims)}
vars_realperiodic = {f'x{i}': (indago.VariableType.RealPeriodic, -100, 100) for i in range(dims)}

def f(design):
    # return np.sum(np.asarray(design)**2)
    x = (np.asarray(design) + 100) / 200
    c = np.cos(x * 2 * np.pi + np.pi + np.arange(np.size(x)) * 0.012345)
    return np.sum(1 + c) #** 2

fig, ax = plt.subplots()
for vars, lbl, c in zip([vars_real, vars_realperiodic], 'Real RealPeriodic'.split(), 'red green'.split()):
    F = []
    V = []
    print(f'{lbl:20s}', end='')
    for run in range(50):
        optimizer = indago.PSO()
        optimizer.variables = vars
        optimizer.evaluator = f
        optimizer.max_evaluations = 25 * dims ** 2
        # optimizer.params['inertia'] = 0.7
        optimizer.optimize()
        F.append(optimizer.best.f)
        V.append(optimizer.v_avg)
        # print(optimizer.best.X, optimizer.best.f)
        print('.', end='')

    ax.plot(np.array(V).T, c=c, lw=0.5)
    ax.plot(np.average(V, axis=0), c=c, lw=2, label=lbl)
    print(f'  {np.median(F)}')
    # break
ax.legend()

X = np.linspace(-100, 100, 201)
F = np.array([f(x) for x in X])

fig, ax = plt.subplots()
ax.plot(X, F)
i_min = np.argmin(F)
ax.plot(X[i_min], F[i_min], 'rx', ms=10)
plt.show()