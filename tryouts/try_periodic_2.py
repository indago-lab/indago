import indago
import numpy as np

variables_real = {'r': (indago.VariableType.Real, 0, 1),
                  'phi': (indago.VariableType.Real, 0, 2 * np.pi)}
variables_periodic = {'r': (indago.VariableType.Real, 0, 1),
                      'phi': (indago.VariableType.RealPeriodic, 0, 2 * np.pi)}

def f(design):
    r, phi = design
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return  (x - 0.9)**2 + (y - 0.1)**2

for vars, lbl in zip([variables_real, variables_periodic], 'Real RealPeriodic'.split()):
    F = []
    for run in range(100):
        optimizer = indago.PSO()
        optimizer.variables = vars
        optimizer.evaluator = f
        optimizer.max_evaluations = 1000
        optimizer.optimize()
        F.append(optimizer.best.f)
        # print(optimizer.best.X, optimizer.best.f)
    print(f'{lbl:20s} {np.median(F)}')