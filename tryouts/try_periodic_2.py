import indago
import numpy as np

variables_real = {'r1': (indago.VariableType.Real, 0, 1),
                  'r2': (indago.VariableType.Real, 0, 1),
                  'phi1': (indago.VariableType.Real, 0, 2 * np.pi),
                  'phi2': (indago.VariableType.Real, 0, 2 * np.pi)}
variables_periodic = {'r1': (indago.VariableType.Real, 0, 1),
                      'r2': (indago.VariableType.Real, 0, 1),
                      'phi1': (indago.VariableType.RealPeriodic, 0, 2 * np.pi),
                      'phi2': (indago.VariableType.RealPeriodic, 0, 2 * np.pi)}

def f(design):
    r1, r2, phi1, phi2 = design
    x1 = r1 * np.cos(phi1)
    y1 = r1 * np.sin(phi1)
    x2 = r2 * np.cos(phi2)
    y2 = r2 * np.sin(phi2)
    return  (x1 - 0.9)**2 + (y1 + 0.1)**2 + (x2 - 0.95)**2 + (y1 + 0.5)**2

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
    print(f'{lbl:20s} {np.mean(F)}')