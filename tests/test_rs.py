
import indago
import numpy as np

def test_initialize():

    optimizer = indago.RS()
    optimizer.lb = -100
    optimizer.ub = 100
    optimizer.dimensions = 10
    optimizer.evaluator = lambda x: np.sum(x**2)
    # optimizer.monitoring = 'basic'
    best = optimizer.optimize()
    print(best)