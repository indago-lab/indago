import indago
from test_utils import *

def test_real_problem():
    optimizer = indago.PSO()

    optimizer.variables = real_variables_10D
    optimizer.evaluator = real_function
    optimizer.monitoring = 'basic'
    optimizer.optimize()
    print(optimizer.best)


def test_numeric_problem():
    optimizer = indago.PSO()

    optimizer.variables = generate_variables_dict('real', 10)
    optimizer.evaluator = real_function
    optimizer.monitoring = 'basic'
    optimizer.optimize()
    print(optimizer.best)


