import numpy as np

import indago
from indago.core._engine import Engine
from indago.core._optimizer import Optimizer


def test_optimizer_inheritance():

    o = Optimizer()
    assert isinstance(o, Engine)

def test_variables_initialization():
    e = Engine()
    e.variables['a'] = (indago.VariableType.Real, -2, 2.2)
    e.variables['b'] = (indago.VariableType.Real, -9.1, 20)
    e._init_variables()

    assert np.all(e.lb == [-2, -9.1])
    assert np.all(e.ub == [2.2, 20])

    e.variables['c'] = (indago.VariableType.Integer, 2, 8)
    e._init_variables()
    assert e.lb is None and e.ub is None

def test_variables_initialization_unbounded():
    e = Engine()
    e.variables['x1'] = (indago.VariableType.Real, -np.inf, np.inf)
    e.variables['x2'] = (indago.VariableType.Real, -np.nan, np.nan)
    e.variables['x3'] = (indago.VariableType.Real, None, None)
    e.variables['x4'] = (indago.VariableType.Real, np.inf, -np.inf)
    e.variables['x5'] = (indago.VariableType.Real, -9.9, np.inf)
    e.variables['x6'] = (indago.VariableType.Real, np.inf, 11.1)
    e._init_variables()
    print(e.lb)

    assert np.all(e.lb == [-1e100, -1e100, -1e100, -1e100, -9.9, -1e100])
    assert np.all(e.ub == [1e100, 1e100, 1e100, 1e100, 1e100, 11.1])


def test_bounds_initialization():

    e = Engine()
    e.lb = -5.432
    e.ub = 6.789
    e.dimensions = 10
    e._init_from_bounds()

    assert np.all(e.lb == -5.432)
    assert np.all(e.ub == 6.789)

    for var_name, (var_type, *var_options) in e.variables.items():
        assert var_options[0] == -5.432 and var_options[1] == 6.789