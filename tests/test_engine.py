import pytest
import numpy as np

import indago
from indago.core._engine import Engine
from indago.core._optimizer import Optimizer
from test_utils import mixed_variables

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
    e.variables['x2'] = (indago.VariableType.Real, -9.9, np.inf)
    e.variables['x3'] = (indago.VariableType.Real, -np.inf, 11.1)
    e._init_variables()

    assert np.all(e.lb == np.array([-np.inf, -9.9, -np.inf]))
    assert np.all(e.ub == np.array([np.inf, np.inf, 11.1]))

    # Mixed design vector
    e = Engine()
    e.variables['x1'] = (indago.VariableType.Real, -10.0, 10.0)
    e.variables['x2'] = (indago.VariableType.RealDiscrete, [0.0, 0.1, 0.2])
    e._init_variables()
    assert e.lb is None and e.ub is None

    e = Engine()
    e.variables['x1'] = (indago.VariableType.Real, -10.0, 10.0)
    e.variables['x2'] = (indago.VariableType.Integer, 10, 19)
    e._init_variables()
    assert e.lb is None and e.ub is None

    e = Engine()
    e.variables['x1'] = (indago.VariableType.Real, -10.0, 10.0)
    e.variables['x2'] = (indago.VariableType.Categorical, 'abc'.split())
    e._init_variables()
    assert e.lb is None and e.ub is None

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


    e = Engine()
    e.lb = -5.432
    e.ub = [6.789, 6.789, 6.789]
    e.dimensions = 3
    e._init_from_bounds()

    assert np.all(e.lb == -5.432)
    assert np.all(e.ub == 6.789)

    for var_name, (var_type, *var_options) in e.variables.items():
        assert var_options[0] == -5.432 and var_options[1] == 6.789


    e = Engine()
    e.lb = [-5.432, -5.432, -5.432]
    e.ub = 6.789
    e.dimensions = 3
    e._init_from_bounds()

    assert np.all(e.lb == -5.432)
    assert np.all(e.ub == 6.789)

    for var_name, (var_type, *var_options) in e.variables.items():
        assert var_options[0] == -5.432 and var_options[1] == 6.789


def test_variables_validation():

    e = Engine()
    with pytest.raises(Exception) as exc:
        e.variables['a'] = (indago.VariableType.Real, 1)
        e._init_variables()
    assert "tuple with exactly three items (indago.VariableType.Real, lb, ub)" in str(exc.value)

    e = Engine()
    with pytest.raises(Exception) as exc:
        e.variables['b'] = (indago.VariableType.RealDiscrete, 1.1, 1.2 ,1.3, 1.4)
        e._init_variables()
    assert "needs to be a tuple with exactly two items (indago.VariableType.RealDiscrete, discrete_values)" in str(exc.value)

    e = Engine()
    with pytest.raises(Exception) as exc:
        e.variables['a'] = (indago.VariableType.Integer, 1, 2, 3)
        e._init_variables()
    assert "tuple with exactly three items (indago.VariableType.Integer, lb, ub)" in str(exc.value)

    e = Engine()
    with pytest.raises(Exception) as exc:
        e.variables['b'] = (indago.VariableType.Categorical, 1.1, 1.2 ,1.3, 1.4)
        e._init_variables()
    assert "needs to be a tuple with exactly two items" in str(exc.value)

    e = Engine()
    e.variables = mixed_variables
    e._init_variables()
