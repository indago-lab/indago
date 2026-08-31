from unittest import case

import pytest
import numpy as np

import indago
from indago.core._engine import Engine
from indago.core._enums import VariableType
from indago.core._optimizer import Optimizer
from test_utils import mixed_variables

def test_optimizer_inheritance():

    o = Optimizer()
    assert isinstance(o, Engine)

def test_variables_initialization():
    e = Engine()
    e.variables['a'] = (indago.VariableType.REAL, -2, 2.2)
    e.variables['b'] = (indago.VariableType.REAL, -9.1, 20)
    e._init_variables()

    assert np.all(e.lb == [-2, -9.1])
    assert np.all(e.ub == [2.2, 20])

    e.variables['c'] = (indago.VariableType.INTEGER, 2, 8)
    e._init_variables()
    assert e.lb is None and e.ub is None

def test_variables_initialization_unbounded():
    e = Engine()
    e.variables['x1'] = (indago.VariableType.REAL, -np.inf, np.inf)
    e.variables['x2'] = (indago.VariableType.REAL, -9.9, np.inf)
    e.variables['x3'] = (indago.VariableType.REAL, -np.inf, 11.1)
    e._init_variables()

    assert np.all(e.lb == np.array([-np.inf, -9.9, -np.inf]))
    assert np.all(e.ub == np.array([np.inf, np.inf, 11.1]))

    # Mixed design vector
    e = Engine()
    e.variables['x1'] = (indago.VariableType.REAL, -10.0, 10.0)
    e.variables['x2'] = (indago.VariableType.REAL_DISCRETE, [0.0, 0.1, 0.2])
    e._init_variables()
    assert e.lb is None and e.ub is None

    e = Engine()
    e.variables['x1'] = (indago.VariableType.REAL, -10.0, 10.0)
    e.variables['x2'] = (indago.VariableType.INTEGER, 10, 19)
    e._init_variables()
    assert e.lb is None and e.ub is None

    e = Engine()
    e.variables['x1'] = (indago.VariableType.REAL, -10.0, 10.0)
    e.variables['x2'] = (indago.VariableType.CATEGORICAL, 'abc'.split())
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
        e.variables['a'] = (indago.VariableType.REAL, 1)
        e._init_variables()
    assert "(indago.VariableType.REAL | indago.VariableType.REAL_PERIODIC, lb, ub)" in str(exc.value)

    e = Engine()
    with pytest.raises(Exception) as exc:
        e.variables['b'] = (indago.VariableType.REAL_DISCRETE, 1.1, 1.2 , 1.3, 1.4)
        e._init_variables()
    assert "(indago.VariableType.REAL_DISCRETE | indago.VariableType.REAL_DISCRETE_PERIODIC, list_of_discrete_values)" in str(exc.value)

    e = Engine()
    with pytest.raises(Exception) as exc:
        e.variables['a'] = (indago.VariableType.INTEGER, 1, 2, 3)
        e._init_variables()
    assert "(indago.VariableType.INTEGER | indago.VariableType.INTEGER_PERIODIC, lb, ub)" in str(exc.value)

    e = Engine()
    with pytest.raises(Exception) as exc:
        e.variables['b'] = (indago.VariableType.CATEGORICAL, 1.1, 1.2 , 1.3, 1.4)
        e._init_variables()
    assert "(indago.VariableType.CATEGORICAL, list_of_string_values)" in str(exc.value)

    e = Engine()
    e.variables = mixed_variables
    e._init_variables()


def test_init_utils():
    e = Engine()
    for i in range(20):

        var_type = np.random.choice(indago.VariableType)
        match var_type:
            case indago.VariableType.REAL | indago.VariableType.REAL_PERIODIC:
                e.variables[f'x{i}'] = (var_type, -10, 10)
            case indago.VariableType.REAL_DISCRETE | indago.VariableType.REAL_DISCRETE_PERIODIC:
                e.variables[f'x{i}'] = (var_type, np.linspace(-10, 10, 41))
            case indago.VariableType.INTEGER | indago.VariableType.INTEGER_PERIODIC:
                e.variables[f'x{i}'] = (var_type, -10, 10)
            case indago.VariableType.CATEGORICAL:
                e.variables[f'x{i}'] = (var_type, 'A B C D E F'.split())
            case _:
                raise NotImplementedError(f'Unknown variable type {var_type}')

    e._init_utils()
    # print(e._var_indices)
    # print(e._var_indices[indago.VariableType.CATEGORICAL])

    c = indago.Candidate(e.variables)
    c._R = 0.5
    X = list(c.X)
    val_cat = 'F'
    for i in e._var_indices[indago.VariableType.CATEGORICAL]:
        X[i] = val_cat

    val_real = 0.998877
    for i in e._var_indices[indago.VariableType.REAL] + e._var_indices[indago.VariableType.REAL_PERIODIC]:
        X[i] = val_real

    c.X = X
    print(c.X)

    for i_var, (var_name, (var_type, *_)) in enumerate(e.variables.items()):
        if var_type in [indago.VariableType.REAL, indago.VariableType.REAL_PERIODIC]:
            assert c.X[i_var] == val_real, 'Wrong value for variable {var_name}'
        elif var_type in [indago.VariableType.CATEGORICAL]:
            assert c.X[i_var] == val_cat, 'Wrong value for variable {var_name}'
