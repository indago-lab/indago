import indago
from indago.core._engine import Engine
from indago.utils._validation import validate_variables
import numpy as np

def quick_validation(variables):
    valid, log = validate_variables(variables)
    for line in log:
        print(line[0], line[1])

    return valid

def test_variables():

    e = Engine()
    e.variables = None
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (1, 2, 3)}
    assert quick_validation(e.variables) == False

    # Real variables
    e.variables = {'x1': (indago.VariableType.Real, 2)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Real, 2.2, 3.3, 4.4)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Real, [2.2, 3.3])}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Real, '-10', '10')}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Real, -3.14, -22.1)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Real, -3.14, 22.1)}
    assert quick_validation(e.variables) == True

    e.variables = {'x1': (indago.VariableType.Real, -np.inf, np.inf)}
    assert quick_validation(e.variables) == True

    e.variables = {'x1': (indago.VariableType.Real, -np.nan, np.nan)}
    assert quick_validation(e.variables) == False

    # RealDiscrete variables
    e.variables = {'x1': (indago.VariableType.RealDiscrete, 1, 5)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.RealDiscrete, 1)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.RealDiscrete, [1, 'a'])}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.RealDiscrete, [1, np.nan, np.inf, -np.inf])}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.RealDiscrete, [1, 2, 3, -4, 5])}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.RealDiscrete, [1, 2, 3])}
    assert quick_validation(e.variables) == True

    # Integer variables
    e.variables = {'x1': (indago.VariableType.Integer, 0)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Integer, -1.1, 2.2)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Integer, 0.0, 10.0)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Integer, 10, 2)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Integer, -1, 5)}
    assert quick_validation(e.variables) == True

    # Categorical variables
    e.variables = {'x1': (indago.VariableType.Categorical, 'a', 'b', 'c')}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Categorical, 'abc')}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Categorical, [1, 2, '3'])}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.Categorical, ['A', 'B', 'C', 'D'])}
    assert quick_validation(e.variables) == True


if __name__ == '__main__':
    test_variables()