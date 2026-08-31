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
    e.variables = {'x1': (indago.VariableType.REAL, 2)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.REAL, 2.2, 3.3, 4.4)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.REAL, [2.2, 3.3])}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.REAL, '-10', '10')}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.REAL, -3.14, -22.1)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.REAL, -3.14, 22.1)}
    assert quick_validation(e.variables) == True

    e.variables = {'x1': (indago.VariableType.REAL, -np.inf, np.inf)}
    assert quick_validation(e.variables) == True

    e.variables = {'x1': (indago.VariableType.REAL, -np.nan, np.nan)}
    assert quick_validation(e.variables) == False

    # RealDiscrete variables
    e.variables = {'x1': (indago.VariableType.REAL_DISCRETE, 1, 5)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.REAL_DISCRETE, 1)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.REAL_DISCRETE, [1, 'a'])}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.REAL_DISCRETE, [1, np.nan, np.inf, -np.inf])}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.REAL_DISCRETE, [1, 2, 3, -4, 5])}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.REAL_DISCRETE, [1, 2, 3])}
    assert quick_validation(e.variables) == True

    # Integer variables
    e.variables = {'x1': (indago.VariableType.INTEGER, 0)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.INTEGER, -1.1, 2.2)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.INTEGER, 0.0, 10.0)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.INTEGER, 10, 2)}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.INTEGER, -1, 5)}
    assert quick_validation(e.variables) == True

    # Categorical variables
    e.variables = {'x1': (indago.VariableType.CATEGORICAL, 'a', 'b', 'c')}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.CATEGORICAL, 'abc')}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.CATEGORICAL, [1, 2, '3'])}
    assert quick_validation(e.variables) == False

    e.variables = {'x1': (indago.VariableType.CATEGORICAL, ['A', 'B', 'C', 'D'])}
    assert quick_validation(e.variables) == True


if __name__ == '__main__':
    test_variables()