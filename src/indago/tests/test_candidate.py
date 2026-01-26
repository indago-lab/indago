import sys

from exceptiongroup import catch

sys.path.append('..')
import indago
import numpy as np
from copy import copy, deepcopy

def test_initialization():

    variables = {'var1': (indago.VariableType.Real, -100, 100), # Real (continuous) bounded
                 'var2': (indago.VariableType.Real, 0, None), # Real (continuous) semi-bounded
                 'var3': (indago.VariableType.Real, None, None), # Real (continuous) unbounded
                 'var4': (indago.VariableType.Discrete, [1.1, 1.2, 1.3, 1.4, 1.5]), # Discrete (float for evaluator, int for optimizer)
                 'var5': (indago.VariableType.Integer, 0, 10), # Integer (bot for optimizer and evaluator)
                 # 'var6': (indago.VariableType.Category, ['a', 'b', 'c', 'd', 'e']),  # Category
                 }

    candidate: indago.Candidate = indago.Candidate(variables=variables)

    assert np.all(np.isnan(candidate._X[:4])) and candidate._X[4] == 0
    assert np.all(np.isnan(candidate.X[:4])) and candidate.X[4] == 0.0

def test_list_X_format():

    variables = {'var1': (indago.VariableType.Real, -100, 100), # Real (continuous) bounded
                 'var2': (indago.VariableType.Real, 0, None), # Real (continuous) semi-bounded
                 'var3': (indago.VariableType.Real, None, None), # Real (continuous) unbounded
                 'var4': (indago.VariableType.Discrete, [1.1, 1.2, 1.3, 1.4, 1.5]), # Discrete (float for evaluator, int for optimizer)
                 'var5': (indago.VariableType.Integer, 0, 10), # Integer (bot for optimizer and evaluator)
                 'var6': (indago.VariableType.Category, ['a', 'b', 'c', 'd', 'e']),  # Category
                 }
    candidate: indago.Candidate = indago.Candidate(variables=variables, x_format=indago.XFormat.List)

    print(candidate.X)
    print([type(x) for x in candidate.X])
    assert np.all(np.isnan(candidate._X[:4])) and candidate._X[4] == 0 and candidate.X[5] == 'X'
    assert [type(x) == float for x in candidate._X[:4]] and type(candidate.X[4]) == int and type(candidate.X[5]) == str

def test_X_assing():

    variables = {'var1': (indago.VariableType.Real, -100, 100), # Real (continuous) bounded
                 'var2': (indago.VariableType.Real, 0, None), # Real (continuous) semi-bounded
                 'var3': (indago.VariableType.Real, None, None), # Real (continuous) unbounded
                 'var4': (indago.VariableType.Discrete, [1.1, 1.2, 1.3, 1.4, 1.5]), # Discrete (float for evaluator, int for optimizer)
                 'var5': (indago.VariableType.Integer, 0, 10), # Integer (bot for optimizer and evaluator)
                 'var6': (indago.VariableType.Category, ['a', 'b', 'c', 'd', 'e']),  # Category
                 }
    candidate: indago.Candidate = indago.Candidate(variables=variables, x_format=indago.XFormat.List)

    candidate.X = [0.11, 0.22, 0.33, 1.5, 7, 'Material A']
    print(candidate.X)

    for i, v in [(0, 1), (1, -3), (2, 'A'), (4, 0.33), (4, 'Z'), (5, -0.99), (5, 10)]:
        try:
            x = list(deepcopy(candidate.X))
            x[i] = v
            candidate.X = x
        except Exception as e:
            print(f' X[{i}] = {v} ({e})')
            continue

        assert False, f' X[{i}] = {v} should be allowed'

    try:
        candidate.X[0] = 'prvi'
        print(candidate.X)
    except Exception as e:
        print('exception caught, all good')
    else:
        assert False, f' X[{i}] = {v} should not be allowed'

if __name__ == '__main__':

    test_initialization()

    test_list_X_format()

    test_X_assing()