
import indago
import numpy as np
from copy import copy, deepcopy
from test_utils import mixed_variables

def test_initialization():

    vars = copy(mixed_variables)
    vars.pop('var6')
    candidate: indago.Candidate = indago.Candidate(variables=vars)

    assert np.all(np.isnan(candidate._X[:4])) and candidate._X[4] == 0
    assert np.all(np.isnan(candidate.X[:4])) and candidate.X[4] == 0.0

def test_list_X_format():
    print()
    candidate: indago.Candidate = indago.Candidate(variables=mixed_variables, x_format=indago.XFormat.List)

    print(candidate.X)
    print([type(x) for x in candidate.X])
    assert np.all(np.isnan(candidate._X[:4])) and candidate._X[4] == 0 and candidate.X[5] == 'X'
    assert [type(x) == float for x in candidate._X[:4]] and type(candidate.X[4]) == int and type(candidate.X[5]) == str

def test_x_assign():
    print()
    candidate: indago.Candidate = indago.Candidate(variables=mixed_variables, x_format=indago.XFormat.Tuple)

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
        print(candidate.X)
        candidate.X[0] = 'prvi'
        print(candidate.X)
    except Exception as e:
        print('exception caught, all good')
    else:
        assert False, f' X[{i}] = {v} should not be allowed'

def test_candidate_copy():
    # print()
    # c1: indago.Candidate = indago.Candidate(variables=variables, x_format=indago.XFormat.Tuple)
    # print(c1.X)
    # print(np.copy(c1.X))
    # c2: indago.Candidate = c1.copy()
    # print(c2.X)

    c1 = indago.Candidate(mixed_variables)
    c1.adjust()
    c1.O = np.array([1.1, 0.01])
    c1.f = 1.11
    c2 = c1.copy()
    assert c1.f == c2.f, 'Candidate.copy() does not work. Copy is not the same as original'
    assert c1 == c2, 'Candidate.copy() does not work. Copy is not the same as original'
    assert c1.X == c2.X, 'Candidate.copy() does not work. Copy is not the same as original'

    x = c1._get_x_as_list()
    x[0] = -99.99
    c1.X = x

    # assert c1 != c2, 'Candidate.copy() does not work. Copy points to original!'
    assert c1.X != c2.X, 'Candidate.copy() does not work. Copy points to original!'

def test_adjust():

    c1: indago.Candidate = indago.Candidate(variables=mixed_variables, x_format=indago.XFormat.Tuple)
    assert c1.adjust()

    X = c1._get_x_as_list()
    X[0] = 1.2345
    c1.X = X
    assert not c1.adjust()
    assert c1.X[0] == X[0]

    X = c1._get_x_as_list()
    X[0] = -987.654
    c1.X = X
    assert c1.adjust()
    assert not c1.X[0] == X[0]

    X = c1._get_x_as_list()
    X[1] = -0.3
    c1.X = X
    assert c1.adjust()
    assert not c1.X[1] == X[1]

    X = c1._get_x_as_list()
    X[2] = 365.25
    c1.X = X
    assert not c1.adjust()
    assert c1.X[1] == X[1]

    X = c1._get_x_as_list()
    X[3] = 8.2
    c1.X = X
    assert c1.adjust()
    assert not c1.X[3] == X[3]

    X = c1._get_x_as_list()
    X[4] = 7
    c1.X = X
    c1.adjust()
    assert c1.X[4] == X[4]

    X = c1._get_x_as_list()
    X[4] = 11
    c1.X = X
    assert c1.adjust()
    assert not c1.X[4] == X[4]

    X = c1._get_x_as_list()
    X[5] = 'D'
    c1.X = X
    assert not c1.adjust()
    assert c1.X[5] == X[5]

    X = c1._get_x_as_list()
    X[5] = 'None'
    c1.X = X
    assert c1.adjust()
    assert not c1.X[5] == X[5]

def test_set_rel_x():
    mixed_variables.pop('var2')
    mixed_variables.pop('var3')
    c = indago.Candidate(mixed_variables)
    n = len(c._variables)

    r = np.random.uniform(0, 1, n)
    c._set_X_rel(r)
    print(f'{r=}')
    print(f'{c.X=}')

    lb = []
    ub = []
    for var_name, (var_type, *var_options) in mixed_variables.items():
        match var_type:
            case indago.VariableType.Real:
                lb.append(var_options[0])
                ub.append(var_options[1])
            case indago.VariableType.RealDiscrete:
                lb.append(var_options[0][0])
                ub.append(var_options[0][-1])
            case indago.VariableType.Integer:
                lb.append(var_options[0])
                ub.append(var_options[1])
            case indago.VariableType.Categorical:
                lb.append(var_options[0][0])
                ub.append(var_options[0][-1])
            case _:
                raise NotImplementedError
    lb = tuple(lb)
    ub = tuple(ub)

    r = np.zeros(n)
    c._set_X_rel(r)
    print(f'{r=}')
    print(f'{c.X=}')
    print(f'{lb=}')
    assert c.X == lb, 'Error in relative vale assignment'

    r = np.ones(n)
    c._set_X_rel(r)
    print(f'{r=}')
    print(f'{c.X=}')
    print(f'{ub=}')
    assert c.X == ub, 'Error in relative vale assignment'

if __name__ == '__main__':
    test_set_rel_x()
