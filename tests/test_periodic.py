import indago
import numpy as np


real_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealPeriodic, 0, 360) for i in range (0, 10)}
real_discrete_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.RealDiscretePeriodic, [float(_) for _ in range(0, 361)]) for i in range (0, 10)}
integer_periodic: indago.VariableDictType = {f'var{i}': (indago.VariableType.IntegerPeriodic, 0, 360) for i in range (0, 10)}

def goalfun(x):
    x = np.asarray(x)
    offset = np.arange(np.size(x))
    return np.sum(np.sin(np.deg2rad(x - offset - 90))) + np.size(x)

# import matplotlib.pyplot as plt
# goalfun = np.vectorize(goalfun)
# plt.plot(np.arange(0, 361), goalfun(np.arange(0, 361)))
# plt.show()

def test_periodic_1():

    for vars in [real_periodic, real_discrete_periodic, integer_periodic]:

        optimizer = indago.PSO()
        optimizer.variables = vars
        optimizer.evaluator = goalfun
        optimizer.max_evaluations = 100
        optimizer.optimize(seed=0)
        print(f"{vars['var0'][0]} solution: {optimizer.best.f}")
        assert optimizer.eval == 100


def test_periodic_2():
    vars = {'x1': (indago.VariableType.Real, -10, 10),
            'x2': (indago.VariableType.RealPeriodic, -20, 20),
            'x3': (indago.VariableType.RealDiscrete, np.linspace(-30, 30, 61)),
            'x4': (indago.VariableType.RealDiscretePeriodic, np.linspace(-40, 40, 81)),
            'x5': (indago.VariableType.Integer, -50, 50),
            'x6': (indago.VariableType.IntegerPeriodic, -60, 60),
            'x7': (indago.VariableType.Categorical, 'A B C'.split()),
            }
    optimizer = indago.PSO()
    optimizer.variables = vars
    candidate = indago.Candidate(variables=vars)
    print()


    # candidate.adjust()
    # print(f'{candidate._R=}')
    # print(f'{candidate.X=}')
    for attempt, r_value in enumerate([0.05, 0.5, 0.99, 1.05, 1.1, 1.9, -0.05, -0.10, -0.9]):
        # r_value = np.random.uniform(-3, 5, 1)
        candidate._R = r_value
        print(f'Attempt {attempt+1}, {r_value=}')
        # print(f'R: {candidate._R.tolist()}')
        # print(f'X: {list(candidate.X)}')

        for v, x, r in zip(vars.items(), candidate.X, candidate._R):
            var_name, (var_type, *var_options) = v
            print(f'  {var_name=}, {var_type=}, {x=}, {r=}')

            match var_type:
                case indago.VariableType.Real:
                    lb, ub = var_options
                    r = np.clip(r, 0, 1)
                    assert np.isclose(x, lb + r * (ub - lb)), f'Periodic mapping wrong for {var_type} variable {var_name}'

                case indago.VariableType.RealPeriodic:
                    lb, ub = var_options
                    r = r % 1.0
                    assert np.isclose(x, lb + r * (ub - lb)), f'Periodic mapping wrong for {var_type} variable {var_name}'

                case indago.VariableType.RealDiscrete:
                    x_dicrete = np.array(var_options[0])
                    x_min = x_dicrete[0] - 0.5 * (x_dicrete[1] - x_dicrete[0])
                    x_max = x_dicrete[-1] + 0.5 * (x_dicrete[-1] - x_dicrete[-2])
                    _x = x_min + np.clip(r, 0, 1) * (x_max - x_min)
                    i = np.argmin(np.abs(x_dicrete - x))
                    assert np.isclose(x, x_dicrete[i]), f'Periodic mapping wrong for {var_type} variable {var_name}'

                case indago.VariableType.RealDiscretePeriodic:
                    x_dicrete = np.array(var_options[0])
                    x_min = x_dicrete[0]
                    x_max = x_dicrete[-1]
                    _x = x_min + (r % 1.0) * (x_max - x_min)
                    i = np.argmin(np.abs(x_dicrete - _x))
                    assert np.isclose(x, x_dicrete[i]), f'Periodic mapping wrong for {var_type} variable {var_name}'

                case indago.VariableType.Integer:
                    lb, ub = var_options
                    r = np.clip(r, 0, 1)
                    _x = int(round(lb - 0.5 + r * (ub - lb + 1)))
                    assert np.isclose(x, _x), f'Periodic mapping wrong for {var_type} variable {var_name} ({x} vs {_x})'

                case indago.VariableType.IntegerPeriodic:
                    lb, ub = var_options
                    print(f'{lb=}, {ub=}')
                    r = r % 1.0
                    _x = int(round(lb + r * (ub - lb)))
                    assert np.isclose(x, _x), f'Periodic mapping wrong for {var_type} variable {var_name} ({x} vs {_x})'

                case indago.VariableType.Categorical:
                    assert x in var_options[0], f'Periodic mapping wrong for {var_type} variable {var_name}'
                case _:
                    raise NotImplementedError(f'Unknown variable type {var_type} for variable {var_name}')
