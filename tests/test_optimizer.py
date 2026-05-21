
import numpy as np
import timeit
from functools import partial

from indago.core._optimizer import Optimizer
from test_utils import *
timeit_runs = 5
timeit_dims = 50
timeit_evals = 1_000


def test_variables_initialization():
    optimizer = Optimizer()
    optimizer.variables['a'] = indago.VariableType.Real, -1, 5
    optimizer.variables['b'] = indago.VariableType.Real, 1, None
    optimizer.variables['c'] = indago.VariableType.Real, None, 0
    optimizer.variables['d'] = indago.VariableType.Real, None, None
    for i in range(10):
        optimizer.variables[f'theta{1 + i}'] = indago.VariableType.Real, -20, 20
    optimizer.variables['n1'] = indago.VariableType.Integer, 2, 10
    optimizer.variables['n2'] = indago.VariableType.Integer, -2, 5
    optimizer.variables['h'] = indago.VariableType.RealDiscrete, np.linspace(1, 2, 21)
    optimizer.variables['s'] = indago.VariableType.RealDiscrete, np.linspace(0, 0.5, 51)
    optimizer.variables['PumpType'] = indago.VariableType.Categorical, 'A B C'.split()

    candidate = indago.Candidate(optimizer.variables)
    print(candidate.X)


def test_real_rs():
    f = lambda x: 0.0

    optimizer = indago.RS()
    for i in range(8):
        optimizer.variables['r{i+1}'] = indago.VariableType.Real, 0, 5
    optimizer.evaluator = f
    # optimizer.max_evaluations = 10_000
    optimizer.optimize()
    assert optimizer.eval == optimizer.max_evaluations

def test_mixed_rs():
    f = lambda x: 0.0

    optimizer = indago.RS()
    optimizer.variables['r1'] = indago.VariableType.Real, -1.1, 5
    optimizer.variables['r2'] = indago.VariableType.Real, -1.2, 51.01
    optimizer.variables['r3'] = indago.VariableType.Real, -1.3, 5.4
    optimizer.variables['i1'] = indago.VariableType.Integer, -5, 0
    optimizer.variables['i2'] = indago.VariableType.Integer, 10, 20
    optimizer.variables['d1'] = indago.VariableType.RealDiscrete, np.linspace(0, 5, 51)
    optimizer.variables['d2'] = indago.VariableType.RealDiscrete, np.linspace(0, 5, 51)
    optimizer.variables['c'] = indago.VariableType.Categorical, ['A', 'B', 'C']

    optimizer.evaluator = f
    # optimizer.max_evaluations = 10_000
    optimizer.optimize()
    assert optimizer.eval == optimizer.max_evaluations

def test_timeit_rs():
    print()
    t_real = timeit.timeit(test_real_rs, number=timeit_runs)
    print(f'RS with 8D real-variable problem, {t_real=}')
    t_mixed = timeit.timeit(test_mixed_rs, number=timeit_runs)
    print(f'RS with 8D mixed-variable problem, {t_mixed=}')

def run_sampler(sampler, variables):
    f = lambda x: 0.0

    optimizer = indago.RS()
    optimizer.variables = variables

    optimizer.evaluator = f
    optimizer.max_evaluations = timeit_evals
    optimizer.sampler = sampler
    optimizer.optimize()
    # print(f'{sampler=}, {optimizer.dimensions=}')
    assert optimizer.eval == optimizer.max_evaluations

def test_samplers():
    print()
    print(f'{timeit_dims}-dimensional problems, {timeit_evals} evaluations')
    for kind in 'real mixed'.split():
        variables = generate_variables_dict(kind, dims=timeit_dims)
        # print(variables)
        for sampler in 'random halton sobol lhs'.split():
            # print(f'{sampler=}')
            run = partial(run_sampler, sampler=sampler, variables=variables)
            t_cpu = timeit.timeit(run, number=timeit_runs)
            print(f'{kind}-variable problem RS with {sampler=}, {t_cpu=}')

def test_bounds_api():
    optimizer = Optimizer()
    optimizer.lb = np.array([-1, -2, -3])
    optimizer.ub = np.array([1, 2, 3])
    optimizer.evaluator = lambda x: 0

    print(f'{optimizer.dimensions=}')

    optimizer._init_optimizer()
    print(f'{optimizer.dimensions=}')
    print(optimizer.variables)

    assert optimizer.dimensions == 3, 'Bounds API does not work (dimensions mismatch)'
    assert optimizer.lb.size == 3, 'Bounds API does not work (dimensions mismatch)'
    assert optimizer.ub.size == 3, 'Bounds API does not work (dimensions mismatch)'
    assert len(optimizer.variables) == 3, 'Bounds API does not work (dimensions mismatch)'


mixed_variables_set_2: indago.VariableDictType = {
             'var1': (indago.VariableType.RealDiscrete, [1.1, 1.2, 1.3, 1.4, 1.5]),  # Discrete (float for evaluator, int for optimizer)
             'var2': (indago.VariableType.Integer, 0, 4),  # Integer (both for optimizer and evaluator)
             'var3': (indago.VariableType.Categorical, ['A', 'B', 'C', 'D', 'E']),  # Category
             'var4': (indago.VariableType.RealDiscretePeriodic, [0.0, 0.25, 0.50, 0.75, 1.0]),
             'var5': (indago.VariableType.IntegerPeriodic, 0, 4),
             'var6': (indago.VariableType.RealDiscrete, [0, 1, 4, 8, 16]),
             'var7': (indago.VariableType.IntegerPeriodic, 0, 1),
             'var8': (indago.VariableType.IntegerPeriodic, 0, 2),
             'var9': (indago.VariableType.IntegerPeriodic, 0, 3),
             'var10': (indago.VariableType.IntegerPeriodic, 0, 4),
             'var11': (indago.VariableType.Real, -10, 5),
             'var12': (indago.VariableType.RealPeriodic, -24, 24),
                   }

def test_uniformity():
    for sampler in 'random halton sobol lhs'.split():
        uniformity_test(sampler)

def uniformity_test(sampler):
    optimizer = Optimizer()
    optimizer.variables = mixed_variables_set_2
    optimizer.evaluator = lambda x: x
    optimizer.sampler = 'random'
    optimizer._init_optimizer()

    XX = []
    RR = []
    n_samples = 100_000

    candidates = [indago.Candidate(variables=mixed_variables_set_2) for _ in range(n_samples)]
    optimizer._initialize_X(candidates)

    # for i, c in enumerate(candidates):
    #     print(f'{c}')

    # c = indago.Candidate(mixed_variables_set_2)
    for i, c in enumerate(candidates):
        # r = np.random.uniform(0, 1, len(c._variables))
        # c._R = r
        XX.append(c.X)

        c._X = c._X
        RR.append(c._R)


    print(f'{len(XX)=}')
    eps = 1e-2

    for i_var, (var_name, (var_type, *var_options)) in enumerate(mixed_variables_set_2.items()):

        print(f'{var_name=} {var_type} {var_options}')
        if var_type not in [indago.VariableType.Real, indago.VariableType.RealPeriodic]:
            # print(XX)
            x = np.asarray([sample[i_var] for sample in XX])
            r = np.asarray([sample[i_var] for sample in RR])
            # print(f'{x=}')
            unique_x = np.unique(x)
            unique_r = np.unique(r)
            # print(f'{unique.shape=}')

            cnt_sammples_x = {}
            cnt_sammples_r = {}
            if var_type in [indago.VariableType.RealDiscrete, indago.VariableType.RealDiscretePeriodic]:
                discrete_values = np.array(var_options[0])
                x_min = discrete_values[0] - 0.5 * (discrete_values[1] - discrete_values[0])
                x_max = discrete_values[-1] + 0.5 * (discrete_values[-1] - discrete_values[-2])
                x_mid = 0.5 * (discrete_values[1:] + discrete_values[:-1])
                print(f'{x_mid=}')
                x_bins = np.hstack([x_min, x_mid, x_max])
                x_density = x_bins[1:] - x_bins[:-1]
                x_density /= np.sum(x_density)
                print(f'{x_bins=}')
                print(f'{x_density=}')
            else:
                x_density = np.full(unique_x.size, 1/ unique_x.size)

            for u in unique_x:
                cnt_sammples_x[u] = np.sum(np.asarray(x) == u) / n_samples
            for u in unique_r:
                cnt_sammples_r[u] = np.sum(np.asarray(r) == u) / n_samples

            print(f'unique:  {len(cnt_sammples_x)}   {len(cnt_sammples_r)}')

            for (k, v), xd in zip(cnt_sammples_x.items(), x_density):
                print(f' x={k} share: {float(v):.6f} / {xd:.6f}')
                assert np.abs(v - xd) < eps, 'Nonuniform distribution detected!'
            for (k, v), xd in zip(cnt_sammples_r.items(), x_density):
                print(f' r={k} share: {float(v):.6f} / {xd:.6f}')
                assert np.abs(v - xd) < eps, 'Nonuniform distribution detected!'

        else:
            lb, ub = var_options
            x = np.asarray([sample[i_var] for sample in XX])
            cnt, bins = np.histogram(x, bins=np.linspace(lb, ub, 10))
            cnt = np.asarray(cnt)
            cnt = cnt / np.sum(cnt)
            print(cnt)
            assert np.max(np.abs(cnt - 1/9)) < eps, 'Nonuniform distribution detected!'


        # print(RR[-3:])
        print()

if __name__ == '__main__':
    test_bounds_api()