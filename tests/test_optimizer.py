from turtledemo.paint import switchupdown

import numpy as np
import timeit
from functools import partial

import indago
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

    candidate.adjust()
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
