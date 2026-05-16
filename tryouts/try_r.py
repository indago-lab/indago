import indago
import numpy as np
from indago.core._optimizer import Optimizer

mixed_variables: indago.VariableDictType = {
             'var1': (indago.VariableType.RealDiscrete, [1.1, 1.2, 1.3, 1.4, 1.5]),  # Discrete (float for evaluator, int for optimizer)
             'var2': (indago.VariableType.Integer, 0, 4),  # Integer (both for optimizer and evaluator)
             'var3': (indago.VariableType.Categorical, ['A', 'B', 'C', 'D', 'E']),  # Category
             'var4': (indago.VariableType.RealDiscretePeriodic, [0.0, 0.25, 0.50, 0.75, 1.0]),
             'var5': (indago.VariableType.IntegerPeriodic, 0, 4),
             'var6': (indago.VariableType.RealDiscrete, [0, 2, 4, 8, 16]),
             'var7': (indago.VariableType.IntegerPeriodic, 0, 1),
             'var8': (indago.VariableType.IntegerPeriodic, 0, 2),
             'var9': (indago.VariableType.IntegerPeriodic, 0, 3),
             'var10': (indago.VariableType.IntegerPeriodic, 0, 4),
             'var11': (indago.VariableType.Real, -10, 5),
             'var12': (indago.VariableType.RealPeriodic, -24, 24),
                   }

def try_uniformity():
    optimizer = Optimizer()
    optimizer.variables = mixed_variables
    optimizer.evaluator = lambda x: x
    optimizer.sampler = 'random'
    optimizer._init_optimizer()

    XX = []
    RR = []
    n_samples = 100_000

    candidates = [indago.Candidate(variables=mixed_variables) for _ in range(n_samples)]
    optimizer._initialize_X(candidates)

    # for i, c in enumerate(candidates):
    #     print(f'{c}')

    # c = indago.Candidate(mixed_variables)
    for i, c in enumerate(candidates):
        # r = np.random.uniform(0, 1, len(c._variables))
        # c._R = r
        XX.append(c.X)

        c._X = c._X
        RR.append(c._R)


    print(f'{len(XX)=}')

    for i_var, (var_name, (var_type, *var_options)) in enumerate(mixed_variables.items()):

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
            # if unique_x.size > 20:
            #     continue
            for u in unique_x:
                cnt_sammples_x[u] = np.sum(np.asarray(x) == u) / n_samples
            for u in unique_r:
                cnt_sammples_r[u] = np.sum(np.asarray(r) == u) / n_samples

            print(f'unique:  {len(cnt_sammples_x)}   {len(cnt_sammples_r)}')

            eps = 0.01
            for k, v in cnt_sammples_x.items():
                print(f' x={k} share: {float(v)}')
                assert np.abs(v - 1/unique_x.size) < eps, 'Nonuniform distribution detected!'
            for k, v in cnt_sammples_r.items():
                print(f' r={k} share: {float(v)}')
                assert np.abs(v - 1/unique_r.size) < eps, 'Nonuniform distribution detected!'

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
    try_uniformity()