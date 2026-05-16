import indago
import numpy as np

mixed_variables: indago.VariableDictType = {  # Real (continuous) unbounded
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
                   }

def try_uniformity():
    c = indago.Candidate(mixed_variables)

    XX = []
    RR = []
    n_samples = 100_000
    for i in range(n_samples):
        r = np.random.uniform(0, 1, len(c._variables))
        c._R = r
        XX.append(c.X)

        c._X = c._X
        RR.append(c._R)


    print(f'{len(XX)=}')

    for i_var, var in enumerate(mixed_variables):

        if var not in [indago.VariableType.Real, indago.VariableType.RealPeriodic]:
            # print(XX)
            x = np.asarray([sample[i_var] for sample in XX])
            r = np.asarray([sample[i_var] for sample in RR])
            # print(f'{x=}')
            unique_x = np.unique(x)
            unique_r = np.unique(r)
            # print(f'{unique.shape=}')

            cnt_sammples_x = {}
            cnt_sammples_r = {}
            # if unique.size < 10:
            for u in unique_x:
                cnt_sammples_x[u] = np.sum(np.asarray(x) == u) / n_samples
            for u in unique_r:
                cnt_sammples_r[u] = np.sum(np.asarray(r) == u) / n_samples

                # cnt_sammples[var] /= n_samples

        print(f'{var=} {mixed_variables[var][0]} {len(cnt_sammples_x)} {len(cnt_sammples_r)}')
        for k, v in cnt_sammples_x.items():
            print(f' x={k} share: {float(v)}')
        for k, v in cnt_sammples_r.items():
            print(f' r={k} share: {float(v)}')
        # print(RR[-3:])
        print()

if __name__ == '__main__':
    try_uniformity()