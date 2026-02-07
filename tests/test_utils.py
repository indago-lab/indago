import indago
import numpy as np

mixed_variables = {'var1': (indago.VariableType.Real, -100, 100),  # Real (continuous) bounded
             # 'var2': (indago.VariableType.Real, 0, None),  # Real (continuous) semi-bounded
             # 'var3': (indago.VariableType.Real, None, None),  # Real (continuous) unbounded
             'var4': (indago.VariableType.RealDiscrete, [1.1, 1.2, 1.3, 1.4, 1.5]),  # Discrete (float for evaluator, int for optimizer)
             'var5': (indago.VariableType.Integer, 0, 10),  # Integer (bot for optimizer and evaluator)
             'var6': (indago.VariableType.Categorical, ['A', 'B', 'C', 'D', 'E']),  # Category
                   }

real_variables_10D = {f'x{i}': (indago.VariableType.Real, -100, 100) for i in range (10)}

def generate_variables_dict(kind, dims):
    variables_dict = {}
    if kind == 'real':
        for i in range(dims):
            variables_dict[f'x{i+1}'] = indago.VariableType.Real, -20, 20

    elif kind == 'mixed' or kind == 'numeric':
        for i in range(dims):
            var_type = np.random.choice(indago.VariableType)
            if kind == 'numeric':
                while var_type == indago.VariableType.Categorical:
                    var_type = np.random.choice(indago.VariableType)
            match var_type:
                case indago.VariableType.Real:
                    variables_dict[f'x{i+1}'] = indago.VariableType.Real, -20, 20
                case indago.VariableType.Integer:
                    variables_dict[f'x{i+1}'] = indago.VariableType.Integer, -20, 20
                case indago.VariableType.RealDiscrete:
                    variables_dict[f'x{i+1}'] = indago.VariableType.RealDiscrete, np.linspace(0, 5, 51)
                case indago.VariableType.Categorical:
                    variables_dict[f'x{i+1}'] = indago.VariableType.Categorical, 'A B C D E F'.split()

    else:
        raise ValueError(f'Unknown kind {kind}')

    return variables_dict

def real_function(x):
    x = np.asarray(x)
    f = np.sum((x - np.arange(x.size)) ** 2)
    # print(f'{x=}, {f=}')
    return f