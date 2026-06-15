import indago
import numpy as np


real_example_result_x = np.array((-1.3688474354012214e-08, 2.0544380419096342e-08, 2.6509004769081912e-08,
                                -1.961297968477993e-08, 7.529059331545795e-08, -5.1880164164685993e-08,
                                2.9547390312245625e-08, 3.158914019252279e-08))

def test_getting_started_real():

    import numpy as np

    # Evaluation function
    def goalfun(x):
        obj = np.sum(x ** 2)  # minimization objective
        constr1 = x[0] - x[1]  # constraint x_0 - x_1 <= 0
        constr2 = - np.sum(x)  # constraint sum x_i >= 0
        return obj, constr1, constr2

    # Initialize the chosen method
    from indago import PSO  # ...or any other Indago method
    optimizer = PSO()

    # Optimization variables settings
    optimizer.lb = -10  # lower bound, here given as scalar (equal for all variables)
    optimizer.ub = 10 + np.arange(8)  # upper bounds, here given as np.array (one bound value per each of the 8 variables)

    # Set evaluation function
    optimizer.evaluator = goalfun

    # Objectives and constraints settings
    optimizer.objectives = 1  # number of objectives (optional parameter, default objectives=1), this is obj in evaluation function
    optimizer.objective_labels = ['Squared sum minimization']  # labels for objectives (optional parameter, used in reporting)
    optimizer.constraints = 2  # number of constraints (optional parameter, default constraints=0), these are constr1 and constr2 in evaluation function
    optimizer.constraint_labels = ['Constraint 1', 'Constraint 2']  # labels for constraints (optional parameter, used in reporting)

    # Print optimizer parameters
    print(optimizer)  # not necessary, but useful for checking the setup of the optimizer

    # Run optimization
    result = optimizer.optimize(seed=0)  # (using default parameters of the method)

    # Extract results
    print(result.f)  # minimum of obj with constr1 and constr2 satisfied
    print(result.X)  # design vector at minimum (as tuple)

    assert np.isclose(result.f, 5.042330734013275e-14, atol=1e-10, rtol=0)
    assert np.isclose(np.array(result.X), real_example_result_x, atol=1e-10, rtol=0).all()


def test_getting_started_real_minimize():

    import numpy as np

    # Evaluation function
    def goalfun(x):
        obj = np.sum(x ** 2)  # minimization objective
        constr1 = x[0] - x[1]  # constraint x_0 - x_1 <= 0
        constr2 = - np.sum(x)  # constraint sum x_i >= 0
        return obj, constr1, constr2

    from indago import minimize
    X, f, O, C = minimize(goalfun, None, -10, 10 + np.arange(8), 'PSO',
                          objectives=1, objective_labels=['Squared sum minimization'],
                          constraints=2, constraint_labels=['Constraint 1', 'Constraint 2'],
                          seed=0)

    assert np.isclose(f, 5.042330734013275e-14, atol=1e-10, rtol=0)
    assert np.isclose(np.array(X), real_example_result_x, atol=1e-10, rtol=0).all()


mixed_example_result_x = ('down', 0.1, 2, 0.4583560838020153)

def test_getting_started_mixed():

    import math
    import numpy as np
    import indago

    # Optimization variables dictionary in the form of {name: (type, bounds | values)}
    VARS = {'type': (indago.VariableType.Categorical, ['up', 'down']),  # only strings allowed for Categorical variables
            'base': (indago.VariableType.RealDiscrete, [0.1, 1.2, 2.3, 3.4]),  # we provide a list of allowed values
            'n': (indago.VariableType.Integer, 2, 5),  # for non-discrete types we must provide bounds...
            'a': (indago.VariableType.Real, -3.3, 3.3)  # ...instead of allowed values
            }

    # Evaluation function
    def goalfun(x):
        type, base, n, a = x  # x is a tuple
        obj = base ** a + math.factorial(n) + a ** 2  # minimization objective
        match type:
            case 'up':
                obj += math.factorial(n - 1)
            case 'down':
                obj -= 1
            case _:
                obj = np.nan
        constr = base - n  # constraint base - n <= 0
        return obj, constr

    # Initialize the chosen method
    from indago import FWA  # ...or any other Indago method
    optimizer = FWA()

    # Optimization variables settings
    optimizer.variables = VARS

    # Set evaluation function
    optimizer.evaluator = goalfun

    # Objectives and constraints settings
    optimizer.objectives = 1  # number of objectives (optional parameter, default objectives=1), this is obj in evaluation function
    optimizer.constraints = 1  # number of constraints (optional parameter, default constraints=0), this is constr in evaluation function

    # Print optimizer parameters
    print(optimizer)  # not necessary, but useful for checking the setup of the optimizer

    # Run optimization
    result = optimizer.optimize(seed=0)  # (using default parameters of the method)

    # Extract results
    print(result.f)  # minimum of obj with constr satisfied
    print(result.X)  # design vector at minimum (as tuple)

    assert np.isclose(result.f, 1.5581421252669752, atol=1e-10, rtol=0)
    assert result.X == mixed_example_result_x


def test_getting_started_mixed_minimize():

    import math
    import numpy as np
    import indago

    # Optimization variables dictionary in the form of {name: (type, bounds | values)}
    VARS = {'type': (indago.VariableType.Categorical, ['up', 'down']),  # only strings allowed for Categorical variables
            'base': (indago.VariableType.RealDiscrete, [0.1, 1.2, 2.3, 3.4]),  # we provide a list of allowed values
            'n': (indago.VariableType.Integer, 2, 5),  # for non-discrete types we must provide bounds...
            'a': (indago.VariableType.Real, -3.3, 3.3)  # ...instead of allowed values
            }

    # Evaluation function
    def goalfun(x):
        type, base, n, a = x  # x is a tuple
        obj = base ** a + math.factorial(n) + a ** 2  # minimization objective
        match type:
            case 'up':
                obj += math.factorial(n - 1)
            case 'down':
                obj -= 1
            case _:
                obj = np.nan
        constr = base - n  # constraint base - n <= 0
        return obj, constr

    from indago import minimize
    X, f, O, C = minimize(goalfun, VARS, None, None, 'FWA',
                          objectives=1, constraints=1,
                          seed=0)

    assert np.isclose(f, 1.5581421252669752, atol=1e-10, rtol=0)
    assert X == mixed_example_result_x