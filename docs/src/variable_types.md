<script src="https://kit.fontawesome.com/067013af35.js" crossorigin="anonymous"></script>

# Variable types

To use Indago variable types, you need to define them in the `optimizer.variables` dictionary (otherwise Indago will expect all variables to be real continuous). A key-value pair for this dictionary is structured as `variable_name: (variable_type, options)`, for example `‘diameter’: (indago.VariableType.RealDiscrete, [2.5, 3.75, 5.0, 7.5, 10.0])`.

For periodic types, the evaluation function needs to return the same objective(s and constraints) for the lower and upper bound, or the first and last discrete value. When using Categorical variables, bear in mind that having many of them makes your problem highly combinatorial, and most Indago methods are not efficient in solving such problems.

You can mix the variable types however you like. Keep in mind that the evaluation function needs to take in variables in the exact order as they are given in the variables dictionary.

## Overview of supported variable types

Available variable types are given in the following table.

| Variable type                         | Description                                                         | Options                                                 | Remarks                                                                 |
|---------------------------------------|---------------------------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------|
| **VariableType.Real**                 | real continuous variable                                            | `[lb: float, ub: float]`                                |                                                                         |
| **VariableType.RealPeriodic**         | real continuous variable with periodic bounds (`f(lb) == f(ub)`)    | `[lb: float, ub: float]`                                | Some methods are ill-equipped for optimizing Periodic variables.        |
| **VariableType.RealDiscrete**         | list of discrete real values                                        | `[x_1: float, x_2: float, x_3: float, ...]`             |                                                                         |
| **VariableType.RealDiscretePeriodic** | "circular" list (`x[-1] == x[0]`) of discrete real values           | `[x_1: float, x_2: float, x_3: float, ..., x_1: float]` | Some methods are ill-equipped for optimizing Periodic variables.        |
| **VariableType.Integer**              | integer continuous variable                                         | `[lb: int, ub: int]`                                    |                                                                         |
| **VariableType.IntegerPeriodic**      | integer continuous variable with periodic bounds (`f(lb) == f(ub)`) | `[lb: int, ub: int]`                                    | Some methods are ill-equipped for optimizing Periodic variables.        |
| **VariableType.Categorical**          | list of strings (no ordinality)                                     | `[x_1: str, x_2: str, x_3: str, ...]`                   | Most methods are not efficient when dealing with multiple Categoricals. |
