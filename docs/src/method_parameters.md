<script src="https://kit.fontawesome.com/067013af35.js" crossorigin="anonymous"></script>

# Method parameters

Method parameters are listed and explained here.

## Overview of methods and their variants

Avaliable methods and their variants are summarized in the following table.

| Method                                       | Variant                                      | Supports constraints                       | Remarks                                                                                                                                                   |
|----------------------------------------------|----------------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Particle Swarm Optimization (PSO)**        | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> |                                                                                                                                                           |
|                                              | 'TVAC'                                       | <i class="fa-regular fa-circle-check"></i> | User must provide at least one of the following stopping criteria: *optimizer.max_iterations*, *optimizer.max_evaluations*, *optimizer.max_elapsed_time*. |
|                                              | 'Chaotic'                                    | <i class="fa-regular fa-circle-check"></i> |                                                                                                                                                           |
| **Fireworks Algorithm (FWA)**                | 'Vanilla'                                    | <i class="fa-solid fa-xmark"></i>          |                                                                                                                                                           |
|                                              | 'Rank' <i class="fa-regular fa-star"></i>    | <i class="fa-regular fa-circle-check"></i> |                                                                                                                                                           |
| **Squirrel Search Algorithm (SSA)**          | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> | Showing poor performance on unbound problems.                                                                                                             |
| **Differential Evolution (DE)**              | 'SHADE'                                      | <i class="fa-solid fa-xmark"></i>          |                                                                                                                                                           |
|                                              | 'LSHADE' <i class="fa-regular fa-star"></i>  | <i class="fa-solid fa-xmark"></i>          | User must provide at least one of the following stopping criteria: *optimizer.max_iterations*, *optimizer.max_evaluations*, *optimizer.max_elapsed_time*. |
| **Bat Algorithm (BA)**                       | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> | Showing poor performance on unbound problems.                                                                                                             |
| **Electromagnetic Field Optimization (EFO)** | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> | Significantly slower than other methods. Does not support parallel evaluation.                                                                            |
| **Manta Ray Foraging Optimization (MRFO)**   | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> | User must provide at least one of the following stopping criteria: *optimizer.max_iterations*, *optimizer.max_evaluations*, *optimizer.max_elapsed_time*. |
| **Artificial Bee Colony (ABC)**              | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> | Showing poor performance on unbound problems.                                                                                                             |
|                                              | 'FullyEmployed'                              | <i class="fa-regular fa-circle-check"></i> | Showing poor performance on unbound problems.                                                                                                             |
| **Grey Wolf Optimizer (GWO)**                | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> | User must provide at least one of the following stopping criteria: *optimizer.max_iterations*, *optimizer.max_evaluations*, *optimizer.max_elapsed_time*. |
| **Heap-Based Optimizer (HBO)**               | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> | User must provide at least one of the following stopping criteria: *optimizer.max_iterations*, *optimizer.max_evaluations*, *optimizer.max_elapsed_time*. |
|                                              | 'Dynamic'                                    | <i class="fa-regular fa-circle-check"></i> | User must provide at least one of the following stopping criteria: *optimizer.max_iterations*, *optimizer.max_evaluations*, *optimizer.max_elapsed_time*. |
| **Controlled Random Search (CRS)**           | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> | Slower than other methods. Does not support parallel evaluation.                                                                                          |
| **Nelder-Mead (NM)**                         | 'Vanilla'                                    | <i class="fa-regular fa-circle-check"></i> |                                                                                                                                                           |
|                                              | 'GaoHan' <i class="fa-regular fa-star"></i>  | <i class="fa-regular fa-circle-check"></i> |                                                                                                                                                           |
| **Multi-Scale Grid Descent (MSGD)**          | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> | Does not support unbound optimization.                                                                                                                    |
| **Random Search (RS)**                       | 'Vanilla' <i class="fa-regular fa-star"></i> | <i class="fa-regular fa-circle-check"></i> | Showing poor performance on unbound optimization.                                                                                                         |

Legend: <i class="fa-regular fa-star"></i> Default variant, <i class="fa-regular fa-circle-check"></i> Constraints supported, <i class="fa-solid fa-xmark"></i> Constraints not supported

Specific parameters for each of the available methods and their variants are listed and explained below.

## Particle Swarm Optimization (PSO)

| Variant       | Parameter      | Allowed values                                      | Range      | Default                | Description                                                |
|---------------|----------------|-----------------------------------------------------|------------|------------------------|------------------------------------------------------------|
| **all**       | swarm_size     | (int)                                               | [1, -]     | max (10, *dimensions*) | Number of PSO particles                                    |
|               | inertia        | (float)                                             | [0.5, 1.0] | 0.72                   | Inertia weight                                             |
|               |                | 'LDIW'                                              |            |                        | Linearly decreasing inertia weight (from 1.0 to 0.4)       |
|               |                | 'HSIW'                                              |            |                        | Half sinusoidal inertia weight (from 0.5 to 0.75 and back) |
|               |                | 'anakatabatic'                                      |            |                        | Adaptive inertia weight technique (Družeta and Ivić, 2020) |
| **'Vanilla'** | cognitive_rate | (float)                                             | [0.0, 2.0] | 1.0                    | PSO parameter also known as c1                             |
|               | social_rate    | (float)                                             | [0.0, 2.0] | 1.0                    | PSO parameter also known as c2                             |
|               | akb_model      | 'Languid', 'TipsySpider', 'FlyingStork', 'MessyTie' |            | 'Languid'              | Secondary parameter when using *inertia*='anakatabatic'    |
| **'TVAC'**    | akb_model      | 'Languid', 'RightwardPeaks', 'OrigamiSnake'         |            | 'Languid'              | Secondary parameter when using *inertia*='anakatabatic'    |
| **'Chaotic'** | max_cls_it     | (int)                                               | [0, -]     | 10                     | Maximum number of chaotic local search iterations          |
|               | chaotic_elite  | (float)                                             | [0.0, 1.0] | 0.2                    | Elite part of the swarm, immune to reinitialization        |
|               | akb_model      | 'Languid'                                           |            | 'Languid'              | Secondary parameter when using *inertia*='anakatabatic'    |

## Fireworks Algorithm (FWA)

| Variant | Parameter | Allowed values | Range  | Default          | Description                        |
|---------|-----------|----------------|--------|------------------|------------------------------------|
| **all** | n         | (int)          | [1, -] | *dimensions*     | Number of fireworks                |
|         | m1        | (int)          | [1, -] | *dimensions* / 2 | Number of explosion sparks         |
|         | m2        | (int)          | [1, -] | *dimensions* / 2 | Number of mutation sparks          |

## Squirrel Search Algorithm (SSA)

| Variant       | Parameter | Allowed values     | Range      | Default                    | Description                        |
|---------------|-----------|--------------------|------------|----------------------------|------------------------------------|
| **'Vanilla'** | pop_size  | (int)              | [1, -]     | max (20, 2 * *dimensions*) | Number of SSA squirrels            |
|               | ata       | (float)            | [0.0, 1.0] | 0.5                        | Acorn tree attraction              |
|               | p_pred    | (float)            | [-, -]     | 0.1                        | Predator presence probability      |
|               | c_glide   | (float)            | [-, -]     | 1.9                        | Gliding constant                   |
|               | gd_lim    | (list of 2 floats) |            | [0.5, 1.11]                | Gliding distance limits (min, max) |

## Differential Evolution (DE)

| Variant | Parameter  | Allowed values | Range      | Default                   | Description                  |
|---------|------------|----------------|------------|---------------------------|------------------------------|
| **all** | pop_init   | (int)          | [1, -]     | max(30, 5 * *dimensions*) | Initial population size      |
|         | f_archive  | (float)        | [0.0, -]   | 2.6                       | External archive size factor |
|         | hist_size  | (int)          | [1, -]     | 6                         | Size of historical memory    |
|         | p_mutation | (float)        | [0.0, 1.0] | 0.11                      | Mutation probability         |

## Bat Algorithm (BA)

| Variant       | Parameter  | Allowed values     | Range  | Default                | Description                |
|---------------|------------|--------------------|--------|------------------------|----------------------------|
| **'Vanilla'** | pop_size   | (int)              | [1, -] | max (15, *dimensions*) | Number of BA bats          |
|               | loudness   | (float)            | [-, -] | 1.0                    | Loudness                   |
|               | pulse_rate | (float)            | [-, -] | 0.001                  | Pulse rate                 |
|               | alpha      | (float)            | [-, -] | 0.9                    | Alpha                      |
|               | gamma      | (float)            | [-, -] | 0.1                    | Gamma                      |
|               | freq_range | (list of 2 floats) |        | [0.0, 1.0]             | Frequency range (min, max) |

## Electromagnetic Field Optimization (EFO)

| Variant       | Parameter | Allowed values | Range       | Default                | Description                                                                |
|---------------|-----------|----------------|-------------|------------------------|----------------------------------------------------------------------------|
| **'Vanilla'** | pop_size  | (int)          | [1, -]      | max (50, *dimensions*) | EFO population size                                                        |
|               | R_rate    | (float)        | [0.1, 0.4]  | 0.25                   | Probability of changing one EM of generated particle with a random EM      |
|               | Ps_rate   | (float)        | [0.1, 0.4]  | 0.25                   | Probability of selecting EMs of generated particle from the positive field |
|               | P_field   | (float)        | [0.05, 0.1] | 0.075                  | Portion of population which belongs to positive field                      |
|               | N_field   | (float)        | [0.4, 0.5]  | 0.45                   | Portion of population which belongs to negative field                      |

## Manta Ray Foraging Optimization (MRFO)

| Variant       | Parameter | Allowed values     | Range  | Default                | Description           |
|---------------|-----------|--------------------|--------|------------------------|-----------------------|
| **'Vanilla'** | pop_size  | (int)              | [1, -] | max (10, *dimensions*) | Number of MRFO mantas |
|               | f_som     | (float)            | [-, -] | 2.0                    | Somersault factor     |

## Artificial Bee Colony (ABC)

| Variant | Parameter   | Allowed values | Range  | Default                       | Description                                                                        |
|---------|-------------|----------------|--------|-------------------------------|------------------------------------------------------------------------------------|
| **all** | pop_size    | (int)          | [2, -] | max (10, 2 * *dimensions*)    | Total number of bees                                                               |
|         | trial_limit | (int)          | [1, -] | *pop_size* * *dimensions* / 2 | Number of times a bee may try to find a better solution before it is reinitialized |

## Grey Wolf Optimizer (GWO)

| Variant       | Parameter | Allowed values | Range  | Default                | Description          |
|---------------|-----------|----------------|--------|------------------------|----------------------|
| **'Vanilla'** | pop_size  | (int)          | [5, -] | max (10, *dimensions*) | Number of GWO wolves |

## Heap-Based Optimizer (HBO)

| Variant | Parameter | Allowed values | Range  | Default                     | Description                                 |
|---------|-----------|----------------|--------|-----------------------------|---------------------------------------------|
| **all** | team_size | (int)          | [2, -] | max (*dimensions* ^ 0.5, 3) | Number of workers under one boss            |
|         | levels    | (int)          | [2, -] | 3                           | Number of levels in the corporate structure |

## Controlled Random Search (CRS)

| Variant       | Parameter | Allowed values | Range  | Default | Description                                                                       |
|---------------|-----------|----------------|--------|---------|-----------------------------------------------------------------------------------|
| **'Vanilla'** | pop_scale | (float)        | [1, -] | 10      | Population scale. Population size iz computed as *pop_scale* * (*dimensions* + 1) |

## Nelder-Mead (NM)

| Variant       | Parameter | Allowed values | Range      | Default | Description                       |
|---------------|-----------|----------------|------------|---------|-----------------------------------|
| **all**       | init_step | (float)        | [0.0, 1.0] | 0.4     | Relative size of initial polytope |
| **'Vanilla'** | alpha     | (float)        | [0.0, -]   | 1.0     | Reflection factor                 |
|               | gamma     | (float)        | [0.0, -]   | 2.0     | Expansion factor                  |
|               | rho       | (float)        | [0.0, -]   | 0.5     | Contraction factor                |
|               | sigma     | (float)        | [0.0, -]   | 0.5     | Reduction factor                  |

## Multi-Scale Grid Descent (MSGD)

| Variant       | Parameter | Allowed values   | Range  | Default | Description                                                                                    |
|---------------|-----------|------------------|--------|---------|------------------------------------------------------------------------------------------------|
| **'Vanilla'** | divisions | (ndarray or int) | [1, -] | 10      | Number of grid divisions in each dimension                                                     |
|               | base      | (int)            | [1, -] | 4       | Base of the exponential grid refining (how many times is grid refined in each scale increment) |
|               | max_scale | (int)            | [1, -] | 15      | Maximum scale (maximum number of grid refinements)                                             |

## Random Search (RS)

| Variant       | Parameter  | Allowed values | Range    | Default      | Description                         |
|---------------|------------|----------------|----------|--------------|-------------------------------------|
| **'Vanilla'** | batch_size | (int)          | [1, -]   | *dimensions* | Number of evaluations per iteration |
