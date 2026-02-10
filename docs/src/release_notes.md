<script src="https://kit.fontawesome.com/067013af35.js" crossorigin="anonymous"></script>

# Release notes

Most important changes in Indago releases are listed here. Breaking changes are indicated by the <i class="fa-solid fa-triangle-exclamation"></i> sign.

***

## Indago 0.6.0

Released on yyyy-mm-dd

### New features
- **New methods: Heap-Based Optimizer (HBO), Controlled Random Search (CRS)**
- New samplers for generating initial population positions (`Optimizer.sampler`): Sobol (`'sobol'`) and LatinHypercube (`'lhs'`)
### Improvements
- **IndagoBench25 results overview and corresponding guidelines added to documentation**
- Template file for new methods included in the methods library (`_new_method_template.py`)
- Anakatabatic PSO variants are now using `scipy.interpolate.make_interp_spline` instead of legacy function `interp1d`
- Literature reference for MSGD provided in documentation
- LSHADE set as the default DE variant (performs much better than SHADE)
- `Optimizer.scatter_method` renamed to `Optimizer.sampler` <i class="fa-solid fa-triangle-exclamation"></i>
### Bug fixes
- Fixed the problem of EFO crashing when using small *pop_size* values
- Using custom anakatabatic PSO without *akb_model* parameter now works without errors
- Fixed the error appearing when logging messages for some methods

***

## Indago 0.5.4

Released on 2025-11-16

### New features
- MSGS method has been completely rewritten, improved and renamed to Multi-Scale Gradient Descent (MSGD)
- **MSGS removed from Indago** <i class="fa-solid fa-triangle-exclamation"></i>
### Improvements
### Bug fixes
- Solved bug in Optimizer.plot_history (negative linthresh in symlog was triggered if c<=0 in all iterations)

***

## Indago 0.5.3

Released on 2024-12-16

### New features
- Adding ESC (Escape Algorithm) as unsupported optimizer (not documented nor tested)
### Improvements
- Improved explanation of parallel evaluation in documentation
### Bug fixes
- `Candidate.copy()` now copies `uniqe_str` attribute. There was a problem when accessing `uniqe_str` in assigned `Optimizer.post_iteration_processing` function.

***

## Indago 0.5.2

Released on 2024-11-02

### New features
- Indago methods (most of them anyway) can now solve unbounded (or mixed) problems. Bounds (`Optimizer.lb`, `Optimizer.ub`) can be skipped, or defined (partially or fully) by +/-np.inf or np.nan values. Indago will replace "open" bounds with +/-1e100. When solving an unbound problem `Optimizer.X0` must be provided
- Implementation of `Status` enum class for optimization status tracking (accessible via `Optimizer.status` attribute)
### Improvements
- **All methods are now significantly faster due to code optimizations in `Candidate` class**
- `Optimizer.optimize()` now returns a `Candidate` instance regardless of the method being used
- A multilevel candidate comparison operator used in Indago methods is now clearly explained in the documentation 
- Behavior of `indago.unconstrain()` is now better explained in the documentation
- Better formatting of elapsed time in dashboard monitoring output
### Bug fixes
- Fixed an issue with occurrences of NaN fitness values in MSGS
- Fixed an issue with MSGS exiting the search space bounds
- `Optimizer.plot_history()` is further improved to avoid division-by-zero warnings.
- Fixed MSGS status and log message for *xtol* stopping criterion

***

## Indago 0.5.1

Released on 2024-07-16

### New features
### Improvements
- Color tweaking and adding method name in history/convergence plot
- Optimization runs can now be resumed and restarting optimizations should not produce problems
- Some code refactoring
- Better literature referencing in documentation
- **New default population (*pop_init*) for DE: max(30, 5 * *dimensions*)**
- Reduced computational cost of BA
### Bug fixes
- Fixed bug in plot history
- Fixed X0-related bug in RS

***

## Indago 0.5.0
Released on 2024-01-12

### New features
- HSIW inertia technique for PSO
- FullyEmployed variant of ABC
- *scatter_method* optimizer parameter, with *'halton'* option for using Halton sequence for initializing design vectors
- **New method: Random Search (RS)**
- *Optimizer.plot_history()* is improved to produce better and fancier plots.
- Export of all evaluated candidate solutions (*Optimizer.evals_db* parameter)
- New utility functions: *unconstrain()* and *read_evals_db()*
- **New method: Grey Wolf Optimization (GWO)**
- Checking for unknown (illegal) optimizer attributes
- Some important optimizer parameters can be given as *evaluation_function* attributes (namely *dimensions*, *lb*, *ub*, *objectives*, *constraints*, *objective_weights*, *objective_labels*, *constraint_labels*)
- *Optimizer.status* providing textual information on the optimizer state
- *CandidateState.is_feasible()* for checking whether the solution is feasible (i.e. whether it satisfies all constraints)
- *Optimizer.copy()* produces a (deep) copy of an Optimizer object
### Improvements
- Cleaner log messages
- Better documentation
- **Completely rewritten ABC (significantly different behavior expected)**
- **New (shorter and more consistent) param names in almost all methods** <i class="fa-solid fa-triangle-exclamation"></i>
- NelderMead method now has an abbreviation (NM), just like all other methods
- CEC 2014 benchmark test is removed (we are working on a much more comprehensive benchmark test suite) <i class="fa-solid fa-triangle-exclamation"></i>
- New defaults in *minimize_exhaustive()*
- More consistent error reporting style
- **New (better) default param values for all methods**
- Better evaluation error handling
- ***number_of_processes* parameter is now called *processes*** <i class="fa-solid fa-triangle-exclamation"></i>
- Updated requirements (most importantly python>=3.9) <i class="fa-solid fa-triangle-exclamation"></i>
- *Optimizer.plot_history()* can skip subplots if needed. Useful for turning off dysfunctional subplots with too many plot lines.
### Bug fixes
- ***target_fitness* stopping criterion is not ignored any more**
- Fixed stopping criteria overshooting defined targets
- **Fixed memory leak due to circular referencing in Optimizer and CandidateState objects**
- Fixed multiple problems in MSGS
- Fixed some evaluation retrying problems
- Rich-based features adapted to a new version of rich

