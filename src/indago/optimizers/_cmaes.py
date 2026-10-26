#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Indago
Python framework for numerical optimization
https://indago.readthedocs.io/
https://pypi.org/project/Indago/

Description: Indago contains several modern methods for real fitness function optimization over a real parameter domain
and supports multiple objectives and constraints. It was developed at the University of Rijeka, Faculty of Engineering.
Authors: Stefan Ivić, Siniša Družeta, Luka Grbčić
Contact: stefan.ivic@riteh.uniri.hr
License: MIT

File content: Definition of Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimizer.
Usage: from indago import CMAES

"""


import numpy as np
from indago import Optimizer, Candidate, OptimizerStatus
from indago import VariableType, VariableDictType, XFormat, X_Content_Type


class Offspring(Candidate):
    """CMA-ES Offspring class. CMA-ES Offspring is a candidate solution sampled from the CMA-ES search distribution.

    Attributes
    ----------
    Y : ndarray or None
        Mutation step of the offspring, i.e. a sample from the normal distribution N(0, C). The (unbounded) position
        of the offspring in the CMA-ES search space is mean + sigma * Y. ``None`` until the offspring is sampled.

    Returns
    -------
    offspring : Offspring
        Offspring instance.
    """

    def __init__(self, variables: VariableDictType, n_objectives: int = 1, n_constraints: int = 0,
                 x_format: XFormat = XFormat.TUPLE) -> None:
        """Offspring constructor.

        Parameters
        ----------
        variables : VariableDictType
            A dictionary containing definitions of all design variables.
        n_objectives : int
            Number of objectives.
        n_constraints : int
            Number of constraints.
        x_format : XFormat
            A format for attribute X's return value.

        Returns
        -------
        Offspring
            Offspring instance.
        """

        super().__init__(variables, n_objectives, n_constraints, x_format)

        self.Y = None

    def copy(self):
        """Method for creating a copy of the offspring.

        Returns
        -------
        Offspring
            Offspring instance

        """

        new = super().copy()
        new.Y = None if self.Y is None else np.copy(self.Y)
        return new


class CMAES(Optimizer):
    """Covariance Matrix Adaptation Evolution Strategy method class.

    Reference: N. Hansen, The CMA Evolution Strategy: A Tutorial, arXiv:1604.00772, 2016.
    IPOP restarts: A. Auger, N. Hansen, A Restart CMA Evolution Strategy With Increasing Population Size,
    Proceedings of the IEEE Congress on Evolutionary Computation (CEC 2005), pp. 1769-1776, 2005.
    BIPOP restarts: N. Hansen, Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed,
    Proceedings of the GECCO 2009 Workshop on Black-Box Optimization Benchmarking, pp. 2389-2396, 2009.

    CMA-ES samples candidate solutions (offspring) from the multivariate normal distribution N(mean, sigma^2 C) and
    adapts the mean, the step size sigma and the covariance matrix C according to the ranking of the offspring.
    The ranking is done by Indago candidate comparison, hence constraints and failed evaluations are supported.
    The search distribution handles all non-categorical variables in the relative space of the variables
    (the unit hypercube). Bounds of non-periodic variables are handled by a smooth periodic transformation, and
    the standard deviation of each variable is kept below one third of its domain (as in N. Hansen's cma package).
    Discrete variables are rounded to the nearest allowed value by Indago, while their standard deviation is kept
    above a lower bound and the best offspring are centered on their discrete values (LB+IC, as in N. Hansen's cma
    package). Categorical variables are randomized as in other Indago methods.
    Whenever the search converges or stagnates, CMA-ES is restarted from a new starting point, with the population
    size depending on the variant.

    Attributes
    ----------
    variant : str
        Name of the CMA-ES variant (``Vanilla``, ``IPOP`` or ``BIPOP``). In ``Vanilla``, all restarts use the initial
        population size, in ``IPOP`` the population size is doubled in each restart, while in ``BIPOP`` restarts
        alternate between a regime of increasing population size and a regime of small population size and small
        initial step size. Default: ``BIPOP``.
    params : dict
        A dictionary of CMA-ES parameters: ``pop_size`` is the initial population size (default: 4 + int(3 ln(n)),
        or 6 + int(3 (ln(n) + ln(n_discrete))) if discrete variables are present, with n being the number of
        non-categorical variables and n_discrete the number of discrete non-categorical variables), and ``init_step``
        is the initial step size relative to the domain width (default: 0.25).
    _pop : list
        Offspring population, i.e. the most recently sampled and evaluated offspring.
    _cma_indices : list
        Indices of variables handled by the CMA-ES search distribution (all non-categorical variables).
    _periodic : ndarray
        Boolean mask of periodic variables among the variables handled by CMA-ES.
    _discrete : ndarray
        Boolean mask of discrete variables among the variables handled by CMA-ES.
    _intervals : dict
        Intervals (in relative space) which are rounded to the same value of a discrete variable, given for each
        discrete variable (key is the position of the variable among the variables handled by CMA-ES) as a tuple of
        two arrays: boundaries of the intervals and widths of the intervals.
    _min_std_factor : float
        Lower bound of the standard deviation of a discrete variable, relative to the width of the interval which
        contains the mean of the search distribution.
    _min_std : ndarray
        Current lower bound of the standard deviation of the search distribution, for each variable handled by CMA-ES
        (zero for continuous variables).
    _pop_size : int
        Population size (lambda) of the current run.
    _mu : int
        Number of parents, i.e. the number of the best offspring used for computing the new mean.
    _weights : ndarray
        Recombination weights (positive for the mu best offspring, negative for the others).
    _mueff : float
        Variance effective selection mass of the positive weights.
    _cc : float
        Learning rate for the cumulation of the evolution path of the covariance matrix.
    _cs : float
        Learning rate for the cumulation of the evolution path of the step size.
    _c1 : float
        Learning rate for the rank-one update of the covariance matrix.
    _cmu : float
        Learning rate for the rank-mu update of the covariance matrix.
    _damps : float
        Damping parameter of the step size update.
    _chi_n : float
        Expected length of an n-dimensional standard normally distributed vector.
    _mean : ndarray
        Mean of the search distribution (in the unbounded CMA-ES search space).
    _sigma : float
        Step size of the search distribution.
    _sigma0 : float
        Initial step size of the current run.
    _C : ndarray
        Covariance matrix of the search distribution.
    _B : ndarray
        Eigenvectors (columns) of the covariance matrix.
    _D : ndarray
        Square roots of eigenvalues of the covariance matrix, i.e. standard deviations along principal axes.
    _invsqrtC : ndarray
        Inverse square root of the covariance matrix.
    _decomposition_ok : bool
        Flag indicating whether the latest eigendecomposition of the covariance matrix succeeded.
    _pc : ndarray
        Evolution path of the covariance matrix.
    _ps : ndarray
        (Conjugate) evolution path of the step size.
    _run_it : int
        Number of iterations (distribution updates) in the current run.
    _run_start_eval : int
        Number of evaluations conducted prior to the current run.
    _max_run_it : float
        Maximum number of iterations of the current run.
    _f_best_history : list
        Fitness of the best offspring in each iteration of the current run (``np.nan`` if not feasible).
    _f_median_history : list
        Median fitness of the offspring in each iteration of the current run (``np.nan`` if not all are feasible).
    _flat_it : int
        Number of consecutive iterations with flat fitness in the current run.
    _restarts : int
        Number of conducted restarts.
    _small_run : bool
        Flag indicating whether the current run is a small population run (BIPOP).
    _large_pop_size : int
        Population size of the latest large population run (BIPOP).
    _evals_small : int
        Number of evaluations used in small population runs (BIPOP).
    _evals_large : int
        Number of evaluations used in large population runs (BIPOP).

    Returns
    -------
    optimizer : CMAES
        CMA-ES optimizer instance.
    """


    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some CMA-ES-specific parameter checks
        and prepares the parameters to be validated by Optimizer._check_params.

        Returns
        -------
        None
            Nothing

        """

        if not self.variant:
            self.variant = 'BIPOP'

        defined_params = list(self.params.keys())
        mandatory_params, optional_params = [], []

        if self.variant in 'Vanilla IPOP BIPOP'.split():
            mandatory_params = 'pop_size init_step'.split()

            if 'pop_size' in self.params:
                self.params['pop_size'] = int(self.params['pop_size'])
                assert self.params['pop_size'] >= 2, \
                    "pop_size parameter should be integer greater than or equal to 2"
            else:
                # Default CMA-ES population size, based on the number of (non-categorical) variables handled by CMA-ES,
                # increased if discrete variables are present (as in Hansen's cma package)
                cma_var_types = [var_type for var_type, *_ in self.variables.values()
                                 if var_type is not VariableType.CATEGORICAL]
                n = len(cma_var_types)
                n_discrete = sum([var_type.is_discrete() for var_type in cma_var_types])
                if n_discrete == 0:
                    self.params['pop_size'] = 4 + int(3 * np.log(max(n, 1)))
                else:
                    self.params['pop_size'] = 6 + int(3 * (np.log(n) + np.log(n_discrete)))
            defined_params += 'pop_size'.split()

            if 'init_step' in self.params:
                self.params['init_step'] = float(self.params['init_step'])
                assert self.params['init_step'] > 0, \
                    "init_step parameter should be positive float"
            else:
                self.params['init_step'] = 0.25
            defined_params += 'init_step'.split()

        else:
            assert False, f'Unknown variant! {self.variant}'

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)


    def _init_method(self):
        """Private method for initializing the CMA-ES optimizer instance.
        Prepares the search distribution, then samples and evaluates the first population.

        Returns
        -------
        None
            Nothing

        """

        # Properties of the variables handled by CMA-ES
        self._periodic = np.array([var_definitions[i][0].is_periodic() for i in self._cma_indices])
        self._discrete = np.array([var_definitions[i][0].is_discrete() for i in self._cma_indices])

        # Intervals in relative space which are rounded (by Indago) to the same value of a discrete variable
        self._intervals = {}
        for i_cma, i_var in enumerate(self._cma_indices):
            var_type, *var_options = var_definitions[i_var]
            if var_type.is_discrete():
                # Allowed values of the discrete variable
                if var_type.is_integer():
                    values = np.arange(var_options[0], var_options[1] + 1)
                else:
                    values = np.array(var_options[0])
                # Range of values which Indago maps onto relative space [0, 1]; for non-periodic variables it is
                # extended by half of the spacing between values at both ends, so that each value gets a whole interval
                if var_type.is_periodic():
                    x_min, x_max = values[0], values[-1]
                else:
                    x_min = values[0] - (values[1] - values[0]) / 2
                    x_max = values[-1] + (values[-1] - values[-2]) / 2
                # Boundaries of the intervals lie at midpoints between consecutive values
                midpoints = (values[:-1] + values[1:]) / 2
                bounds = np.concatenate([[0], (midpoints - x_min) / (x_max - x_min), [1]])
                widths = np.diff(bounds)
                # In periodic variables, the intervals of the first and the last value are the two halves of
                # the interval around the boundary of the period, hence the width of the whole interval is used for both
                if var_type.is_periodic():
                    widths[0] = widths[-1] = widths[0] + widths[-1]
                self._intervals[i_cma] = bounds, widths

        # Restart bookkeeping (the first run is regarded as a small population run, as in Hansen's cma package)
        self._restarts = 0
        self._small_run = True
        self._large_pop_size = self.params['pop_size']
        self._evals_small = 0
        self._evals_large = 0

        # Evaluate initial candidates (if provided), the best of them is used as the starting point of the first run
        self._evaluate_initial_candidates()
        if self._initial_candidates is None:
            start = Candidate(**self._candidate_init_info)
            self._initialize_X([start])
        else:
            start = self._initial_candidates[0]

        # Prepare the search distribution of the first run
        self._start_run(start._R, self.params['pop_size'], self.params['init_step'])

        # Sample and evaluate the first population
        self._sample_population()
        self._evaluate_population()

        # if all candidates are NaNs
        if np.isnan([c.f for c in self._pop]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'

        self._finalize_iteration()


    def _start_run(self, R_start, pop_size, init_step):
        """Private method for (re)starting a CMA-ES run, i.e. for setting the strategy parameters
        and initializing the search distribution.

        Parameters
        ----------
        R_start : ndarray
            Relative position (of all variables) used as the mean of the initial search distribution.
        pop_size : int
            Population size (lambda) of the run.
        init_step : float
            Initial step size (sigma) of the run, relative to the domain of the variables.

        Returns
        -------
        None
            Nothing

        """

        n = len(self._cma_indices)

        # Population size
        self._pop_size = pop_size
        self._pop = [Offspring(**self._candidate_init_info) for _ in range(pop_size)]

        # Raw recombination weights, decreasing with rank (positive for better half, negative for worse half)
        weights_raw = np.log((pop_size + 1) / 2) - np.log(np.arange(1, pop_size + 1))
        self._mu = int(np.sum(weights_raw > 0))
        positive_raw, negative_raw = weights_raw[:self._mu], weights_raw[self._mu:]

        # Variance effective selection masses of positive and negative weights
        self._mueff = np.sum(positive_raw) ** 2 / np.sum(positive_raw ** 2)
        mueff_negative = np.sum(negative_raw) ** 2 / np.sum(negative_raw ** 2)

        # Learning rates for cumulation and step size control (Hansen, 2016)
        self._cc = (4 + self._mueff / n) / (n + 4 + 2 * self._mueff / n)
        self._cs = (self._mueff + 2) / (n + self._mueff + 5)
        self._damps = 1 + 2 * max(0, np.sqrt((self._mueff - 1) / (n + 1)) - 1) + self._cs

        # Learning rates for rank-one and rank-mu update of the covariance matrix (as in Hansen's cma package)
        self._c1 = min(1, pop_size / 6) * 2 / ((n + 1.3) ** 2 + self._mueff)
        self._cmu = min(1 - self._c1,
                        2 * (0.25 + self._mueff - 2 + 1 / self._mueff) / ((n + 2) ** 2 + self._mueff))

        # Positive weights sum to one, while negative weights sum to the largest value which keeps the covariance
        # matrix from decaying, preserves the selection mass, and keeps the covariance matrix positive-definite
        # (the latter limit is clipped at zero, since round-off can make it negative when cmu = 1 - c1)
        negative_sum = min(1 + self._c1 / self._cmu,
                           1 + 2 * mueff_negative / (self._mueff + 2),
                           max(0, (1 - self._c1 - self._cmu) / (n * self._cmu)))
        self._weights = np.concatenate([positive_raw / np.sum(positive_raw),
                                        negative_sum * negative_raw / np.sum(np.abs(negative_raw))])

        # Approximation of the expected length of an n-dimensional standard normally distributed vector
        self._chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Lower bound of the standard deviation of discrete variables, as a fraction of the width of the interval
        # which is rounded to the same value (as in Hansen's cma package, where all intervals are of unit width)
        self._min_std_factor = min(0.2, self._mueff / n)

        # Initial search distribution
        self._mean = self._box_transform_inverse(R_start[self._cma_indices], R_start[self._cma_indices])
        self._sigma0 = init_step
        self._sigma = init_step
        self._C = np.eye(n)
        self._pc = np.zeros(n)
        self._ps = np.zeros(n)
        self._limit_std()
        self._decomposition_ok = self._decompose_C()

        # Run bookkeeping
        self._run_it = 0
        self._run_start_eval = self.eval
        self._max_run_it = 100 + 150 * (n + 3) ** 2 / np.sqrt(pop_size)
        self._f_best_history = []
        self._f_median_history = []
        self._flat_it = 0


    def _sample_population(self):
        """Private method for sampling the offspring population from the current search distribution.

        Returns
        -------
        None
            Nothing

        """

        # Standard normally distributed samples z
        Z = np.random.standard_normal((self._pop_size, len(self._cma_indices)))

        # Mutation steps y = B D z, which are distributed as N(0, C)
        Y = Z @ (self._B * self._D).T

        # Relative positions within bounds, obtained by transforming the unbounded positions mean + sigma * y
        R_cma = self._box_transform(self._mean + self._sigma * Y)

        for offspring, y, r_cma in zip(self._pop, Y, R_cma):
            offspring.Y = y
            # Categorical variables are given NaN values, which Indago replaces with random values
            R = np.full(self.dimensions, np.nan)
            R[self._cma_indices] = r_cma
            offspring._R = R

        # Treat categorical variables (variables not handled by CMA-ES) as in other Indago methods
        if len(self._cma_indices) < self.dimensions:
            self._randomize_categorical(self._pop)


    def _evaluate_population(self):
        """Private method for evaluating the offspring population. Offspring which failed to evaluate and were
        moved towards the best solution by Indago's ``'retry'`` mechanism (see ``Optimizer.eval_fail_behavior``)
        get their mutation steps recomputed from their actual positions.

        Returns
        -------
        None
            Nothing

        """

        # Sampled design vectors are stored only if the 'retry' mechanism can move the offspring
        retry = self.eval_fail_behavior == 'retry'
        X_sampled = [offspring.X for offspring in self._pop] if retry else None
        fail_count = self.eval_fail_count

        self._collective_evaluation(self._pop)

        # Offspring moved by the 'retry' mechanism are treated as candidates not sampled by CMA-ES
        if retry and self.eval_fail_count > fail_count:
            for offspring, x_sampled in zip(self._pop, X_sampled):
                if not np.array_equal(offspring.X, x_sampled):
                    offspring.Y = self._injected_step(offspring._R)


    def _update_distribution(self):
        """Private method for updating the search distribution (mean, evolution paths, covariance matrix and
        step size) according to the ranking of the last evaluated population (Hansen, 2016).

        Returns
        -------
        None
            Nothing

        """

        n = len(self._cma_indices)

        # Rank the offspring by Indago candidate comparison (supports constraints and failed evaluations)
        ranked = sorted(self._pop)

        # Integer centering of discrete variables of the mu best offspring (parents)
        if np.any(self._discrete):
            self._center_discrete(ranked[:self._mu])

        Y = np.array([offspring.Y for offspring in ranked])

        # Weighted mean of the mu best mutation steps moves the mean
        y_w = self._weights[:self._mu] @ Y[:self._mu]
        self._mean = self._mean + self._sigma * y_w

        # Mean of periodic variables is wrapped to [0, 1), which does not change the search distribution
        self._mean[self._periodic] = np.mod(self._mean[self._periodic], 1)

        # Cumulation of the conjugate evolution path for step size control
        self._ps = (1 - self._cs) * self._ps \
            + np.sqrt(self._cs * (2 - self._cs) * self._mueff) * (self._invsqrtC @ y_w)
        self._run_it += 1

        # The rank-one update is stalled when the step size path is long (i.e. when sigma increases rapidly)
        ps_norm = np.linalg.norm(self._ps) / np.sqrt(1 - (1 - self._cs) ** (2 * self._run_it))
        h_sigma = 1 if ps_norm < (1.4 + 2 / (n + 1)) * self._chi_n else 0

        # Cumulation of the evolution path for the rank-one update of the covariance matrix
        self._pc = (1 - self._cc) * self._pc \
            + h_sigma * np.sqrt(self._cc * (2 - self._cc) * self._mueff) * y_w

        # Negative weights are rescaled by n / ||C^(-1/2) y||^2, which bounds the variance reduction they cause
        # (a negligible epsilon is added to the norm in order to avoid division by zero)
        weights_C = np.copy(self._weights)
        negative = weights_C < 0
        weights_C[negative] *= n / (np.linalg.norm(Y[negative] @ self._invsqrtC, axis=1) + 1e-9) ** 2

        # Covariance matrix update: decay, rank-one update and (active) rank-mu update
        variance_loss = (1 - h_sigma) * self._cc * (2 - self._cc)
        self._C = (1 + self._c1 * variance_loss - self._c1 - self._cmu * np.sum(self._weights)) * self._C \
            + self._c1 * np.outer(self._pc, self._pc) \
            + self._cmu * (Y.T * weights_C) @ Y

        # Step size update by cumulative step size adaptation (increase factor limited to e)
        self._sigma *= np.exp(min(1, self._cs / self._damps * (np.linalg.norm(self._ps) / self._chi_n - 1)))

        # Keep the standard deviations of variables within limits and update the eigendecomposition
        self._limit_std()
        self._decomposition_ok = self._decompose_C()

        # Fitness information used for restart criteria (fitness of infeasible offspring is not considered)
        F = np.array([offspring.f if offspring.is_feasible() else np.nan for offspring in ranked])
        self._f_best_history.append(F[0])
        self._f_median_history.append(np.median(F))
        # Only the history needed for stagnation detection is kept
        del self._f_best_history[:-20000]
        del self._f_median_history[:-20000]

        # Fitness is flat if the best offspring is as good as the one at 75% of the ranking
        if F[0] == F[int(0.75 * self._pop_size)]:
            self._flat_it += 1
        else:
            self._flat_it = 0


    def _center_discrete(self, parents):
        """Private method for integer centering of discrete variables of the parents (the best offspring), as in
        Hansen's cma package (T. Marty, N. Hansen, A. Auger, Y. Semet, S. Heron, LB+IC-CMA-ES: Two Simple
        Modifications of CMA-ES to Handle Mixed-Integer Problems, PPSN 2024). If the discrete value of a parent
        differs from the discrete value of the mean, the parent is moved to the relative position of its discrete
        value. The bias introduced by this is compensated by moving parents which have the same discrete value as
        the mean towards the relative position of that value. Mutation steps of the parents are changed accordingly.
        In periodic variables, the first and the last value are the same state and the moves are periodic (along
        the shorter way around the period).

        Parameters
        ----------
        parents : list of Offspring
            The best offspring (used for updating the mean).

        Returns
        -------
        None
            Nothing

        """

        discrete = self._discrete

        # Relative positions of the discrete values of the mean (Indago rounds the relative position of the mean)
        mean_candidate = Candidate(**self._candidate_init_info)
        R_mean = np.full(self.dimensions, np.nan)
        R_mean[self._cma_indices] = self._box_transform(self._mean)
        mean_candidate._R = R_mean
        R_mean = mean_candidate._R[self._cma_indices]

        # Unbounded positions of the parents and relative positions of their (rounded) values
        G = self._mean + self._sigma * np.array([parent.Y for parent in parents])
        R = np.array([parent._R[self._cma_indices] for parent in parents])

        # In periodic variables, the first and the last value (relative positions 0 and 1) are the same state,
        # hence relative positions are wrapped to [0, 1)
        R_mean = np.where(self._periodic, np.mod(R_mean, 1), R_mean)
        R = np.where(self._periodic, np.mod(R, 1), R)

        # Non-periodic discrete variables of the parents are repaired into the principal segment of the box
        # transformation, which does not change their values (as in Hansen's cma package, where they are repaired
        # into bounds)
        G = np.where(discrete & ~self._periodic, self._box_transform_inverse(self._box_transform(G), G), G)

        # Unbounded positions of the parents' values and of the mean's values (in the principal segment of
        # non-periodic variables, and as the periodic images closest to the parents in periodic variables)
        G_value = self._box_transform_inverse(R, G)
        G_mean_value = self._box_transform_inverse(R_mean, G)

        # Parents with values different from the values of the mean are centered, which introduces a bias
        centered = discrete & (R != R_mean)
        bias = np.sum(np.where(centered, G_value - G, 0), axis=0)

        # Moves of the other parents towards the position of the mean's value, which counteract the bias
        moves = np.where(discrete & ~centered, G_mean_value - G, 0)
        counteracting = moves * bias < 0
        counteracting_sum = np.sum(np.where(counteracting, moves, 0), axis=0)

        # Fraction of counteracting moves needed for compensating the bias (at most the full moves)
        fraction = np.minimum(1, np.divide(np.abs(bias), np.abs(counteracting_sum),
                                           out=np.zeros_like(bias), where=counteracting_sum != 0))

        # Centered and bias compensating positions of the parents give their new mutation steps
        G = np.where(centered, G_value, G + np.where(counteracting, fraction * moves, 0))
        for parent, g in zip(parents, G):
            parent.Y = (g - self._mean) / self._sigma


    def _limit_std(self):
        """Private method for keeping the coordinate-wise standard deviations of the search distribution within
        limits, as in Hansen's cma package. The standard deviation of a variable should not exceed one third of the
        domain width, while the standard deviation of a discrete variable should not fall below a fraction of the
        width of the interval which contains the mean and is rounded to the same value. Standard deviations are
        changed by scaling the corresponding rows and columns of the covariance matrix, along with the corresponding
        components of the evolution path of the covariance matrix.

        Returns
        -------
        None
            Nothing

        """

        # Upper limit of standard deviations (relative to the domain width)
        max_std = 1 / 3

        # Lower limit of standard deviations of discrete variables, computed from the width of the interval
        # which contains the mean (in relative space) and is rounded to the same value
        self._min_std = np.zeros(len(self._cma_indices))
        R_mean = self._box_transform(self._mean)
        for i_cma, (bounds, widths) in self._intervals.items():
            i_interval = np.clip(np.searchsorted(bounds, R_mean[i_cma], side='right') - 1, 0, len(widths) - 1)
            self._min_std[i_cma] = self._min_std_factor * widths[i_interval]

        # Standard deviations are clipped into limits by scaling rows and columns of C and components of pc
        std = self._sigma * np.sqrt(np.diag(self._C))
        scale = np.clip(std, self._min_std, max_std) / std
        self._C = self._C * np.outer(scale, scale)
        self._pc = self._pc * scale


    def _decompose_C(self):
        """Private method for the eigendecomposition of the covariance matrix.

        Returns
        -------
        success : bool
            ``True`` if the covariance matrix is finite and positive-definite, ``False`` otherwise.

        """

        # Enforce symmetry of the covariance matrix (removes round-off asymmetry)
        self._C = (self._C + self._C.T) / 2

        if not np.all(np.isfinite(self._C)):
            return False

        # C = B D^2 B^T, where B holds eigenvectors (in columns) and D^2 eigenvalues
        eigenvalues, B = np.linalg.eigh(self._C)
        if np.min(eigenvalues) <= 0:
            return False

        self._B = B
        self._D = np.sqrt(eigenvalues)
        # C^(-1/2) = B D^-1 B^T
        self._invsqrtC = (self._B / self._D) @ self._B.T
        return True


    def _box_transform(self, G):
        """Private method which transforms unbounded positions in the CMA-ES search space into relative positions
        of variables within bounds [0, 1]. A smooth (piecewise linear and quadratic) periodic transformation is used,
        as in Hansen's cma package (class BoxConstraintsLinQuadTransformation). Periodic variables are not
        transformed, since Indago maps their relative positions periodically.

        Parameters
        ----------
        G : ndarray
            Unbounded positions in the CMA-ES search space (vector or matrix with positions in rows).

        Returns
        -------
        R : ndarray
            Relative positions of variables handled by CMA-ES.

        """

        # Width of the margin with quadratic transformation, at both bounds (as in Hansen's cma package for
        # bounds which are symmetric around zero, where the margin is 1/40 of the domain width)
        a = 0.025

        # Reflect the positions periodically into [-a, 1 + a] (the period is 2 + 4a)
        period = 2 * (1 + 2 * a)
        R = np.mod(G + a, period)
        R = np.where(R > period / 2, period - R, R) - a

        # Quadratic transformation of the margins [-a, a] into [0, a] and [1 - a, 1 + a] into [1 - a, 1]
        R = np.where(R < a, (R + a) ** 2 / (4 * a), R)
        R = np.where(R > 1 - a, 1 - (R - 1 - a) ** 2 / (4 * a), R)

        return np.where(self._periodic, G, R)


    def _box_transform_inverse(self, R, G_near):
        """Private method which transforms relative positions of variables into unbounded positions in the CMA-ES
        search space (inverse of the ``_box_transform`` method). For non-periodic variables, the preimage within
        the principal segment [-a, 1 + a] (which is mapped onto [0, 1] monotonically) is taken, while for periodic
        variables the periodic image closest to the given reference positions is taken.

        Parameters
        ----------
        R : ndarray
            Relative positions of variables handled by CMA-ES.
        G_near : ndarray
            Reference unbounded positions, used for choosing the periodic image of periodic variables.

        Returns
        -------
        G : ndarray
            Unbounded positions in the CMA-ES search space.

        """

        # Width of the margin with quadratic transformation, at both bounds (as in Hansen's cma package for
        # bounds which are symmetric around zero, where the margin is 1/40 of the domain width)
        a = 0.025

        # Inverse of the quadratic transformation in the margins, identity elsewhere (gives a preimage in [-a, 1 + a])
        R_box = np.clip(R, 0, 1)
        G = np.where(R_box < a, np.sqrt(4 * a * R_box) - a, R_box)
        G = np.where(R_box > 1 - a, 1 + a - np.sqrt(4 * a * (1 - R_box)), G)

        # Periodic image closest to the reference position (shifted by an integer number of periods)
        G_periodic = R + np.round(G_near - R)

        return np.where(self._periodic, G_periodic, G)


    def _injected_step(self, R):
        """Private method for computing the mutation step leading from the mean of the search distribution to
        the given relative position, used for candidates which were not sampled by CMA-ES. The step length is
        limited in order to prevent disrupting the search distribution (N. Hansen, Injecting External Solutions
        Into CMA-ES, arXiv:1110.4181, 2011).

        Parameters
        ----------
        R : ndarray
            Relative position (of all variables).

        Returns
        -------
        y : ndarray
            Mutation step.

        """

        n = len(self._cma_indices)

        # Unbounded position corresponding to the given relative position (closest to the mean, if periodic)
        G = self._box_transform_inverse(R[self._cma_indices], self._mean)

        # Mutation step and its length measured with respect to the covariance matrix (Mahalanobis norm)
        y = (G - self._mean) / self._sigma
        length = np.linalg.norm(self._invsqrtC @ y)

        # Steps longer than typical are shortened
        max_length = np.sqrt(n) + 2 * n / (n + 2)
        if length > max_length:
            y = y * max_length / length
        return y


    def _restart_criteria(self):
        """Private method for checking the criteria for restarting CMA-ES, i.e. the termination criteria of
        a CMA-ES run (as in Hansen's cma package).

        Returns
        -------
        met_criteria : list of str
            Names of the met restart criteria (empty if no criterion is met).

        """

        n = len(self._cma_indices)

        # Current coordinate-wise standard deviations of the search distribution
        std = self._sigma * np.sqrt(np.diag(self._C))

        # Fitness of the current offspring (fitness of infeasible offspring is not considered)
        F = np.array([offspring.f if offspring.is_feasible() else np.nan for offspring in self._pop])

        # Recent history of the best fitness
        f_recent = np.array(self._f_best_history[-int(10 + 30 * n / self._pop_size):])

        # Number of iterations with flat fitness tolerated (more when discrete variables are present)
        tol_flat_it = 3 + 30 * np.mean(self._discrete) if np.any(self._discrete) else 1

        # Fitness range criteria are not used if all variables are discrete (as in Hansen's cma package)
        tol_fun = 0 if np.all(self._discrete) else 1e-11
        tol_hist_fun = 0 if np.all(self._discrete) else 1e-12

        # Stagnation: in the history window (latest 20% of iterations, but at least 120 + 30 n / lambda),
        # the median of the most recent 30% of best and median fitness values is not better than
        # the median of the oldest 30% of these values (Hansen, 2009)
        window = int(min(max(0.2 * self._run_it, 120 + 30 * n / self._pop_size), 20000))
        stagnation = False
        if len(self._f_best_history) >= window:
            k = int(0.3 * window)
            f_best = np.array(self._f_best_history[-window:])
            f_median = np.array(self._f_median_history[-window:])
            stagnation = np.median(f_best[-k:]) >= np.median(f_best[:k]) \
                and np.median(f_median[-k:]) >= np.median(f_median[:k])

        criteria = {
            # Maximal number of iterations of the run is reached
            'MaxIter': self._run_it >= self._max_run_it,
            # Range of the current fitness values and of the recent best fitness values is very small
            'TolFun': np.ptp(F) < tol_fun and np.ptp(f_recent) < tol_fun,
            # Range of the recent best fitness values (at least 10 iterations) is very small
            'TolHistFun': len(self._f_best_history) > 9 and np.ptp(f_recent) < tol_hist_fun,
            # Fitness was flat for too many iterations
            'FlatFitness': self._flat_it > tol_flat_it,
            # All standard deviations and evolution path components are very small (close to numerical resolution)
            'TolX': np.all(std < 5e-14) and np.all(self._sigma * np.abs(self._pc) < 5e-14),
            # Adding 0.1 standard deviation along any principal axis does not change the mean
            'NoEffectAxis': np.any(np.all(self._mean + 0.1 * self._sigma * (self._B * self._D).T == self._mean,
                                          axis=1)),
            # Adding 0.2 standard deviation in any coordinate does not change the mean
            'NoEffectCoord': np.any(self._mean + 0.2 * std == self._mean),
            # Condition number of the covariance matrix is excessive
            'ConditionCov': np.max(self._D) > 1e7 * np.min(self._D),
            # Step size increased excessively compared to the scale of the covariance matrix
            'TolUpSigma': self._sigma / np.max(self._D) > 1e20 * self._sigma0,
            # No progress in median and best fitness
            'Stagnation': stagnation,
            # Covariance matrix is no longer finite and positive-definite
            'NumericalError': not self._decomposition_ok,
        }

        return [name for name, met in criteria.items() if met]


    def _restart(self, met_criteria):
        """Private method for restarting CMA-ES from a new random starting point, with population size and initial
        step size set according to the variant: constant (``Vanilla``), increasing (``IPOP``) or alternating between
        large and small population regimes (``BIPOP``).

        Parameters
        ----------
        met_criteria : list of str
            Names of the met restart criteria.

        Returns
        -------
        None
            Nothing

        """

        n = len(self._cma_indices)

        # Account the evaluations of the finished run to its population regime (BIPOP)
        run_evals = self.eval - self._run_start_eval
        if self._small_run:
            self._evals_small += run_evals
        else:
            self._evals_large += run_evals

        # By default, a restart uses the initial population size and the initial step size
        pop_size = self.params['pop_size']
        init_step = self.params['init_step']
        max_run_it = None

        if self.variant == 'IPOP':
            # Population size is doubled in each restart
            pop_size = 2 * self._pop_size

        elif self.variant == 'BIPOP':
            if self._evals_small < max(1, self._evals_large):
                # Small population regime: population size is chosen randomly between the default one and
                # the one of the latest large run, initial step size is randomly reduced (down to 1/100),
                # and the run is limited to half of the evaluations used in large population runs
                self._small_run = True
                pop_size = int(self.params['pop_size']
                               * (self._large_pop_size / self.params['pop_size']) ** (np.random.uniform() ** 2))
                init_step = self.params['init_step'] * 0.01 ** np.random.uniform()
                max_run_it = min(100 + 150 * (n + 3) ** 2 / np.sqrt(self.params['pop_size']),
                                 0.5 * self._evals_large / pop_size)
            else:
                # Large population regime: population size is doubled with respect to the latest large run
                self._small_run = False
                self._large_pop_size = 2 * self._large_pop_size
                pop_size = self._large_pop_size

        self._restarts += 1
        self._log(f'CMA-ES restart #{self._restarts} after {self._run_it} iterations ({", ".join(met_criteria)}), '
                  f'pop_size={pop_size}, init_step={init_step:.3e}')

        # New run starts from a random point
        start = Candidate(**self._candidate_init_info)
        self._initialize_X([start])
        self._start_run(start._R, pop_size, init_step)
        if max_run_it is not None:
            self._max_run_it = max_run_it


    def _run(self):
        """Run procedure for the CMA-ES method.

        Returns
        -------
        optimum: Offspring
            Best solution found during the CMA-ES optimization.

        """

        # SynCO-hybridization: injected candidates replace the worst offspring of the last evaluated population
        if self._inject:
            injected = self._synco_inject(self._pop)
            # Injected candidates were not sampled by CMA-ES, hence their mutation steps are computed
            for offspring in injected:
                offspring.Y = self._injected_step(offspring._R)

        # Checking user/default-defined parameters
        self._check_params()

        # If optimization is resumed
        self._resuming()

        while True:

            # Adapt the search distribution according to the last evaluated population
            self._update_distribution()

            # Restart CMA-ES if the search has converged or stagnated
            met_criteria = self._restart_criteria()
            if met_criteria:
                self._restart(met_criteria)

            # Sample and evaluate a new population
            self._sample_population()
            self._evaluate_population()

            # Iteration done - all checks and administration are performed now
            if self._finalize_iteration():
                break

        # Optimization finished
        return self.best
