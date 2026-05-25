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

File content: Definition of Controlled Random Search (CRS) optimizer.
Usage: from indago import CRS

"""


import numpy as np
from indago.core._optimizer import Optimizer, Status
from indago.core._candidate import X_Content_Type
from indago import Candidate, VariableType, VariableDictType, XFormat


class CRS(Optimizer):
    """Controlled Random Search class.
    This is a CRS2-type method variant with local mutation.

    Reference: P. Kaelo and M. M. Ali. Some variants of the controlled random search algorithm
    for global optimization, J. Optim. Theory Appl. 130 (2), 253-264 (2006).

    Due to evaluating only one or two candidates per iteration, this method:
    - needs many iterations to be efficient (much more than other methods)
    - is effectively not parallelized in **_collective_evaluation** (hence parallel evaluation is not allowed, to avoid user confusion)

    Attributes
    ----------
    variant : str
        Name of the CRS variant (only ``Vanilla`` available). Default: ``Vanilla``.
    params : dict
        A dictionary of CRS parameters.
    _pop : ndarray
        Population of solution candidates.
    _simp : ndarray
        Simplex for direct search.
    _trial: Candidate
        Trial candidate of the simplex.

    Returns
    -------
    optimizer : CRS
        CRS optimizer instance.
    """

    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some CRS-specific parameter checks
        and prepares the parameters to be validated by Optimizer._check_params.

        Returns
        -------
        None
            Nothing
            
        """

        if not self.variant:
            self.variant = 'Vanilla'

        defined_params = list(self.params.keys())
        mandatory_params, optional_params = [], []

        if self.variant == 'Vanilla':
            mandatory_params = 'pop_scale'.split()

            if 'pop_scale' in self.params:
                self.params['pop_scale'] = float(self.params['pop_scale'])
                assert self.params['pop_scale'] >= 1, \
                    "pop_scale parameter should be greater or equal to 1"
            else:
                self.params['pop_scale'] = 10
            defined_params += 'pop_scale'.split()

        else:
            assert False, f'Unknown variant! {self.variant}'

        if self.processes > 1:
            self._log('Warning: CRS does not support parallelization. Defaulting to processes=1.')
            self.processes = 1

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)


    def _init_method(self):
        """Private method for initializing the CRS optimizer instance.
        Initializes and evaluates the candidate population.

        Returns
        -------
        None
            Nothing
            
        """

        if self._all_real:
            self._x_format = XFormat.Ndarray

        # Generate population
        pop_size = int(self.params['pop_scale'] * (self.dimensions + 1))
        self._pop: list[Candidate] = [Candidate(**self._problem_info) for _ in range(pop_size)]

        # Initialize simplex
        self._simp = [None for _ in range(self.dimensions + 1)]

        # Embedding initial candidates in the population
        self._evaluate_initial_candidates()
        n0 = 0 if self._initial_candidates is None else self._initial_candidates.size
        # Using specified particles initial positions
        for p in range(len(self._pop)):
            if p < n0:
                self._pop[p] = self._initial_candidates[p].copy()

        # Generate X for the rest
        self._initialize_X(self._pop[n0:])

        # Evaluate
        if n0 < pop_size:
            self._collective_evaluation(self._pop[n0:])

        # If all candidates are NaNs
        if np.isnan([point.f for point in self._pop]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'

        # Done
        self._finalize_iteration()


    def _run(self):
        """Run procedure for the CRS method.

        Returns
        -------
        optimum: Candidate
            Best solution found during the CRS optimization.
            
        """

        # EEEO
        if self._inject:
            worst = np.max(self._pop)
            worst.X = np.copy(self._inject.X)
            worst.O = np.copy(self._inject.O)
            worst.C = np.copy(self._inject.C)
            worst.f = self._inject.f

        # Checking user/default-defined parameters
        self._check_params()

        # If optimization is resumed
        if self.status == Status.RESUMED:
            if self._stopping_criteria():
                return self.best
            # TODO inspect why this is necessary for resume to work:
            self.it += 1
        else:
            self._init_method()

        while True:

            # Update simplex
            self._pop = sorted(self._pop)
            self._simp[0] = self._pop[0]
            self._simp[1:] = np.random.choice(self._pop[1:], self.dimensions, replace=False)
            self._simp = sorted(self._simp)

            # First trial candidate
            self._trial = self._simp[-1].copy()
            G = np.mean([c._R for c in self._simp[:-1]], axis=0)  # centroid
            self._trial._R = 2 * G - self._simp[-1]._R
            self._randomize_categorical([self._trial])
            self._collective_evaluation([self._trial])

            # Insert or mutate
            if self._trial < self._pop[-1]:
                self._pop[-1] = self._trial
            else:
                # Second trial candidate (local mutation)
                omega = np.random.uniform(0, 1, size=self.dimensions)
                self._trial._R = (1 + omega) * self._simp[0]._R - omega * self._trial._R
                self._randomize_categorical([self._trial])
                self._collective_evaluation([self._trial])
                if self._trial < self._pop[-1]:
                    self._pop[-1] = self._trial

            # Iteration done
            if self._finalize_iteration():
                break

        # Optimization finished
        return self.best