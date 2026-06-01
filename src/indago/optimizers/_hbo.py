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

File content: Definition of Heap Based Optimizer (HBO) optimizer.
Usage: from indago import HBO

"""


import numpy as np
from indago import Optimizer, Candidate, Status
from indago import VariableType, VariableDictType, XFormat, X_Content_Type


class HBO(Optimizer):
    """Heap Based Optimizer method class.

    Reference: Q. Askari, M. Saeed, I. Younas, Heap-based optimizer inspired by corporate rank hierarchy
    for global optimization, Expert Systems with Applications, vol. 161, pp. 113702, 2020,
    https://doi.org/10.1016/j.eswa.2020.113702
    
    Attributes
    ----------
    variant : str
        Name of the HBO variant (``Vanilla`` or ``Dynamic``). Default: ``Vanilla``.
    params : dict
        A dictionary of HBO parameters.
    _pop : ndarray
        Solution candidates.
    _boss_of : list
        List of boss indices corresponding to worker indices.
    _colleagues_of : list
        List of colleague indices corresponding to worker indices.

    Returns
    -------
    optimizer : HBO
        HBO optimizer instance.
    """
    

    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some HBO-specific parameter checks
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

        if self.variant == 'Vanilla' or self.variant == 'Dynamic':
            mandatory_params = 'team_size levels'.split()

            if 'team_size' in self.params:
                self.params['team_size'] = int(self.params['team_size'])
                assert self.params['team_size'] > 0, \
                    "team_size parameter should be positive integer"
            else:
                self.params['team_size'] = int(max(self.dimensions**0.5, 3))
            defined_params += 'team_size'.split()

            if 'levels' in self.params:
                self.params['levels'] = int(self.params['levels'])
                assert self.params['levels'] > 0, \
                    "levels parameter should be positive integer"
            else:
                self.params['levels'] = 3
            defined_params += 'levels'.split()
    
        else:
            assert False, f'Unknown variant! {self.variant}'

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)


    def _init_method(self):
        """Private method for initializing the HBO optimizer instance.
        Initializes and evaluates the candidate population.

        Returns
        -------
        None
            Nothing
            
        """

        assert self.max_iterations or self.max_evaluations or self.max_elapsed_time, \
            'optimizer.max_iteration, optimizer.max_evaluations, or optimizer.max_elapsed_time should be provided for this method/variant'

        assert self.params['team_size'] >= 2, \
            'team size (team_size param) should be greater than or equal to 2'

        assert self.params['levels'] >= 2, \
            'hierarchy levels (levels param) should be greater than or equal to 2'

        # Generate population
        popsize = np.sum([self.params['team_size']**i for i in range(self.params['levels'])])
        self._pop: list[Candidate] = [Candidate(**self._candidate_init_info) for _ in range(popsize)]

        self._evaluate_initial_candidates()
        n0 = 0 if self._initial_candidates is None else self._initial_candidates.size
        # Using specified particles initial positions
        for p in range(len(self._pop)):
            if p < n0:
                self._pop[p] = self._initial_candidates[p].copy()

        self._initialize_X(self._pop[n0:])

        # Evaluate
        if n0 < popsize:
            self._collective_evaluation(self._pop[n0:])

        # if all candidates are NaNs
        if np.isnan([point.f for point in self._pop]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'

        # Sort and organize
        self._pop = sorted(self._pop)

        # helper lists
        hierarchy, level_of = [], []
        counter_from = 0
        for i in range(self.params['levels']):
            counter_to = counter_from + self.params['team_size'] ** i
            hierarchy.append(list(range(counter_from, counter_to)))
            level_of.append([i] * (counter_to - counter_from))
            counter_from = counter_to
        level_of = sum(level_of, [])  # flatten list

        self._boss_of = [[None]]
        for i in range(1, len(hierarchy)):
            for boss in hierarchy[i - 1]:
                self._boss_of.append([boss] * self.params['team_size'])
        self._boss_of = sum(self._boss_of, [])  # flatten list

        self._colleagues_of = [None]
        for i in range(1, len(self._pop)):
            colleagues_of_i = []
            for j in range(1, len(self._pop)):
                if level_of[i] == level_of[j] and i != j:
                    colleagues_of_i.append(j)
            self._colleagues_of.append(colleagues_of_i)

        self._finalize_iteration()


    def _run(self):
        """Run procedure for the HBO method.

        Returns
        -------
        optimum: Candidate
            Best solution found during the HBO optimization.
            
        """

        if self._inject:
            self._eeeo_inject(self._pop)

            # HBO specific
            self._pop = sorted(self._pop)

        self._check_params()

        self._resuming()

        while True:

            p1 = 1 - self._progress_factor()
            p2 = p1 + (1 - p1) / 2

            if self._progress_factor() != 0:
                T = self.it / self._progress_factor()
            else:
                T = 0
            C = int(T / 25)
            if C != 0:
                gamma = np.abs(2 - (T % int(T / C)) / (T / (4 * C)))
            else:
                gamma = 2

            test_i = []
            test_c = []

            for i, c in enumerate(self._pop):

                # top boss is not doing anything
                if i == 0:
                    continue

                lambd = 2 * np.random.uniform(size=self.dimensions) - 1
                p = np.random.uniform(size=self.dimensions)

                for k in range(self.dimensions):

                    if p1 < p[k] <= p2:
                        boss = self._pop[self._boss_of[i]]
                        c_copy = c.copy()
                        R = np.copy(c_copy._R)
                        R[k] = boss._R[k] + gamma * lambd[k] * np.abs(boss._R[k] - c._R[k])
                        c_copy._R = R
                        test_i.append(i)
                        test_c.append(c_copy)

                    elif p2 < p[k] <= 1:
                        colleague = self._pop[np.random.choice(self._colleagues_of[i])]
                        c_copy = c.copy()
                        R = np.copy(c_copy._R)
                        if colleague < c:
                            R[k] = colleague._R[k] + gamma * lambd[k] * np.abs(colleague._R[k] - c._R[k])
                        else:
                            R[k] = c._R[k] + gamma * lambd[k] * np.abs(colleague._R[k] - c._R[k])
                        c_copy._R = R
                        test_i.append(i)
                        test_c.append(c_copy)

            if test_i:
                self._randomize_categorical(test_c)
                self._collective_evaluation(test_c)

                for i, c in zip(test_i, test_c):
                    if c < self._pop[i]:
                        self._pop[i] = c

                if self.variant == 'Vanilla':
                    for i, c in enumerate(self._pop):

                        # top boss has no boss
                        if not self._boss_of[i]:
                            continue

                        boss = self._pop[self._boss_of[i]]
                        if c < boss:
                            self._pop[i], boss = c, self._pop[i]

                elif self.variant == 'Dynamic':
                    self._pop = sorted(self._pop)

            if self._finalize_iteration():
                break

        return self.best
