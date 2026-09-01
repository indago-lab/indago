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

File content: Definition of Ant Colony Optimization (ACO) optimizer.
Usage: from indago import ACO

"""


import numpy as np
from indago import Optimizer, Candidate, OptimizerStatus
from indago import VariableType, VariableDictType, XFormat, X_Content_Type
import random


class ACO(Optimizer):
    """Ant Colony Optimization method class.

    Reference: (citation of paper on which the implementation is based)
    
    Attributes
    ----------
    variant : str
        Name of the ACO variant (``Vanilla`` or ``Spillover``). Default: ``Vanilla``.
    params : dict
        A dictionary of ACO parameters.
    _pop : list
        Solution candidates.
    _net : list of list
        Network topology with variable values associated with each link.
    _ph : list of ndarray
        Pheromone values for each network link, i.e. variable value.

    Returns
    -------
    optimizer : ACO
        ACO optimizer instance.
    """
    

    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some ACO-specific parameter checks
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

        mandatory_params = 'pop_size evap_rate'.split()

        if 'pop_size' in self.params:
            self.params['pop_size'] = int(self.params['pop_size'])
            assert self.params['pop_size'] > 0, \
                "pop_size parameter should be positive integer"
        else:
            self.params['pop_size'] = int(max(self.dimensions, 10))
        defined_params += 'pop_size'.split()

        if 'evap_rate' in self.params:
            self.params['evap_rate'] = float(self.params['evap_rate'])
            assert 0 <= self.params['evap_rate'] <= 1, \
                "evap_rate parameter should be positive float in [0, 1]"
        else:
            self.params['evap_rate'] = 0.1
        defined_params += 'evap_rate'.split()

        if self.variant == 'Vanilla':
            pass

        elif self.variant == 'Spillover':
            mandatory_params += 'so_rate'.split()

            if 'so_rate' in self.params:
                self.params['so_rate'] = float(self.params['so_rate'])
                assert self.params['so_rate'] > 0, \
                    "so_rate parameter should be positive float"
            else:
                self.params['so_rate'] = 0.3
            defined_params += 'so_rate'.split()

        else:
            assert False, f'Unknown variant! {self.variant}'

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)


    def _init_method(self):
        """Private method for initializing the ACO optimizer instance.
        Initializes and evaluates the candidate population.

        Returns
        -------
        None
            Nothing
            
        """

        self._net = []
        self._ph = []

        for _, (var_name, (var_type, *var_options)) in enumerate(self.variables.items()):
            assert var_type.is_discrete(), \
                f'Variable {var_name} is not of discrete/integer/categorical type and as such cannot be used with ACO'

            if var_type.is_integer():
                self._net.append([n for n in range(var_options[0], var_options[1] + 1)])
            else:
                self._net.append(var_options[0])

            self._ph.append(np.full(len(self._net[-1]), 1.0))

        # print(f'Network topology: {self._net}')

        # Generate population
        self._pop = [Candidate(**self._candidate_init_info) for _ in range(self.params['pop_size'])]

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
        if n0 < self.params['pop_size']:
            self._collective_evaluation(self._pop[n0:])

        # if all candidates are NaNs
        if np.isnan([point.f for point in self._pop]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'

        # Done
        self._finalize_iteration()


    def _run(self):
        """Run procedure for the ACO method.

        Returns
        -------
        optimum: Candidate
            Best solution found during the ACO optimization.
            
        """

        if self._inject:
            new = self._eeeo_inject(self._pop)
            if new < self.best:
                self.best = new

        # Checking user/default-defined parameters
        self._check_params()

        # If optimization is resumed
        self._resuming()

        while True:

            # apply pheromone based on the best candidate
            for i, (var_list, (_, (var_type, *__))) in enumerate(zip(self._net, self.variables.items())):
                for j, val in enumerate(var_list):
                    if val == self.best.X[i]:  # alternatively: if val == self._pop[0].X[i] (with sorted _pop)
                        self._ph[i][j] += 1

                        if self.variant == 'Spillover':
                            if var_type is not VariableType.CATEGORICAL:
                                if j == 0:
                                    if var_type.is_periodic():
                                        self._ph[i][-2] += self.params['so_rate']
                                    self._ph[i][j+1] += self.params['so_rate']
                                elif j == len(var_list) - 1:
                                    self._ph[i][j-1] += self.params['so_rate']
                                    if var_type.is_periodic():
                                        self._ph[i][1] += self.params['so_rate']
                                else:
                                    if j - 1 >= 0:
                                        self._ph[i][j-1] += self.params['so_rate']
                                    if j + 1 <= len(var_list) - 1:
                                        self._ph[i][j+1] += self.params['so_rate']

                # equalize boundary values for periodic variables
                if var_type.is_periodic():
                    self._ph[i][0] = max(self._ph[i][0], self._ph[i][-1])
                    self._ph[i][-1] = self._ph[i][0]

            # evaporation
            for i, var_list in enumerate(self._net):
                for j, val in enumerate(var_list):
                    self._ph[i][j] *= (1 - self.params['evap_rate'])

            # prepare new generation
            for c in self._pop:
                newX = []
                for i, var_list in enumerate(self._net):
                    newX.append(random.choices(var_list, weights=self._ph[i])[0])
                c.X = tuple(newX)

            # When you prepare the list of candidates which must be evaluated
            self._collective_evaluation(self._pop)

            # Iteration done - all checks and administration are performed now
            if self._finalize_iteration():
                break

        # Optimization finished
        return self.best