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

File content: Definition of Artificial Bee Colony (ABC) optimizer.
Usage: from indago import ABC

"""


import numpy as np
from indago import Optimizer, Candidate, Status
from indago import VariableType, VariableDictType, XFormat, X_Content_Type


class ABC(Optimizer):
    """Artificial Bee Colony method class.
    
    Attributes
    ----------
    variant : str
        Name of the ABC variant (``Vanilla`` or ``FullyEmployed``). Default: ``Vanilla``.
    params : dict
        A dictionary of ABC parameters.
    _hive_em : list
        Employed part of the bee population.
    _hive_em_v : list
        Mutated employed bees.
    _trials_em : ndarray
        Trial counters for the employed bees.
    _hive_on : list
        Onlooker part of the bee population.
    _hive_on_v : list
        Mutated onlooker bees.
    _trials_on : ndarray
        Trial counters for the onlooker bees.
    _probability : ndarray
        Probability for selecting informer bees.
        
    Returns
    -------
    optimizer : ABC
        ABC optimizer instance.
    """

    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some ABC-specific parameter checks
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
            mandatory_params += 'pop_size trial_limit'.split()
            
            if 'pop_size' in self.params:
                self.params['pop_size'] = int(self.params['pop_size'])
            else:
                self.params['pop_size'] = max(10, self.dimensions * 2)
            defined_params += 'pop_size'.split()
            
            if 'trial_limit' in self.params:
                self.params['trial_limit'] = int(self.params['trial_limit'])            
            else:
                self.params['trial_limit'] = int(self.params['pop_size'] * self.dimensions / 2) # Karaboga and Gorkemli 2014 - "A quick artificial bee colony (qabc) algorithm and its performance on optimization problems"
                defined_params += 'trial_limit'.split()
        
        elif self.variant == 'FullyEmployed':
            mandatory_params += 'pop_size trial_limit'.split()
            
            if 'pop_size' in self.params:
                self.params['pop_size'] = int(self.params['pop_size'])
            else:
                self.params['pop_size'] = max(10, self.dimensions * 2)
            defined_params += 'pop_size'.split()
            
            if 'trial_limit' in self.params:
                self.params['trial_limit'] = int(self.params['trial_limit'])            
            else:
                self.params['trial_limit'] = int(self.params['pop_size'] * self.dimensions / 2) # Karaboga and Gorkemli 2014 - "A quick artificial bee colony (qabc) algorithm and its performance on optimization problems"
                defined_params += 'trial_limit'.split()
    
        else:
            assert False, f'Unknown variant! {self.variant}'

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)


    def _init_method(self):
        """Private method for initializing the ABC optimizer instance.
        Initializes and evaluates the population.

        Returns
        -------
        None
            Nothing
            
        """
        # self._pop: list[Candidate] = [Candidate(**self._candidate_init_info) for _ in range(popsize)]
        # Generate employed bees
        if self.variant == 'FullyEmployed':
            self._hive_em: list[Candidate] = [Candidate(**self._candidate_init_info) \
                                              for _ in range(self.params['pop_size'])]
        else:
            self._hive_em: list[Candidate] = [Candidate(**self._candidate_init_info) \
                                              for _ in range(self.params['pop_size']//2)]
        self._hive_em_v = np.copy(self._hive_em).tolist()
        self._trials_em = np.zeros(len(self._hive_em), dtype=np.int32)
        self._probability = np.zeros(len(self._hive_em))
        
        # Generate onlooker bees
        if not self.variant == 'FullyEmployed':
            self._hive_on: list[Candidate] = [Candidate(**self._candidate_init_info) \
                                              for _ in range(self.params['pop_size']//2)]
            self._hive_on_v = np.copy(self._hive_on).tolist()
            self._trials_on = np.zeros(len(self._hive_on), dtype=np.int32)
        
        self._evaluate_initial_candidates()
        n0 = 0 if self._initial_candidates is None else self._initial_candidates.size
        
        self._initialize_X(self._hive_em)
        if not self.variant == 'FullyEmployed':
            self._initialize_X(self._hive_on)
        
        # Using specified particles initial positions
        for p in range(len(self._hive_em)):
            if p < n0:
                self._hive_em[p] = self._initial_candidates[p].copy()
            
        # Evaluate
        if n0 < len(self._hive_em):
            self._collective_evaluation(self._hive_em[n0:])
        if not self.variant == 'FullyEmployed':
            self._collective_evaluation(self._hive_on)

        # if all candidates are NaNs       
        if np.isnan([bee.f for bee in self._hive_em]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'

        self._finalize_iteration()
    
        
    def _run(self):
        """Main loop of ABC method.

        Returns
        -------
        optimum: Candidate
            Best solution found during the ABC optimization.
        """

        if self._inject:
            self._eeeo_inject(self._hive_on)

        self._check_params()

        self._resuming()

        while True:
            
            """employed bees phase"""                 
            for p, bee in enumerate(self._hive_em):
                
                self._hive_em_v[p] = bee.copy()

                informer = np.random.choice(np.delete(self._hive_em, p))
                d = np.random.randint(0, self.dimensions)
                phi = np.random.uniform(-1, 1)

                R = np.copy(self._hive_em_v[p]._R)
                R[d] = bee._R[d] + phi * (bee._R[d] - informer._R[d])
                self._hive_em_v[p]._R = R

            self._randomize_categorical(self._hive_em_v)

            self._collective_evaluation(self._hive_em_v)
            
            for p, bee in enumerate(self._hive_em_v):
                
                if bee < self._hive_em[p]:
                    self._hive_em[p] = bee.copy()
                    self._trials_em[p] = 0
                else:
                    self._trials_em[p] += 1
            
            if not self.variant == 'FullyEmployed':
            
                """probability update"""
                ranks = np.argsort(np.argsort(self._hive_em))
                self._probability = (np.max(ranks) - ranks) / np.sum(np.max(ranks) - ranks)
                
                # # original probability (fitness based)
                # fits = np.array([c.f for c in self._hive_em])
                # self._probability = (np.max(fits) - fits) / np.sum(np.max(fits) - fits)

                """onlooker bee phase"""
                for p, bee in enumerate(self._hive_on):
                    
                    self._hive_on_v[p] = bee.copy()
                    
                    informer = np.random.choice(self._hive_em, p=self._probability) 
                    d = np.random.randint(0, self.dimensions)
                    phi = np.random.uniform(-1, 1)

                    R = np.copy(self._hive_on_v[p]._R)
                    R[d] = bee._R[d] + phi * (bee._R[d] - informer._R[d])
                    self._hive_on_v[p]._R = R

                self._randomize_categorical(self._hive_on_v)
    
                self._collective_evaluation(self._hive_on_v)
                
                for p, bee in enumerate(self._hive_on_v):
                    
                    if bee < self._hive_on[p]:
                        self._hive_on[p] = bee.copy()
                        self._trials_on[p] = 0
                    else:
                        self._trials_on[p] += 1

            """scout bee phase"""
            for p, bee in enumerate(self._hive_em):
                if self._trials_em[p] > self.params['trial_limit']:
                    bee._R = np.random.uniform(0, 1)
                    self._trials_em[p] = 0
            
            if not self.variant == 'FullyEmployed':
                for p, bee in enumerate(self._hive_on):
                    if self._trials_on[p] > self.params['trial_limit']:
                        bee._R = np.random.uniform(0, 1)
                        self._trials_on[p] = 0
            
            if self._finalize_iteration():
                break
        
        return self.best
