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

File content: Definition of Grey Wolf Optimizer (GWO) optimizer.
Usage: from indago import GWO

"""


import numpy as np
from indago.core._optimizer import Optimizer, Status
from indago.core._candidate import X_Content_Type
from indago import Candidate, VariableType, VariableDictType, XFormat


class GWO(Optimizer):
    """Grey Wolf Optimizer method class.
    
    Reference: S. Mirjalili, S. M. Mirjalili, A. Lewis, Grey Wolf Optimizer, 
    Advances in Engineering Software, vol. 69, pp. 46-61, 2014, 
    DOI: http://dx.doi.org/10.1016/j.advengsoft.2013.12.007
    
    Attributes
    ----------
    variant : str
        Name of the GWO variant (``Vanilla`` or ``HSA``). Default: ``Vanilla``.
    params : dict
        A dictionary of GWO parameters.
    _wolves : ndarray
        Array of Candidate instances.
        
    Returns
    -------
    optimizer : GWO
        GWO optimizer instance.
    """

    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some GWO-specific parameter checks
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

        if self.variant == 'Vanilla' or self.variant == 'HSA':
            mandatory_params += 'pop_size'.split()
            
            if 'pop_size' in self.params:
                self.params['pop_size'] = int(self.params['pop_size'])
            else:
                self.params['pop_size'] = max(10, self.dimensions)
            defined_params += 'pop_size'.split()
          
        else:
            assert False, f'Unknown variant! {self.variant}'

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)


    def _init_method(self):
        """Private method for initializing the GWO optimizer instance.
        Initializes and evaluates the population.
        
        Returns
        -------
        None
            Nothing
        """

        assert self.max_iterations or self.max_evaluations or self.max_elapsed_time, \
            'optimizer.max_iteration, optimizer.max_evaluations, or optimizer.max_elapsed_time should be provided for this method/variant'

        assert self.params['pop_size'] >= 5, \
            'population size (pop_size param) should be greater than or equal to 5'

        # Generate population
        self._wolves = [Candidate(**self._candidate_init_info) for _ in range(self.params['pop_size'])]
        
        self._evaluate_initial_candidates()
        n0 = 0 if self._initial_candidates is None else self._initial_candidates.size
        # Using specified particles initial positions
        for p in range(len(self._wolves)):
            if p < n0:
                self._wolves[p] = self._initial_candidates[p].copy()

        self._initialize_X(self._wolves[n0:])

        # Evaluate
        if n0 < len(self._wolves):
            self._collective_evaluation(self._wolves[n0:])

        # If all candidates are NaNs       
        if np.isnan([c.f for c in self._wolves]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'

        self._finalize_iteration()
    
        
    def _run(self):
        """Main loop of GWO method.

        Returns
        -------
        optimum: Candidate
            Best solution found during the GWO optimization.
            
        """

        # for EEEO
        if self._inject:
            worst = np.max(self._wolves)
            worst.X = self._inject.X
            worst.O = np.copy(self._inject.O)
            worst.C = np.copy(self._inject.C)
            worst.f = self._inject.f

        self._check_params()

        if self.status == Status.RESUMED:
            if self._stopping_criteria():
                return self.best
            # TODO inspect why this is necessary for resume to work:
            self.it += 1
        else:
            self._init_method()

        while True:
            
            # find alpha, beta, delta
            alpha, beta, delta = sorted(self._wolves)[:3]
            
            # calculate a
            if self.variant == 'Vanilla':
                # linearly decreasing (from 2 to 0)
                a = 2 * (1 - self._progress_factor()) 
            elif self.variant == 'HSA':
                # changing over (3/4 of) half-sinusoid (from 2*sin(pi/4) to 0)
                a = 2 * np.sin(np.pi/4 + self._progress_factor() * (np.pi - np.pi/4))
            
            # move wolves
            for c in self._wolves:
            
                r1 = np.random.uniform(-1, 1, self.dimensions)
                r2 = np.random.uniform(0, 2, self.dimensions)
                pos1 = alpha._R - a * r1 * np.abs(r2 * alpha._R - c._R)

                r1 = np.random.uniform(-1, 1, self.dimensions)
                r2 = np.random.uniform(0, 2, self.dimensions)            
                pos2 = beta._R - a * r1 * np.abs(r2 * beta._R - c._R)
                
                r1 = np.random.uniform(-1, 1, self.dimensions)
                r2 = np.random.uniform(0, 2, self.dimensions)               
                pos3 = delta._R - a * r1 * np.abs(r2 * delta._R - c._R)

                c._R = (pos1 + pos2 + pos3) / 3

            self._randomize_categorical(self._wolves)

            # evaluate
            self._collective_evaluation(self._wolves)

            if self._finalize_iteration():
                break
        
        return self.best
