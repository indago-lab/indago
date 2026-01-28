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

File content: Definition of Random Search (RS) optimizer.
Usage: from indago import RS

"""


import numpy as np
from indago.core._optimizer import Optimizer, Candidate, Status


class RS(Optimizer):
    """Random Search method class.
    
    Attributes
    ----------
    variant : str
        Name of the RS variant. Default (and the only available option): ``Vanilla``.
    params : dict
        A dictionary of RS parameters.
    _points : ndarray
        Solution candidates.
        
    Returns
    -------
    optimizer : RS
        RS optimizer instance.
    """
    

    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some RS-specific parameter checks
        and prepares the parameters to be validated by Optimizer._check_params.

        Returns
        -------
        None
            Nothing
            
        """

        if not self.variant:
            self.variant = 'Vanilla'

        defined_params = list(self.params.keys())
        mandatory_params, optional_params = ['batch_size'], []
        
        if 'batch_size' in self.params:
            self.params['batch_size'] = int(self.params['batch_size'])
            assert self.params['batch_size'] > 0, \
                "batch_size parameter should be positive integer"
        else:
            self.params['batch_size'] = self.dimensions
        defined_params += 'batch_size'.split()

        if self.variant == 'Vanilla':
            pass
    
        else:
            assert False, f'Unknown variant! {self.variant}'

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)


    def _init_method(self):
        """Private method for initializing the RS optimizer instance.
        Initializes and evaluates optimizer.max_evaluations number of candidates.

        Returns
        -------
        None
            Nothing
            
        """
        
        self._evaluate_initial_candidates()

        # Generate all candidates
        n0 = 0 if self._initial_candidates is None else self._initial_candidates.size
        self._points = np.array([Candidate(self.variables) for _ in range(self.params['batch_size'])], dtype=Candidate)
        self._initialize_X(self._points[n0:])
        
        # Using specified particles initial positions
        for p in range(np.size(self._points)):
            if p < n0:
                self._points[p] = self._initial_candidates[p].copy()
            
        # Evaluate
        if n0 < self.params['batch_size']:
            self._collective_evaluation(self._points[n0:])

        # if all candidates are NaNs
        if np.isnan([point.f for point in self._points]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'

        self._finalize_iteration()
    
        
    def _run(self):
        """Run procedure for the RS method. 

        Returns
        -------
        optimum: Candidate
            Best solution found during the RS optimization.
            
        """

        self._check_params()

        if self.status == Status.RESUMED:
            if self._stopping_criteria():
                return self.best
            # TODO inspect why this is necessary for resume to work:
            self.it += 1
        else:
            self._init_method()

        while True:

            self._initialize_X(self._points)
            self._collective_evaluation(self._points)

            if self._finalize_iteration():
                break

        return self.best