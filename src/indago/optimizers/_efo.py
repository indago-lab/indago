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

File content: Definition of Electromagnetic Field Optimization (EFO) optimizer.
Usage: from indago import EFO

"""


import numpy as np
from indago import Optimizer, Candidate, Status
from indago import VariableType, VariableDictType, XFormat, X_Content_Type
from scipy.constants import golden_ratio as phi


Candidate = Candidate

class EFO(Optimizer):
    """Electromagnetic Field Optimization method class.
    
    Reference: Abedinpourshotorban, H., Shamsuddin, S.M., Beheshti, Z., & Jawawi, 
    D.N. (2016). Electromagnetic field optimization: a physics-inspired metaheuristic 
    optimization algorithm. Swarm and Evolutionary Computation, 26, 8-22.
    
    Due to evaluating only one candidate per iteration, this method:
    - needs many iterations to be efficient (much more than other methods)
    - is effectively not parallelized in **_collective_evaluation** (hence parallel evaluation is not allowed, to avoid user confusion)
    
    Attributes
    ----------
    variant : str
        Name of the EFO variant. Default: ``Vanilla``.
    params : dict
        A dictionary of EFO parameters.
    _EM : list
        Solution candidates (emParticles).
    _emNew : Candidate
        Test solution.
        
    Returns
    -------
    optimizer : EFO
        EFO optimizer instance.
        
    """

    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some EFO-specific parameter checks
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
        
        if 'pop_size' in self.params:
            self.params['pop_size'] = int(self.params['pop_size'])

        if self.variant == 'Vanilla':
            mandatory_params = 'pop_size R_rate Ps_rate P_field N_field'.split()
            if 'pop_size' not in self.params:
                self.params['pop_size'] = max(50, self.dimensions)
                defined_params += 'pop_size'.split()
            if 'R_rate' not in self.params:
                self.params['R_rate'] = 0.25 # recommended [0.1, 0.4]
                defined_params += 'R_rate'.split()
            if 'Ps_rate' not in self.params:
                self.params['Ps_rate'] = 0.25 # recommended [0.1, 0.4]
                defined_params += 'Ps_rate'.split()
            if 'P_field' not in self.params:
                self.params['P_field'] = 0.075 # recommended [0.05, 0.1]
                defined_params += 'P_field'.split()
            if 'N_field' not in self.params:
                self.params['N_field'] = 0.45 # recommended [0.4, 0.5]
                defined_params += 'N_field'.split()
        else:
            assert False, f'Unknown variant! {self.variant}'
            
        if self.processes > 1:
            self._log('Warning: EFO does not support parallelization. Defaulting to processes=1.')
            self.processes = 1

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)
      
                    
    def _init_method(self):
        """Private method for initializing the EFO optimizer instance.
        Initializes and evaluates the swarm.

        Returns
        -------
        None
            Nothing
            
        """

        self._evaluate_initial_candidates()
        
        # Generate a population
        self._EM = [Candidate(**self._candidate_init_info) for _ in range(self.params['pop_size'])]

        # Initialize
        n0 = 0 if self._initial_candidates is None else self._initial_candidates.size
        
        self._initialize_X(self._EM)
        
        # Using specified particles initial positions
        for p in range(len(self._EM)):
            if p < n0:
                self._EM[p] = self._initial_candidates[p].copy()
        
        # Evaluate
        if n0 < len(self._EM):
            self._collective_evaluation(self._EM[n0:])
        self._EM = sorted(self._EM)
        
        # if all candidates are NaNs       
        if np.isnan([c.f for c in self._EM]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'
        
        self._finalize_iteration()
        

    def _run(self):
        """Main loop of EFO method.

        Returns
        -------
        optimum: Candidate
            Best solution found during the EFO optimization.
            
        """

        if self._inject:
            self._eeeo_inject(self._EM)

        self._check_params()

        N_emp = self.params['pop_size']
        P_field = self.params['P_field']
        N_field = self.params['N_field']
        Ps_rate = self.params['Ps_rate']
        R_rate = self.params['R_rate']

        self._resuming()
                        
        RI = 0

        emNew = Candidate(**self._candidate_init_info)
        self._initialize_X([emNew])
        
        while True:

            R = emNew._R.copy()
                          
            for d, (var_name, (var_type, *var_options)) in enumerate(self.variables.items()):

                force = np.random.uniform(0,1)
                l_pos = np.random.randint(1, max(np.floor(N_emp * P_field), 2))
                l_neg = np.random.randint(np.floor((1 - N_field) * N_emp), N_emp)
                l_neu = np.random.randint(np.ceil(N_emp * P_field), np.ceil((1 - N_field) * N_emp))

                if np.random.uniform(0,1) < Ps_rate: 
                    R[d] = self._EM[l_pos]._R[d]
                else: 
                    R[d] = self._EM[l_neu]._R[d] + \
                                phi * force * (self._EM[l_pos]._R[d] - self._EM[l_neu]._R[d]) \
                                - force * (self._EM[l_neg]._R[d] - self._EM[l_neu]._R[d])

                if var_type not in \
                        [VariableType.RealPeriodic, VariableType.RealDiscretePeriodic, VariableType.IntegerPeriodic]:
                    if not 0 < R[d] < 1:
                        R[d] = np.random.uniform(0, 1)

            if np.random.uniform(0,1) < R_rate:
                R[RI] = np.random.uniform(0, 1)
                RI += 1
                if RI > self.dimensions - 1:
                    RI = 0

            emNew._R = R

            self._randomize_categorical([emNew])
            
            self._collective_evaluation([emNew])
            
            # insert emNew if better than last in list
            if emNew < self._EM[-1]:
                self._EM[-1] = emNew.copy()
            self._EM = sorted(self._EM)
                    
            if self._finalize_iteration():
                break
        
        return self.best
