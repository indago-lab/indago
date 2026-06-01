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

File content: Definition of Differential Evolution (DE) optimizer.
Usage: from indago import DE

"""


import numpy as np
from indago import Optimizer, Candidate, Status
from indago import VariableType, VariableDictType, XFormat, X_Content_Type
from scipy.stats import cauchy


class Solution(Candidate):
    """DE solution class.
    
    Attributes
    ----------
    V : ndarray
        Mutant vector.  
    CR : float
        DE control parameter CR.
    F : float
        DE control parameter F.
         
    Returns
    -------
    solution : Solution
        Solution instance.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.CR = None
        self.F = None
        self.V = np.full(len(self._variables), np.nan)  # mutant vector


class DE(Optimizer):
    """Differential Evolution method class.
    
    Reference: R. Tanabe and A. S. Fukunaga. Improving the search performance of SHADE using 
    linear population size reduction”, in Proceedings of the 2014 IEEE Congress on 
    Evolutionary Computation (CEC), pp. 1658–1665, Beijing, China, July 2014.
    
    Attributes
    ----------
    variant : str
        Name of the DE variant (``SHADE`` or ``LSHADE``). Default: ``LSHADE``.
    params : dict
        A dictionary of DE parameters.
    _Pop : ndarray
        Array of Solution instances.
    _Trials : ndarray
        Trial solutions.
    _A : ndarray
        Archive.
    _M_CR : ndarray
        M_CR values.
    _M_F : ndarray
        M_F values.
    _k : int
        Memory index.
        
    Returns
    -------
    optimizer : DE
        DE optimizer instance.
        
    """

    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some DE-specific parameter checks
        and prepares the parameters to be validated by Optimizer._check_params.

        Returns
        -------
        None
            Nothing
            
        """

        if not self.variant:
            self.variant = 'LSHADE'
        
        defined_params = list(self.params.keys())
        mandatory_params, optional_params = [], []
        
        if 'pop_init' in self.params:
            self.params['pop_init'] = int(self.params['pop_init'])
        if 'hist_size' in self.params:
            self.params['hist_size'] = int(self.params['hist_size'])

        if self.variant == 'SHADE':
            mandatory_params = 'pop_init f_archive hist_size p_mutation'.split()
            if 'pop_init' not in self.params:
                self.params['pop_init'] = max(30, self.dimensions * 5)
                defined_params += 'pop_init'.split()
            if 'f_archive' not in self.params:
                self.params['f_archive'] = 2.6
                defined_params += 'f_archive'.split()
            if 'hist_size' not in self.params:  # aka H
                self.params['hist_size'] = 6
                defined_params += 'hist_size'.split()
            if 'p_mutation' not in self.params:
                self.params['p_mutation'] = 0.11
                defined_params += 'p_mutation'.split()    
            optional_params = 'rank_enabled'.split()
            if 'rank_enabled' not in self.params:
                self.params['rank_enabled'] = False # Rank-based variant off by default
                defined_params += 'rank_enabled'.split()  
        elif self.variant == 'LSHADE':
            mandatory_params = 'pop_init f_archive hist_size p_mutation'.split()
            if 'pop_init' not in self.params:
                self.params['pop_init'] = max(30, self.dimensions * 5)
                defined_params += 'pop_init'.split()
            if 'f_archive' not in self.params:
                self.params['f_archive'] = 2.6
                defined_params += 'f_archive'.split()
            if 'hist_size' not in self.params:  # aka H
                self.params['hist_size'] = 6
                defined_params += 'hist_size'.split()
            if 'p_mutation' not in self.params:
                self.params['p_mutation'] = 0.11
                defined_params += 'p_mutation'.split()  
            optional_params = 'rank_enabled'.split()
            if 'rank_enabled' not in self.params:
                self.params['rank_enabled'] = False  # Rank-based variant off by default
                defined_params += 'rank_enabled'.split()
        else:
            assert False, f'Unknown variant! {self.variant}'
        
        if self.constraints > 0:
            assert False, 'DE does not support constraints'
        
        assert isinstance(self.params['pop_init'], int) \
            and self.params['pop_init'] > 0, \
            "pop_init should be positive integer"
        assert self.params['f_archive'] > 0, \
            "external_archive_size should be positive"
        assert isinstance(self.params['hist_size'], int) \
            and self.params['hist_size'] > 0, \
            "hist_size should be positive integer"

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)
        
    def _init_method(self):
        """Private method for initializing the DE optimizer instance.
        Initializes and evaluates the population.

        Returns
        -------
        None
            Nothing
            
        """

        if self.variant == 'LSHADE':
            assert self.max_iterations or self.max_evaluations or self.max_elapsed_time, \
                'Error: optimizer.max_iteration, optimizer.max_evaluations, or self.max_elapsed_time should be provided for this method/variant'

        self._evaluate_initial_candidates()

        # Generate a population
        self._Pop: list[Solution] = [Solution(**self._candidate_init_info) for c in range(self.params['pop_init'])]
        
        # Generate a trial population
        self._Trials: list[Solution] = [Solution(**self._candidate_init_info) for c in range(self.params['pop_init'])]
        
        # Initalize Archive
        self._A = []
        
        # Prepare historical memory
        self._M_CR = np.full(self.params['hist_size'], 0.5)
        self._M_F = np.full(self.params['hist_size'], 0.5)

        # Generate initial positions
        n0 = 0 if self._initial_candidates is None else self._initial_candidates.size
        
        self._initialize_X(self._Pop)
        
        # Using specified particles initial positions
        for i in range(self.params['pop_init']):
            if i < n0:
                self._Pop[i] = self._initial_candidates[i].copy()

        # Evaluate
        if n0 < self.params['pop_init']:
            self._collective_evaluation(self._Pop[n0:])

        # if all candidates are NaNs
        if np.isnan([p.f for p in self._Pop]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'
        
        self._finalize_iteration()
        
    def _run(self):
        """Main loop of DE method.

        Returns
        -------
        optimum: Solution
            Best solution found during the DE optimization.
        """

        if self._inject:
            self._eeeo_inject(self._Pop)

        self._check_params()

        if not self._resuming():
            self._k = 0  # memory index

        while True:
            
            S_CR = np.empty([0])
            S_F = np.empty([0])
            S_df = np.empty([0])
            
            # find pbest
            top = max(round(len(self._Pop) * self.params['p_mutation']), 1)
            pbest = np.random.choice(sorted(self._Pop)[0:top])
            
            if self.params['rank_enabled']:
                self._Pop = sorted(self._Pop)
            
            for p, t in zip(self._Pop, self._Trials):
                
                # Update CR, F
                r = np.random.randint(self.params['hist_size'])
                if np.isnan(self._M_CR[r]):
                    p.CR = 0
                else:
                    p.CR = np.random.normal(self._M_CR[r], 0.1)
                    p.CR = np.clip(p.CR, 0, 1)
                p.F = -1
                while p.F <= 0:
                    p.F = min(cauchy.rvs(self._M_F[r], 0.1), 1)
                
                # Compute mutant vector
                r1 = r2 = p
                while r1 is r2 or r1 is p or r2 is p:
                    r1 = np.random.choice(self._Pop)
                    r2 = np.random.choice(self._Pop + self._A)
                p.V = p._R + p.F * (pbest._R - p._R) + p.F * (r1._R - r2._R)
                p.V = np.clip(p.V, (p._R + 0)/2, (p._R + 1)/2)
                
                # Compute trial vector
                t.CR = p.CR
                t.F = p.F
                jrand = np.random.randint(self.dimensions)
                R = t._R.copy()
                for j in range(self.dimensions):
                    if np.random.rand() <= p.CR or j == jrand:
                        R[j] = p.V[j]
                    else:
                        R[j] = p._R[j]
                t._R = R

            self._randomize_categorical(self._Trials)

            # Evaluate population
            self._collective_evaluation(self._Trials)
            
            # Survival for next generation
            if not self.params['rank_enabled']:
                for p, t in zip(self._Pop, self._Trials):
                    if not np.isnan(t.f) and t.f < p.f:
                        # Update external archive
                        self._A.append(p)
                        if len(self._A) > round(len(self._Pop) * self.params['f_archive']):
                            # self._A = np.delete(self._A, np.random.randint(np.size(self._A)))
                            self._A.remove(np.random.choice(self._A))
                        S_CR = np.append(S_CR, t.CR) 
                        S_F = np.append(S_F, t.F)
                        S_df = np.append(S_df, p.f - t.f)
                        # Update population
                        p._R = np.copy(t._R)
                        p.f = t.f
                        p.O, p.C = np.copy(t.O), np.copy(t.C)
            
            # Rank-based variant - very poor performance
            if self.params['rank_enabled']: 
                # p_ranking = np.argsort(self._Pop)
                p_ranking = np.arange(len(self._Pop))
                # self._Trials = np.sort(self._Trials)
                t_ranking = np.argsort(self._Trials)
                # Calculating rank-based improvement of trials
                S_df = (p_ranking - t_ranking) / (1 + p_ranking)**2
                S_df = S_df[S_df > 0]
                # S_df = 1.0 + S_df ** 2                                               
                for p, t, p_rank, t_rank in zip(self._Pop, self._Trials, p_ranking, t_ranking):
                    if t_rank < p_rank:                      
                        # Update external archive
                        self._A.append(p)
                        if len(self._A) > round(len(self._Pop) * self.params['f_archive']):
                            # self._A = np.delete(self._A, np.random.randint(np.size(self._A)))
                            self._A.remove(np.random.choice(self._A))
                        S_CR = np.append(S_CR, t.CR) 
                        S_F = np.append(S_F, t.F)
                        #S_df = np.append(S_df, p_rank - t_rank)
                        # Update population
                        p._R = np.copy(t._R)
                        p.f = t.f 
                        p.O, p.C = np.copy(t.O), np.copy(t.C)

            # Memory update
            if np.size(S_CR) != 0 and np.size(S_F) != 0:
                w = S_df / np.sum(S_df)
                if np.isnan(self._M_CR[self._k]) or np.max(S_CR) < 1e-100:
                    self._M_CR[self._k] = np.nan
                else:
                    self._M_CR[self._k] = np.sum(w * S_CR**2) / np.sum(w * S_CR)
                self._M_F[self._k] = np.sum(w * S_F**2) / np.sum(w * S_F)
                self._k += 1
                if self._k >= self.params['hist_size']:
                    self._k = 0
                    
            # Linear Population Size Reduction (LPSR)
            if self.variant == 'LSHADE':
                N_init = self.params['pop_init']
                N_new = round((4 - N_init) * self._progress_factor() + N_init)
                if N_new < len(self._Pop):
                    self._Pop = sorted(self._Pop)[:N_new]
                    self._Trials = self._Trials[:N_new]
                
            if self._finalize_iteration():
                break
        
        return self.best
