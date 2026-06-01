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

File content: Definition of Fireworks Algorithm (FWA) optimizer.
Usage: from indago import FWA

"""


import numpy as np
from indago import Optimizer, Candidate, Status
from indago import VariableType, VariableDictType, XFormat, X_Content_Type


class FWA(Optimizer):
    """Fireworks Algorithm method class.
    
    Attributes
    ----------
    variant : str
        Name of the FWA variant (``Vanilla`` or ``Rank``). Default: ``Rank``.
    params : dict
        A dictionary of FWA parameters.
    _sparks : list
        Solution candidates.
        
    Returns
    -------
    optimizer : FWA
        FWA optimizer instance.
        
    """

    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some FWA-specific parameter checks
        and prepares the parameters to be validated by Optimizer._check_params.

        Returns
        -------
        None
            Nothing
            
        """

        if not self.variant:
            self.variant = 'Rank'
        
        defined_params = list(self.params.keys())
        mandatory_params, optional_params = [], []
        
        if 'n' in self.params:
            self.params['n'] = int(self.params['n'])
        if 'm1' in self.params:
            self.params['m1'] = int(self.params['m1'])
        if 'm2' in self.params:
            self.params['m2'] = int(self.params['m2'])

        if self.variant == 'Vanilla':
            mandatory_params = 'n m1 m2'.split()
            if 'n' not in self.params:
                self.params['n'] = self.dimensions
                defined_params += 'n'.split()
            if 'm1' not in self.params:
                self.params['m1'] = self.dimensions // 2
                defined_params += 'm1'.split()
            if 'm2' not in self.params:
                self.params['m2'] = self.dimensions // 2
                defined_params += 'm2'.split()                
            optional_params = ''.split()
        elif self.variant == 'Rank':
            mandatory_params = 'n m1 m2'.split()
            if 'n' not in self.params:
                self.params['n'] = self.dimensions
                defined_params += 'n'.split()
            if 'm1' not in self.params:
                self.params['m1'] = self.dimensions // 2
                defined_params += 'm1'.split()
            if 'm2' not in self.params:
                self.params['m2'] = self.dimensions // 2
                defined_params += 'm2'.split()
            optional_params = ''.split()
        else:
            assert False, f'Unknown variant \'{self.variant}\''
        
        if self.constraints > 0:
            assert self.variant == 'Rank', f"Variant '{self.variant}' does not support constraints! Use 'Rank' variant instead"

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)

    def _init_method(self):
        """Private method for initializing the FWA optimizer instance.
        Initializes and evaluates the swarm.

        Returns
        -------
        None
            Nothing
            
        """

        self._evaluate_initial_candidates()

        self._sparks = [Candidate(**self._candidate_init_info) for p in range(self.params['n'])]
        
        # Generate initial positions
        n0 = 0 if self._initial_candidates is None else self._initial_candidates.size
        
        self._initialize_X(self._sparks)
        
        # Using specified initial positions
        for p in range(self.params['n']):
            if p < n0:
                self._sparks[p] = self._initial_candidates[p].copy()

        # Evaluate all
        if n0 < self.params['n']:
            self._collective_evaluation(self._sparks[n0:])

        # if all candidates are NaNs       
        if np.isnan([p.f for p in self._sparks]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'

        # Sort
        self._sparks = sorted(self._sparks)
        
        self._finalize_iteration()


    def _run(self):
        """Main loop of FWA method.

        Returns
        -------
        optimum: Candidate
            Best solution found during the FWA optimization.
            
        """

        if self._inject:
            self._eeeo_inject(self._sparks)

        self._check_params()

        self._resuming()
        
        n = self.params['n']

        while True:

            self._sparks += self._explosion() + self._gaussian_mutation()
            self._collective_evaluation(self._sparks[n:])
            self._sparks = sorted(self._sparks)[:n]

            if self._finalize_iteration():
                break
        
        return self.best


    def _explosion(self):
        """Private method for computing explosion sparks.

        Returns
        -------
        explosion_sparks : list of Candidate
            FWA explosion sparks.
        """
        
        eps = 0.001
        amp = 10
        a = 0.01
        b = 10

        F = np.array([c.f for c in self._sparks])
        fmin = np.nanmin(F)
        fmax = np.nanmax(F)
        
        explosion_sparks: list[Candidate] = []

        for p in range(self.params['n']):
               
            cFw = self._sparks[p].copy()
            
            if self.variant == 'Vanilla':
                # Number of sparks
                n1 = self.params['m1'] * (fmax - cFw.f + eps) / np.sum(fmax - F + eps)
                n1 = self._min_max_round(n1, self.params['m1'] * a, self.params['m2'] * b)
                
                # Amplitude
                A = amp * (cFw.f - fmin + eps) /  (np.sum(F - fmin) + eps)

                R = cFw._R.copy()
                for j in range(n1):
                    for k in range(self.dimensions):
                        if np.random.choice([True, False]):
                            R[k] += np.random.uniform(-A, A)
                    cFw._R = R
                    explosion_sparks.append(cFw.copy())
                
            if self.variant == 'Rank':
                
                # Number of sparks
                n1 = self.params['m1'] * (self.params['n'] - p)**1 / np.sum(np.arange(self.params['n']+1)**1)
                n1 = np.random.choice([int(np.floor(n1)), int(np.ceil(n1))])

                Rs = np.array([c._R for c in self._sparks])
                # print(Rs.shape)
                
                # Uniform
                dev = np.std(Rs, 0)
                avg_scale = np.average(np.sqrt(np.arange(self.params['n']) + 1))
                scale = np.sqrt(p + 1) / avg_scale
                
                # avg_scale = np.average(np.arange(self.params['n']) + 1)
                # scale = (p + 1) / avg_scale

                A = np.sqrt(12) / 2 * dev * scale
                A *= 1.5

                R = cFw._R.copy()
                for j in range(n1):
                    R += np.random.uniform(-A, A) * np.random.randint(0, 1, A.size)
                    
                    for k in range(self.dimensions):
                        if np.random.choice([True, False]):
                            R[k] += np.random.uniform(-A[k], A[k])
                            # R[k] += np.random.normal(-A[k], A[k])
                    
                    cFw._R = R
                    explosion_sparks.append(cFw.copy())

        self._randomize_categorical(explosion_sparks)

        return explosion_sparks


    def _gaussian_mutation(self):
        """Private method for computing mutation sparks.

        Returns
        -------
        mutation_sparks : list of Candidate
            FWA mutation sparks.
        """
        
        mutation_sparks: list[Candidate] = []

        for j in range(self.params['m2']):
            cFw = self._sparks[np.random.randint(self.params['n'])].copy()
            g = np.random.normal(1, 1)
            R = cFw._R.copy()
            for k in range(self.dimensions):
                if np.random.choice([True, False]):
                    # cFw.X[k] *= g
                    R[k] *= g
            cFw._R = R
            mutation_sparks.append(cFw)

        self._randomize_categorical(mutation_sparks)

        return mutation_sparks


    def _min_max_round(self, s, smin, smax):
        """Private method for calculating round of min of max of input parameters.

        Parameters
        ----------
        s : float
            Preliminary population size.
        smin : float
            Preliminary population size.
        smax : float
            Preliminary population size.
             
        Returns
        -------
        min_max_round : int
            Round of min of max of input parameters.
        """
        
        return int(np.round(np.min([np.max([s, smin]), smax])))
