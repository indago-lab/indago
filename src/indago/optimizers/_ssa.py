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

File content: Definition of Squirrel Search Algorithm (SSA) optimizer.
Usage: from indago import SSA

"""


import numpy as np
from indago import Optimizer, Candidate, Status
from indago import VariableType, VariableDictType, XFormat, X_Content_Type
from scipy.special import gamma


class SSA(Optimizer):
    """Squirrel Search Algorithm method class.
    
    Reference: Jain, M., Singh, V., & Rani, A. (2019). A novel nature-inspired algorithm 
    for optimization: Squirrel search algorithm. Swarm and evolutionary computation, 
    44, 148-175.
    
    Attributes
    ----------
    variant : str
        Name of the SSA variant. Default: ``Vanilla``.
    params : dict
        A dictionary of SSA parameters.
    _swarm : ndarray
        Array of Candidate instances.
        
    Returns
    -------
    optimizer : SSA
        SSA optimizer instance.
        
    """

    def __init__(self):        
        super().__init__()


    def _check_params(self):
        """Private method which prepares the parameters to be validated by Optimizer._check_params.

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
            mandatory_params = 'pop_size ata'.split()
            # the following params are better left at default
            mandatory_params += 'p_pred c_glide gd_lim'.split() 
            if 'pop_size' not in self.params:
                self.params['pop_size'] = max(20, 2 * self.dimensions)
                defined_params += 'pop_size'.split()
            if 'ata' not in self.params:
                self.params['ata'] = 0.5
                defined_params += 'ata'.split()
            # the following params are better left at default
            if 'p_pred' not in self.params:
                self.params['p_pred'] = 0.1
                defined_params += 'p_pred'.split()
            if 'c_glide' not in self.params:
                self.params['c_glide'] = 1.9
                defined_params += 'c_glide'.split()
            if 'gd_lim' not in self.params:
                self.params['gd_lim'] = [0.5, 1.11]
                defined_params += 'gd_lim'.split()
            optional_params = ''.split()
        else:
            assert False, f'Unknown variant! {self.variant}'
            
        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)
        
    def _init_method(self):
        """Private method for initializing the SSA optimizer instance.
        Initializes and evaluates the swarm.

        Returns
        -------
        None
            Nothing
            
        """

        assert self.max_iterations or self.max_evaluations or self.max_elapsed_time, \
            'optimizer.max_iteration, optimizer.max_evaluations, or self.max_elapsed_time should be provided for this method/variant'

        # Generate a swarm of FS
        self._swarm = [Candidate(**self._candidate_init_info) for c in range(self.params['pop_size'])]
        
        # Generate initial positions
        n0 = 0 if self._initial_candidates is None else self._initial_candidates.size
        
        self._initialize_X(self._swarm)
        
        # Using specified particles initial positions
        for p in range(self.params['pop_size']):
            if p < n0:
                self._swarm[p] = self._initial_candidates[p].copy()
        
        # Evaluate
        if n0 < self.params['pop_size']:
            self._collective_evaluation(self._swarm[n0:])

        # if all candidates are NaNs       
        if np.isnan([s.f for s in self._swarm]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE.'
        
        self._finalize_iteration()
        
    def _run(self):
        """Main loop of SSA method.

        Returns
        -------
        optimum: Candidate
            Best solution found during the SSA optimization.
            
        """

        if self._inject:
            self._eeeo_inject(self._swarm)

        self._check_params()
      
        # Load params
        Pdp = self.params['p_pred']
        Gc = self.params['c_glide']
        gd_lim = self.params['gd_lim']
        ATA = self.params['ata'] # part of FSnt moving to FSat
        # ATA=0 (all move to FSht) - emphasize local search
        # ATA=1 (all move to FSat's) - emphasize global search

        self._resuming()

        while True:
            
            # Categorizing FS's
            self._swarm = sorted(self._swarm)
            FSht = self._swarm[0]  # best FS (hickory nut trees)
            FSat = self._swarm[1:4]  # good FS (acorn trees)
            FSnt = self._swarm[5:]  # bad FS (normal trees)

            # Moving FSnt
            Nnt2at = int(len(FSnt) * ATA)  # attracted to acorn trees
            np.random.shuffle(FSnt)
            for fs in FSnt[:Nnt2at]:
                if np.random.rand() >= Pdp:  # move towards FSat
                    dg = np.random.uniform(gd_lim[0], gd_lim[1])
                    # fs._R = fs._R + dg * Gc * (np.random.choice(FSat)._R - fs._R)
                    fs._R = fs._R + dg * Gc * np.random.choice(FSat)._distance_to(fs)
                else:  # respawning randomly
                    fs._R = np.random.uniform(0, 1, self.dimensions)
            for fs in FSnt[Nnt2at:]:
                if np.random.rand() >= Pdp:  # move towards FSht
                    dg = np.random.uniform(gd_lim[0], gd_lim[1])
                    # fs._R = fs._R + dg * Gc * (FSht._R - fs._R)
                    fs._R = fs._R + dg * Gc * FSht._distance_to(fs)
                else:  # respawning randomly
                    fs._R = np.random.uniform(0, 1, self.dimensions)
            self._randomize_categorical(FSnt)
            
            # Moving FSat
            for fs in FSat:
                if np.random.rand() >= Pdp:  # move towards FSht
                    dg = np.random.uniform(gd_lim[0], gd_lim[1])
                    # fs._R = fs._R + dg * Gc * (FSht._R - fs._R)
                    fs._R = fs._R + dg * Gc * FSht._distance_to(fs)
                else:  # respawning randomly
                    fs._R = np.random.uniform(0, 1, self.dimensions)
            self._randomize_categorical(FSat)
            
            # Seasonal constants (for FSat)
            Sc = np.empty(3)
            for i, fs in enumerate(FSat):
                # Sc[i] = np.sqrt(np.sum((fs._R - FSht._R)**2))
                Sc[i] = np.sqrt(np.sum(fs._distance_to(FSht) ** 2))
                
            # Minimum value of seasonal constant
            Scmin = 1e-6 / (365**(self._progress_factor() * 2.5))
            
            # Random-Levy relocation at the end of winter season
            if (Sc < Scmin).all():
                for fs in FSnt:
                    fs._R = self._Levy() * np.ones(self.dimensions)
                self._randomize_categorical(FSnt)
                
            # Evaluate swarm
            self._collective_evaluation(self._swarm)
             
            if self._finalize_iteration():
                break
        
        return self.best

    def _Levy(self):
        """Private method for generating random Levy flight.

        Returns
        -------
        None
            Nothing

        """

        ra, rb = np.random.normal(0, 1), np.random.normal(0, 1)
        beta = 1.5
        sigma = ((gamma(1 + beta) * np.sin(np.pi * beta / 2)) / \
                 gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)) ** (1 / beta)
        return 0.01 * (ra * sigma) / (np.abs(rb) ** (1 / beta))