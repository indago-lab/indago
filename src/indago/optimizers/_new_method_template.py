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

File content: Definition of My New Method (MNM) optimizer.
Usage: from indago import MNM

"""


import numpy as np
from pip._internal.models import candidate

from indago import Optimizer, Candidate, Status
from indago import VariableType, VariableDictType, XFormat, X_Content_Type


# SAVE FILE AS _mnm.py ("my new method")
# ADD TO __init__.py: from indago.optimizers._mnm import MNM


class MyCandidate(Candidate):
    """MNM MyCandidate class. MNM MyCandidate is a specific member of a MNM population.

    This class is used to expand the features/properties of the Candidate class for MNM.
    If there is no need for this, remove this class entirely and just use Candidate instead.

    Attributes
    ----------
    a : ndarray
        MyCandidate attribute.

    Returns
    -------
    candidate : MyCandidate
        MyCandidate instance.
    """

    def __init__(self, variables: VariableDictType, n_objectives: int = 1, n_constraints: int = 0,
                 x_format: XFormat = XFormat.Tuple) -> None:

        super().__init__(variables, n_objectives, n_constraints, x_format)

        self.a = 0


class MNM(Optimizer):
    """My New Method method class.

    Reference: (citation of paper on which the implementation is based)
    
    Attributes
    ----------
    variant : str
        Name of the MNM variant (``Vanilla`` or ``SomeVariant``). Default: ``Vanilla``.
    params : dict
        A dictionary of MNM parameters.
    _pop : list
        Solution candidates.

    Returns
    -------
    optimizer : MNM
        MNM optimizer instance.
    """
    

    def __init__(self):
        super().__init__()


    def _check_params(self):
        """Private method which performs some MNM-specific parameter checks
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
            mandatory_params = 'param1 param2'.split()

            if 'param1' in self.params:
                self.params['param1'] = int(self.params['param1'])
                assert self.params['param1'] > 0, \
                    "param1 parameter should be positive integer"
            else:
                self.params['param1'] = int(max(self.dimensions, 10))
            defined_params += 'param1'.split()

            if 'param2' in self.params:
                self.params['param2'] = int(self.params['param2'])
                assert self.params['param2'] > 0, \
                    "levels parameter should be positive float"
            else:
                self.params['param2'] = 3
            defined_params += 'param2'.split()

        elif self.variant == 'SomeVariant':
            ...

        else:
            assert False, f'Unknown variant! {self.variant}'

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)


    def _init_method(self):
        """Private method for initializing the MNM optimizer instance.
        Initializes and evaluates the candidate population.

        Returns
        -------
        None
            Nothing
            
        """

        # Leave this if your method uses self._progress_factor(), remove otherwise
        assert self.max_iterations or self.max_evaluations or self.max_elapsed_time, \
            'optimizer.max_iteration, optimizer.max_evaluations, or optimizer.max_elapsed_time should be provided for this method/variant'

        # Parameter checks can go here, e.g.
        assert self.params['param1'] >= 2, \
            'ParameterA (param1 param) should be greater than or equal to 2'
        assert self.params['param2'] >= 1, \
            'ParameterB (param2 param) should be greater than or equal to 1'

        # Generate population
        popsize = self.params['param1'] ** 2  # or some other population size definition
        self._pop = [MyCandidate(**self._candidate_init_info) for _ in range(popsize)]  # use Candidate instead of MyCandidate if possible

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
        if n0 < popsize:
            self._collective_evaluation(self._pop[n0:])

        # if all candidates are NaNs
        if np.isnan([point.f for point in self._pop]).all():
            self._err_msg = 'ALL CANDIDATES FAILED TO EVALUATE'

        # Sort (if needed)
        self._pop = sorted(self._pop)

        # Whatever other preparation is needed
        ...

        # Done
        self._finalize_iteration()


    def _run(self):
        """Run procedure for the MNM method.

        Returns
        -------
        optimum: MyCandidate
            Best solution found during the MNM optimization.
            
        """

        # If your method supports EEEO-hybridization, use the following code, otherwise remove this if intermittent
        # injecting of an externally generated candidate into the population would strongly disrupt the method
        if self._inject:
            new = self._eeeo_inject(self._pop)
            # if you need to do something special with the newly injected solution (new), do it here
            ...

        # Checking user/default-defined parameters
        self._check_params()

        # If optimization is resumed
        self._resuming()

        while True:

            ##### MAIN LOOP #####
            # Method algorithm goes here

            # Use candidate._R (not candidate.X) for changing candidate's position
            # R is a real 1d ndarray in [0,1] representing X as mapped to a unit hypercube
            R = candidate._R.copy()
            R = ...
            candidate._R = R

            # Whenever you change a candidate's position, randomize its categorical variables (or otherwise treat them somehow)
            self._randomize_categorical(list_of_changed_candidates)

            # If you need information on method progress in terms of iterations/evaluations spent
            T = self._progress_factor()  # goes from 0 to 1 over the course of the optimization

            # When you prepare the list of candidates which must be evaluated
            self._collective_evaluation(list_of_candidates)

            # When checking which candidate is better, do not use fitness/objectives, use logic operators, e.g.
            if candidate1 < candidate2:
                candidate1 = candidate2.copy()
                self.initialize_X([candidate2])

            # If you must sort a candidate list
            self._pop = sorted(self._pop)

            # Iteration done - all checks and administration are performed now
            if self._finalize_iteration():
                break

        # Optimization finished
        return self.best