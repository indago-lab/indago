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

File content: Definition of Synchronous Cooperation of Optimizers (SynCO) optimizer.
Usage: from indago import SynCO

"""


import numpy as np
from indago.core._optimizer import Optimizer, OptimizerStatus
from indago.core._candidate import X_Content_Type
from indago import Candidate, VariableType, VariableDictType, XFormat
from indago import optimizers_dict


class SynCO(Optimizer):
    """Synchronous Cooperation of Optimizers method class.

    Synchronous Cooperation of Optimizers (SynCO) runs a selection of optimizers in
    parallel and after each iteration injects the overall-best found solution into
    the employed optimizers.

    Attributes
    ----------
    variant : str
        Name of the SynCO variant. Default: ``Vanilla``.
    methods : dict or None
        Indago methods (variant, params) to use. Default: ``{'PSO': (None, None),
        'FWA': (None, None)}``. ``None`` values for variant and params will activate the
        corresponding default variant and params.
    _optimizers : list of Optimizer subclass objects
        Private list of optimizers used in SynCO.

    Returns
    -------
    optimizer : SynCO
        SynCO optimizer instance.

    """

    def __init__(self):
        super().__init__()

        self.methods = None

    def _check_params(self):
        """Private method which performs some SynCO-specific parameter checks
        and prepares the parameters to be validated by Optimizer._check_params.

        Returns
        -------
        None
            Nothing

        """

        if not self.variant:
            self.variant = 'Vanilla'

        if not self.methods:
            self.methods: dict = {'PSO': (None, None),
                                  'FWA': (None, None)}

        assert len(self.methods) >= 2, \
            'optimizer.methods should provide at least 2 optimization methods'

        for method in self.methods:
            assert method in 'ABC DE NM FWA GWO PSO RS HBO CRS EFO SSA'.split(), \
                'SynCO does not support {method} at this time'

        defined_params = list(self.params.keys())
        mandatory_params, optional_params = [], []

        if self.variant == 'Vanilla':
            pass

        else:
            assert False, f'Unknown variant! {self.variant}'

        Optimizer._check_params(self, mandatory_params, optional_params, defined_params)

    def _init_method(self):
        """Private method for initializing the SynCO optimizer instance.

        Returns
        -------
        None
            Nothing

        """

        # Prepare optimizers
        self._optimizers = []

        for opt_name, (variant, params) in self.methods.items():

            opt = optimizers_dict[opt_name]()
            opt.variant = variant
            opt.params = params if params else {}

            # parallel evaluation
            opt.processes = max(1, self.processes // len(self.methods))

            # pass parameters
            opt.evaluator = self.evaluator
            if len(self.variables) > 0:
                opt.variables = self.variables
                opt.lb, opt.ub = None, None
            else:
                opt.lb, opt.ub = self.lb, self.ub
            opt.dimensions = self.dimensions
            opt.X0 = self.X0
            opt.sampler = self.sampler

            # check parameters
            opt._check_params()

            # pass progress information
            opt._progress_factor = self._progress_factor

            self._optimizers.append(opt)

        # Initialize SynCO best
        self.best = None

    def _run(self):
        """Main loop of SynCO method.

        Returns
        -------
        optimum: Candidate
            Best solution found during the SynCO optimization.

        """

        self._check_params()

        self._resuming()

        evals = [0] * len(self.methods)

        prev_bests = []

        while True:

            resume = True if self.it > 0 else False

            bests = []
            for i, opt in enumerate(self._optimizers):
                opt.max_iterations = self.it + 1
                if len(opt.variables) > 0:
                    opt.lb, opt.ub = None, None
                opt.optimize(resume=resume,
                             inject=[c for c in prev_bests if c < opt.best] if prev_bests else None,
                             seed=self._seed)
                bests.append(opt.best)
                self.eval += opt.eval - evals[i]
                evals[i] = opt.eval

            self.best = np.min(bests)
            prev_bests = [c for c in bests]

            self._update_history()

            if self._finalize_iteration():
                break

        return self.best
