# -*- coding: utf-8 -*-
"""
INDAGO ENUMs TEST

"""

# need this for local (non-pip) install only
import sys
sys.path.append('..')

import indago

def test_variable_type_categories():
    assert indago.VariableType.REAL.is_real()
    assert indago.VariableType.REAL_PERIODIC.is_real()
    assert indago.VariableType.REAL_DISCRETE.is_real()
    assert indago.VariableType.REAL_DISCRETE_PERIODIC.is_real()
    assert not indago.VariableType.INTEGER.is_real()
    assert not indago.VariableType.CATEGORICAL.is_real()

    assert indago.VariableType.REAL_DISCRETE.is_discrete()
    assert indago.VariableType.REAL_DISCRETE_PERIODIC.is_discrete()
    assert indago.VariableType.INTEGER.is_discrete()
    assert indago.VariableType.CATEGORICAL.is_discrete()
    assert not indago.VariableType.REAL.is_discrete()

    assert indago.VariableType.INTEGER.is_integer()
    assert indago.VariableType.INTEGER_PERIODIC.is_integer()
    assert not indago.VariableType.REAL.is_integer()
    assert not indago.VariableType.CATEGORICAL.is_integer()

    assert indago.VariableType.REAL_DISCRETE_PERIODIC.is_periodic()
    assert indago.VariableType.INTEGER_PERIODIC.is_periodic()
    assert not indago.VariableType.REAL.is_periodic()
    assert not indago.VariableType.CATEGORICAL.is_periodic()
