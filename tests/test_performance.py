# -*- coding: utf-8 -*-
"""
INDAGO OPTIMIZERS PERFORMANCE TEST
a (mostly) comprehensive test of Indago optimizers performance

A TEST FOR EVERY NEW METHOD/FEATURE PERFORMANCE SHOULD BE ADDED HERE
"""

# need this for local (non-pip) install only
import sys
sys.path.append('..')
sys.path.append('../indagobench')

import numpy as np
from indago import PSO, RS #, FWA, SSA, DE, BA, EFO, MRFO, ABC, MSGD, NM, GWO, HBO, CRS, EEEO


def F(x):
    """CEC f3 - Discus Function"""
    return 1e6 * x[0] ** 2 + np.sum(x[1:] ** 2)

DIM = 10
MAXEVAL = 1000
TOL = 1e-10


def run(optimizer):
    optimizer.evaluator = F
    optimizer.dimensions = DIM
    optimizer.lb = -100
    optimizer.ub = 100
    optimizer.max_evaluations = MAXEVAL
    return np.log10(optimizer.optimize(seed=0).f)


# test functions

def test_PSO_defaults() -> None:
    description = 'PSO defaults'
    optimizer = PSO()
    expected_result = 2.5793297920299136
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_solution_given():
    description = 'PSO defaults, solution given (1D X0)'
    optimizer = PSO()
    optimizer.X0 = np.zeros(DIM)
    expected_result = 0.0
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_1D_X0() -> None:
    description = 'PSO defaults, 1D X0'
    optimizer = PSO()
    optimizer.X0 = np.ones(DIM)
    expected_result = 2.517708163727869
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_2D_X0() -> None:
    description = 'PSO defaults, 2D X0'
    optimizer = PSO()
    optimizer.X0 = np.array([1*np.ones(DIM), 2*np.ones(DIM), 3*np.ones(DIM)])
    expected_result = 0.37301715942874253
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_int_X0() -> None:
    description = 'PSO defaults, int X0'
    optimizer = PSO()
    optimizer.X0 = 25
    expected_result = 2.6704917369893884
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_custom_parameters() -> None:
    description = 'PSO Vanilla custom parameters'
    optimizer = PSO()
    optimizer.variant = 'Vanilla'
    optimizer.params['swarm_size'] = 10
    optimizer.params['inertia'] = 0.6
    optimizer.params['cognitive_rate'] = 2.0
    optimizer.params['social_rate'] = 2.0
    expected_result = 4.026320535708186
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_TVAC_defaults() -> None:
    description = 'PSO TVAC defaults'
    optimizer = PSO()
    optimizer.variant = 'TVAC'
    expected_result = 3.2573071465104713
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_LDIW() -> None:
    description = 'PSO Vanilla LDIW'
    optimizer = PSO()
    optimizer.params['inertia'] = 'LDIW'
    expected_result = 3.304806494663155
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_HSIW() -> None:
    description = 'PSO Vanilla HSIW'
    optimizer = PSO()
    optimizer.params['inertia'] = 'HSIW'
    expected_result = 2.5417401479737785
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_anakatabatic_FlyingStork() -> None:
    description = 'PSO Vanilla anakatabatic FlyingStork'
    optimizer = PSO()
    optimizer.params['inertia'] = 'anakatabatic'
    optimizer.params['akb_model'] = 'FlyingStork'
    expected_result = 3.1457484365150234
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_anakatabatic_TipsySpider() -> None:
    description = 'PSO Vanilla anakatabatic TipsySpider'
    optimizer = PSO()
    optimizer.params['inertia'] = 'anakatabatic'
    optimizer.params['akb_model'] = 'TipsySpider'
    expected_result = 3.3754146413536024
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_anakatabatic_OrigamiSnake() -> None:
    description = 'PSO Vanilla anakatabatic OrigamiSnake'
    optimizer = PSO()
    optimizer.variant = 'TVAC'
    optimizer.params['inertia'] = 'anakatabatic'
    optimizer.params['akb_model'] = 'OrigamiSnake'
    expected_result = 2.822197390624899
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_anakatabatic_Languid() -> None:
    description = 'PSO TVAC anakatabatic Languid'
    optimizer = PSO()
    optimizer.variant = 'TVAC'
    optimizer.params['inertia'] = 'anakatabatic'
    optimizer.params['akb_model'] = 'Languid'
    expected_result = 3.8318983720461195
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_anakatabatic_DoubleSummit() -> None:
    description = 'PSO TVAC anakatabatic DoubleSummit'
    optimizer = PSO()
    optimizer.variant = 'TVAC'
    optimizer.params['inertia'] = 'anakatabatic'
    optimizer.params['akb_model'] = 'DoubleSummit'
    expected_result = 4.0967505955847
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_multiprocessing_on_4_processors() -> None:
    # Note: multiprocessing is slower due to pool start/stop each run
    description = 'PSO defaults, multiprocessing on 4 processors'
    optimizer = PSO()
    optimizer.processes = 4
    expected_result = 2.5793297920299136
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_multiprocessing_on_maximum_processors() -> None:
    # Note: multiprocessing is slower due to pool start/stop each run
    description = 'PSO defaults, multiprocessing on maximum processors'
    optimizer = PSO()
    optimizer.processes = 'max'
    expected_result = 2.5793297920299136
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Chaotic_defaults() -> None:
    description = 'PSO Chaotic defaults'
    optimizer = PSO()
    optimizer.variant = 'Chaotic'
    expected_result = 4.273075440664452
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Chaotic_anakatabatic_Languid() -> None:
    description = 'PSO Chaotic anakatabatic Languid'
    optimizer = PSO()
    optimizer.variant = 'Chaotic'
    optimizer.params['inertia'] = 'anakatabatic'
    expected_result = 4.027548994755644
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_halton_initializer() -> None:
    description = 'PSO defaults, halton initializer'
    optimizer = PSO()
    optimizer.sampler = 'halton'
    expected_result = 2.559935460137834
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

"""
def test_FWA_defaults():
    description = 'FWA defaults'
    optimizer = FWA()
    expected_result = -14.3552157847343
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_FWA_custom_parameters():
    description = 'FWA custom parameters'
    optimizer = FWA()
    optimizer.params['n'] = 12
    optimizer.params['m1'] = 8
    optimizer.params['m2'] = 6
    expected_result = -12.384148004226958
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_SSA_defaults():
    description = 'SSA defaults'
    optimizer = SSA()
    expected_result = 4.850766721900909
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_SSA_custom_parameters():
    description = 'SSA custom parameters'
    optimizer = SSA()
    optimizer.params['pop_size'] = 12
    optimizer.params['ata'] = 0.8
    expected_result = 4.438743024215149
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_SSA_custom_additional_parameters():
    description = 'SSA custom additional parameters'
    optimizer = SSA()
    optimizer.params['pop_size'] = 12
    optimizer.params['p_pred'] = 0.2
    optimizer.params['c_glide'] = 1.5
    expected_result = 4.47154424234304
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_DE_defaults():
    description = 'DE defaults'
    optimizer = DE()
    expected_result = 2.3996066775068625
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_DE_LSHADE_defaults():
    description = 'DE SHADE defaults'
    optimizer = DE()
    optimizer.variant = 'SHADE'
    expected_result = 3.3448851150990944
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_DE_LSHADE_custom_parameters():
    description = 'DE LSHADE custom parameters'
    optimizer = DE()
    optimizer.variant = 'LSHADE'
    optimizer.params['pop_init'] = 20
    optimizer.params['f_archive'] = 2
    expected_result = 1.4083783202220084
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_BA_defaults():
    description = 'BA defaults'
    optimizer = BA()
    expected_result = 4.213229947486668
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_EFO_defaults():
    description = 'EFO defaults'
    optimizer = EFO()
    expected_result = 1.01888834670826
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_MRFO_defaults():
    description = 'MRFO defaults'
    optimizer = MRFO()
    expected_result = 3.265045494474331
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_ABC_defaults():
    description = 'ABC defaults'
    optimizer = ABC()
    expected_result = 4.33976322802834
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_ABC_FullyEmployed_defaults():
    description = 'ABC FullyEmployed defaults'
    optimizer = ABC()
    optimizer.variant = 'FullyEmployed'
    expected_result = 3.716602431815081
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_ABC_Vanilla_custom_parameters():
    description = 'ABC Vanilla custom parameters'
    optimizer = ABC()
    optimizer.variant = 'Vanilla'
    optimizer.params['pop_size'] = 20
    optimizer.params['trial_limit'] = 50
    expected_result = 4.33976322802834
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_GWO_defaults():
    description = 'GWO defaults'
    optimizer = GWO()
    expected_result = -3.6171180468126125
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_GWO_HSA_defaults():
    description = 'GWO HSA defaults'
    optimizer = GWO()
    optimizer.variant = 'HSA'
    expected_result = -7.0488083215685196
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_MSGD_defaults():
    description = 'MSGD defaults'
    optimizer = MSGD()
    expected_result = 0  # TODO: -inf in old Indago
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_NM_defaults():
    description = 'NM defaults'
    optimizer = NM()
    expected_result = 4.197968360852144
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_NM_Vanilla_defaults():
    description = 'NelderMead Vanilla defaults'
    optimizer = NM()
    optimizer.variant = 'Vanilla'
    expected_result = 4.099420846773123
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'
"""

def test_RS_defaults():
    description = 'RS defaults'
    optimizer = RS()
    expected_result = 4.649421244751145
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_RS_halton_initializer():
    description = 'RS halton initializer'
    optimizer = RS()
    optimizer.sampler = 'halton'
    expected_result = 4.613384041419631
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_RS_sobol_initializer():
    description = 'RS sobol initializer'
    optimizer = RS()
    optimizer.sampler = 'sobol'
    expected_result = 4.67130384878125
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_RS_lhs_initializer():
    description = 'RS lhs initializer'
    optimizer = RS()
    optimizer.sampler = 'lhs'
    expected_result = 4.050654337618511
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

"""
def test_HBO_defaults():
    description = 'HBO defaults'
    optimizer = HBO()
    expected_result = 4.166997012436769
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_HBO_Dynamic_defaults():
    description = 'HBO Dynamic defaults'
    optimizer = HBO()
    optimizer.variant = 'Dynamic'
    expected_result = 4.72960644554505
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_CRS_defaults():
    description = 'CRS defaults'
    optimizer = CRS()
    expected_result = 4.437560183138608
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_CRS_custom_parameters_1():
    description = 'CRS custom parameters 1'
    optimizer = CRS()
    optimizer.params['pop_scale'] = 1
    expected_result = 4.525681028714209
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_CRS_custom_parameters_2():
    description = 'CRS custom parameters 2'
    optimizer = CRS()
    optimizer.params['pop_scale'] = 3.3
    expected_result = 4.778298365107319
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_EEEO_defaults():
    description = 'EEEO defaults'
    optimizer = EEEO()
    expected_result = -2.825058356190423
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_EEEO_custom_parameters():
    description = 'EEEO custom parameters'
    optimizer = EEEO()
    optimizer.methods = {'DE': ('LSHADE', {'pop_init': 30}),
                         'GWO': ('Vanilla', {'pop_size': 20})}
    expected_result = 1.6084339646218162
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'
"""

# stand-alone testing
if __name__ == '__main__':

    print(f'Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
    print(f'Numpy version: {np.__version__}')
    print('Running tests ...')

    test_PSO_defaults()
    test_PSO_defaults_solution_given()
    test_PSO_defaults_1D_X0()
    test_PSO_defaults_2D_X0()
    test_PSO_defaults_int_X0()
    test_PSO_Vanilla_custom_parameters()
    test_PSO_TVAC_defaults()
    test_PSO_Vanilla_LDIW()
    test_PSO_Vanilla_HSIW()
    test_PSO_Vanilla_anakatabatic_FlyingStork()
    test_PSO_Vanilla_anakatabatic_TipsySpider()
    test_PSO_Vanilla_anakatabatic_OrigamiSnake()
    test_PSO_Vanilla_anakatabatic_Languid()
    test_PSO_defaults_multiprocessing_on_4_processors()
    test_PSO_defaults_multiprocessing_on_maximum_processors()
    test_PSO_Chaotic_defaults()
    test_PSO_Chaotic_anakatabatic_Languid()
    test_PSO_defaults_halton_initializer()
    # test_FWA_defaults()
    # test_FWA_defaults_1D_X0()
    # test_FWA_custom_parameters()
    # test_SSA_defaults()
    # test_SSA_custom_parameters()
    # test_SSA_custom_additional_parameters()
    # test_DE_defaults()
    # test_DE_LSHADE_defaults()
    # test_DE_LSHADE_custom_parameters()
    # test_BA_defaults()
    # test_EFO_defaults()
    # test_MRFO_defaults()
    # test_ABC_defaults()
    # test_ABC_FullyEmployed_defaults()
    # test_ABC_Vanilla_custom_parameters()
    # test_GWO_defaults()
    # test_GWO_HSA_defaults()
    # test_MSGD_defaults()
    # test_NM_defaults()
    # test_NM_Vanilla_defaults()
    test_RS_defaults()
    test_RS_halton_initializer()
    test_RS_sobol_initializer()
    test_RS_lhs_initializer()
    # test_HBO_Dynamic_defaults()
    # test_HBO_defaults()
    # test_CRS_defaults()
    # test_CRS_custom_parameters_1()
    # test_CRS_custom_parameters_2()
    # test_EEEO_defaults()
    # test_EEEO_custom_parameters()