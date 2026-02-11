# -*- coding: utf-8 -*-
"""
INDAGO OPTIMIZERS TEST
a (mostly) comprehensive test of Indago optimizers

SHOULD BE RUN AT EVERY INDAGO UPGRADE
A TEST FOR EVERY NEW OPTIMIZER PERFORMANCE-RELATED FEATURE SHOULD BE ADDED HERE
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

def test_PSO_defaults_1D_X0() -> None:
    description = 'PSO defaults, 1D X0'
    optimizer = PSO()
    optimizer.X0 = np.ones(DIM)
    expected_result = 4.142899631878027  # needs updating
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
def test_FWA_defaults() -> None:
    description = 'FWA defaults'
    optimizer = FWA()
    expected_result = 3.200343710545890
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_FWA_defaults_1D_X0() -> None:
    description = 'FWA defaults with 1D X0'
    optimizer = FWA()
    optimizer.X0 = np.zeros(DIM)
    expected_result = 4.0215439156384525
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_FWA_custom_parameters() -> None:
    description = 'FWA custom parameters'
    optimizer = FWA()
    optimizer.params['n'] = 12
    optimizer.params['m1'] = 8
    optimizer.params['m2'] = 6
    expected_result = 3.7916074172176613
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_SSA_defaults() -> None:
    description = 'SSA defaults'
    optimizer = SSA()
    expected_result = 4.557141362961143
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_SSA_custom_parameters() -> None:
    description = 'SSA custom parameters'
    optimizer = SSA()
    optimizer.params['pop_size'] = 12
    optimizer.params['ata'] = 0.8
    expected_result = 4.217644352979585
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_SSA_custom_additional_parameters() -> None:
    description = 'SSA custom additional parameters'
    optimizer = SSA()
    optimizer.params['pop_size'] = 12
    optimizer.params['p_pred'] = 0.2
    optimizer.params['c_glide'] = 1.5
    expected_result = 4.852974034102223
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_DE_defaults() -> None:
    description = 'DE defaults'
    optimizer = DE()
    expected_result = 4.06562470951338
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_DE_LSHADE_defaults() -> None:
    description = 'DE SHADE defaults'
    optimizer = DE()
    optimizer.variant = 'SHADE'
    expected_result = 4.059037126643917
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_DE_LSHADE_custom_parameters() -> None:
    description = 'DE LSHADE custom parameters'
    optimizer = DE()
    optimizer.variant = 'LSHADE'
    optimizer.params['pop_init'] = 20
    optimizer.params['f_archive'] = 2
    expected_result = 3.927170854340928
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_BA_defaults() -> None:
    description = 'BA defaults'
    optimizer = BA()
    expected_result = 5.016784239297477
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_EFO_defaults() -> None:
    description = 'EFO defaults'
    optimizer = EFO()
    expected_result = 4.345023660392395
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_MRFO_defaults() -> None:
    description = 'MRFO defaults'
    optimizer = MRFO()
    expected_result = 5.028612889414191
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_ABC_defaults() -> None:
    description = 'ABC defaults'
    optimizer = ABC()
    expected_result = 4.678938084932741
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_ABC_FullyEmployed_defaults() -> None:
    description = 'ABC FullyEmployed defaults'
    optimizer = ABC()
    optimizer.variant = 'FullyEmployed'
    expected_result = 4.945925069777881
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_ABC_Vanilla_custom_parameters() -> None:
    description = 'ABC Vanilla custom parameters'
    optimizer = ABC()
    optimizer.variant = 'Vanilla'
    optimizer.params['pop_size'] = 20
    optimizer.params['trial_limit'] = 50
    expected_result = 4.678938084932741
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_GWO_defaults() -> None:
    description = 'GWO defaults'
    optimizer = GWO()
    expected_result = 4.158491127774654
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_GWO_HSA_defaults() -> None:
    description = 'GWO HSA defaults'
    optimizer = GWO()
    optimizer.variant = 'HSA'
    expected_result = 4.218103907364332
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_MSGD_defaults() -> None:
    description = 'MSGD defaults'
    optimizer = MSGD()
    expected_result = 3.9757473965378893
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_NM_defaults() -> None:
    description = 'NM defaults'
    optimizer = NM()
    expected_result = 4.13236103332557
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_NM_Vanilla_defaults() -> None:
    description = 'NelderMead Vanilla defaults'
    optimizer = NM()
    optimizer.variant = 'Vanilla'
    expected_result = 4.1220197428515535
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'
"""

def test_RS_defaults() -> None:
    description = 'RS defaults'
    optimizer = RS()
    expected_result = 4.649421244751145
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_RS_halton_initializer() -> None:
    description = 'RS halton initializer'
    optimizer = RS()
    optimizer.sampler = 'halton'
    expected_result = 4.613384041419631
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_RS_sobol_initializer() -> None:
    description = 'RS sobol initializer'
    optimizer = RS()
    optimizer.sampler = 'sobol'
    expected_result = 4.67130384878125
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_RS_lhs_initializer() -> None:
    description = 'RS lhs initializer'
    optimizer = RS()
    optimizer.sampler = 'lhs'
    expected_result = 4.050654337618511
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

"""
def test_HBO_defaults() -> None:
    description = 'HBO defaults'
    optimizer = HBO()
    expected_result = 4.2530537129953
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_HBO_Dynamic_defaults() -> None:
    description = 'HBO Dynamic defaults'
    optimizer = HBO()
    optimizer.variant = 'Dynamic'
    expected_result = 4.555548715599656
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_CRS_defaults() -> None:
    description = 'CRS defaults'
    optimizer = CRS()
    expected_result = 4.407850741909356
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_CRS_custom_parameters_1() -> None:
    description = 'CRS custom parameters 1'
    optimizer = CRS()
    optimizer.params['pop_scale'] = 1
    expected_result = 5.119988901716537
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_CRS_custom_parameters_2() -> None:
    description = 'CRS custom parameters 2'
    optimizer = CRS()
    optimizer.params['pop_scale'] = 3.3
    expected_result = 4.609663346899482
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_EEEO_defaults() -> None:
    description = 'EEEO defaults'
    optimizer = EEEO()
    expected_result = 4.468478265803395
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_EEEO_custom_parameters() -> None:
    description = 'EEEO custom parameters'
    optimizer = EEEO()
    optimizer.methods = {'DE': ('LSHADE', {'pop_init': 30}),
                         'GWO': ('Vanilla', {'pop_size': 20})}
    expected_result = 4.235634551617493
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