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
from indagobench.problems._cec2014 import CEC2014
from indago import PSO, FWA, SSA, DE, BA, EFO, MRFO, ABC, MSGD, NM, GWO, RS, HBO, CRS, EEEO

DIM = 10
F = CEC2014(problem='F3', dimensions=DIM)
MAXEVAL = 1000
TOL = 1e-10


def run(optimizer):
    optimizer.evaluation_function = F
    optimizer.lb = F.lb
    optimizer.ub = F.ub
    optimizer.max_evaluations = MAXEVAL
    return np.log10(optimizer.optimize(seed=0).f)


# test functions

def test_PSO_defaults():
    description = 'PSO defaults'
    optimizer = PSO()
    expected_result = 4.451701712692289
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_1D_X0():
    description = 'PSO defaults, 1D X0'
    optimizer = PSO()
    optimizer.X0 = np.zeros(DIM)
    expected_result = 4.142899631978027
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_2D_X0():
    description = 'PSO defaults, 2D X0'
    optimizer = PSO()
    optimizer.X0 = np.array([1*np.ones(DIM), 2*np.ones(DIM), 3*np.ones(DIM)])
    expected_result = 4.167936861974944
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_int_X0():
    description = 'PSO defaults, int X0'
    optimizer = PSO()
    optimizer.X0 = 25
    expected_result = 4.737423558263711
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_custom_parameters():
    description = 'PSO Vanilla custom parameters'
    optimizer = PSO()
    optimizer.variant = 'Vanilla'
    optimizer.params['swarm_size'] = 10
    optimizer.params['inertia'] = 0.6
    optimizer.params['cognitive_rate'] = 2.0
    optimizer.params['social_rate'] = 2.0
    expected_result = 4.845127336717785
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_TVAC_defaults():
    description = 'PSO TVAC defaults'
    optimizer = PSO()
    optimizer.variant = 'TVAC'
    expected_result = 4.727754890370569
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_LDIW():
    description = 'PSO Vanilla LDIW'
    optimizer = PSO()
    optimizer.params['inertia'] = 'LDIW'
    expected_result = 4.899434784596884
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_HSIW():
    description = 'PSO Vanilla HSIW'
    optimizer = PSO()
    optimizer.params['inertia'] = 'HSIW'
    expected_result = 4.2897686157498995
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_anakatabatic_FlyingStork():
    description = 'PSO Vanilla anakatabatic FlyingStork'
    optimizer = PSO()
    optimizer.params['inertia'] = 'anakatabatic'
    optimizer.params['akb_model'] = 'FlyingStork'
    expected_result = 3.9438986702688554
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_anakatabatic_TipsySpider():
    description = 'PSO Vanilla anakatabatic TipsySpider'
    optimizer = PSO()
    optimizer.params['inertia'] = 'anakatabatic'
    optimizer.params['akb_model'] = 'TipsySpider'
    expected_result = 4.9856511403930694
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_anakatabatic_OrigamiSnake():
    description = 'PSO Vanilla anakatabatic OrigamiSnake'
    optimizer = PSO()
    optimizer.variant = 'TVAC'
    optimizer.params['inertia'] = 'anakatabatic'
    optimizer.params['akb_model'] = 'OrigamiSnake'
    expected_result = 4.447881816359018
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Vanilla_anakatabatic_Languid():
    description = 'PSO TVAC anakatabatic Languid'
    optimizer = PSO()
    optimizer.variant = 'TVAC'
    optimizer.params['inertia'] = 'anakatabatic'
    optimizer.params['akb_model'] = 'Languid'
    expected_result = 4.673009823249845
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_multiprocessing_on_4_processors():
    # Note: multiprocessing is slower due to pool start/stop each run
    description = 'PSO defaults, multiprocessing on 4 processors'
    optimizer = PSO()
    optimizer.processes = 4
    expected_result = 4.451701712692289
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_multiprocessing_on_maximum_processors():
    # Note: multiprocessing is slower due to pool start/stop each run
    description = 'PSO defaults, multiprocessing on maximum processors'
    optimizer = PSO()
    optimizer.processes = 'max'
    expected_result = 4.451701712692289
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Chaotic_defaults():
    description = 'PSO Chaotic defaults'
    optimizer = PSO()
    optimizer.variant = 'Chaotic'
    expected_result = 4.787600353116293
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_Chaotic_anakatabatic_Languid():
    description = 'PSO Chaotic anakatabatic Languid'
    optimizer = PSO()
    optimizer.variant = 'Chaotic'
    optimizer.params['inertia'] = 'anakatabatic'
    expected_result = 4.660721050519926
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_PSO_defaults_halton_initializer():
    description = 'PSO defaults, halton initializer'
    optimizer = PSO()
    optimizer.sampler = 'halton'
    expected_result = 3.894723171409022
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_FWA_defaults():
    description = 'FWA defaults'
    optimizer = FWA()
    expected_result = 3.200343710545890
    tolerance = 1e-4
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_FWA_defaults_1D_X0():
    description = 'FWA defaults with 1D X0'
    optimizer = FWA()
    optimizer.X0 = np.zeros(DIM)
    expected_result = 4.0215439156384525
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_FWA_custom_parameters():
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

def test_SSA_defaults():
    description = 'SSA defaults'
    optimizer = SSA()
    expected_result = 4.557141362961143
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_SSA_custom_parameters():
    description = 'SSA custom parameters'
    optimizer = SSA()
    optimizer.params['pop_size'] = 12
    optimizer.params['ata'] = 0.8
    expected_result = 4.217644352979585
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
    expected_result = 4.852974034102223
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_DE_defaults():
    description = 'DE defaults'
    optimizer = DE()
    expected_result = 4.06562470951338
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_DE_LSHADE_defaults():
    description = 'DE SHADE defaults'
    optimizer = DE()
    optimizer.variant = 'SHADE'
    expected_result = 4.059037126643917
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
    expected_result = 3.927170854340928
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_BA_defaults():
    description = 'BA defaults'
    optimizer = BA()
    expected_result = 5.016784239297477
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_EFO_defaults():
    description = 'EFO defaults'
    optimizer = EFO()
    expected_result = 4.345023660392395
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_MRFO_defaults():
    description = 'MRFO defaults'
    optimizer = MRFO()
    expected_result = 5.028612889414191
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_ABC_defaults():
    description = 'ABC defaults'
    optimizer = ABC()
    expected_result = 4.678938084932741
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_ABC_FullyEmployed_defaults():
    description = 'ABC FullyEmployed defaults'
    optimizer = ABC()
    optimizer.variant = 'FullyEmployed'
    expected_result = 4.945925069777881
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
    expected_result = 4.678938084932741
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_GWO_defaults():
    description = 'GWO defaults'
    optimizer = GWO()
    expected_result = 4.158491127774654
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_GWO_HSA_defaults():
    description = 'GWO HSA defaults'
    optimizer = GWO()
    optimizer.variant = 'HSA'
    expected_result = 4.218103907364332
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_MSGD_defaults():
    description = 'MSGD defaults'
    optimizer = MSGD()
    expected_result = 3.9757473965378893
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_NM_defaults():
    description = 'NM defaults'
    optimizer = NM()
    expected_result = 4.13236103332557
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_NM_Vanilla_defaults():
    description = 'NelderMead Vanilla defaults'
    optimizer = NM()
    optimizer.variant = 'Vanilla'
    expected_result = 4.1220197428515535
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_RS_defaults():
    description = 'RS defaults'
    optimizer = RS()
    expected_result = 4.707063872301493
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_RS_halton_initializer():
    description = 'RS halton initializer'
    optimizer = RS()
    optimizer.sampler = 'halton'
    expected_result = 4.964551645464896
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_RS_sobol_initializer():
    description = 'RS sobol initializer'
    optimizer = RS()
    optimizer.sampler = 'sobol'
    expected_result = 4.893882682936462
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_RS_lhs_initializer():
    description = 'RS lhs initializer'
    optimizer = RS()
    optimizer.sampler = 'lhs'
    expected_result = 5.024877929981968
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_HBO_defaults():
    description = 'HBO defaults'
    optimizer = HBO()
    expected_result = 4.2530537129953
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_HBO_Dynamic_defaults():
    description = 'HBO Dynamic defaults'
    optimizer = HBO()
    optimizer.variant = 'Dynamic'
    expected_result = 4.555548715599656
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_CRS_defaults():
    description = 'CRS defaults'
    optimizer = CRS()
    expected_result = 4.407850741909356
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_CRS_custom_parameters_1():
    description = 'CRS custom parameters 1'
    optimizer = CRS()
    optimizer.params['pop_scale'] = 1
    expected_result = 5.119988901716537
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_CRS_custom_parameters_2():
    description = 'CRS custom parameters 2'
    optimizer = CRS()
    optimizer.params['pop_scale'] = 3.3
    expected_result = 4.609663346899482
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_EEEO_defaults():
    description = 'EEEO defaults'
    optimizer = EEEO()
    expected_result = 4.468478265803395
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'

def test_EEEO_custom_parameters():
    description = 'EEEO custom parameters'
    optimizer = EEEO()
    optimizer.methods = {'DE': ('LSHADE', {'pop_init': 30}),
                         'GWO': ('Vanilla', {'pop_size': 20})}
    expected_result = 4.235634551617493
    tolerance = TOL
    result = run(optimizer)
    assert expected_result - tolerance < result < expected_result + tolerance, \
        f'{description} FAILED, result={result}, expected={expected_result}'


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
    test_FWA_defaults()
    test_FWA_defaults_1D_X0()
    test_FWA_custom_parameters()
    test_SSA_defaults()
    test_SSA_custom_parameters()
    test_SSA_custom_additional_parameters()
    test_DE_defaults()
    test_DE_LSHADE_defaults()
    test_DE_LSHADE_custom_parameters()
    test_BA_defaults()
    test_EFO_defaults()
    test_MRFO_defaults()
    test_ABC_defaults()
    test_ABC_FullyEmployed_defaults()
    test_ABC_Vanilla_custom_parameters()
    test_GWO_defaults()
    test_GWO_HSA_defaults()
    test_MSGD_defaults()
    test_NM_defaults()
    test_NM_Vanilla_defaults()
    test_RS_defaults()
    test_RS_halton_initializer()
    test_RS_sobol_initializer()
    test_RS_lhs_initializer()
    test_HBO_Dynamic_defaults()
    test_HBO_defaults()
    test_CRS_defaults()
    test_CRS_custom_parameters_1()
    test_CRS_custom_parameters_2()
    test_EEEO_defaults()
    test_EEEO_custom_parameters()