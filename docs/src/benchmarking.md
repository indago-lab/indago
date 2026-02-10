<script src="https://kit.fontawesome.com/067013af35.js" crossorigin="anonymous"></script>

# Benchmarking
In order to analyze the performance of Indago methods, we designed **IndagoBench**, a benchmark suite designed to reflect real-world engineering optimization scenarios and assess the practical performance of modern algorithms. In addition to reporting achieved objective values, it employs nonlinear normalization based on random sampling. The suite is under continuous development as new engineering problems and optimization methods are incorporated, with planned extensions to discrete-variable, multi-objective, and constrained optimization tasks.

**IndagoBench is not published with Indago**, but as separate code collection periodically whenever a sufficient number of new problems or methods are added, or when methodological improvements are introduced. Each version is accompanied by complete result files for the entire benchmark. Because it is under active development, these published milestones ensure the reproducibility of results.

## IndagoBench25

The initial problems set and benchmark results as presented in _**Randomness as Reference: Benchmark Metric for Optimization in Engineering**_ by Stefan Ivić, Siniša Družeta, and Luka Grbčić. The preprint of the paper is available at [arXiv](https://arxiv.org/abs/2511.17226) while supplementary materials including source code and the results are available on [OSF repository](https://osf.io/7r6jz).

To cite the paper, please use the following:
```bibtex
@misc{indagobench25,
      title={Randomness as Reference: Benchmark Metric for Optimization in Engineering},
      author={Stefan Ivić and Siniša Družeta and Luka Grbčić},
      year={2025},
      eprint={2511.17226},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2511.17226},
}
```

Indagobench25 consists of 231 unconstrained bounded continuous optimization problems, with dimensionality ranging from 3D to 58D.

The optimization methods implemented in Indago were tested on IndagoBench25 and key results are shown below. For a more detailed analysis consult the paper.

### Overall method performance

```{image} images/post_analysis_method_scores_performance.png
:alt: Indago methods performance
:width: 60%
:align: center
```

On the left side of the above figure you can see the overall performance scores of Indago methods in terms of *G*-score. The *G*-score is defined in the range [-1, 1], where zero stands for performing exactly like Random Search (i.e. poorly), and one for always finding the best solutions. 

On the right side you can see the frequency in which the methods showed excellent (*G* > 0.9) and best of all (*G* = max *G*) performance, as well as catastrophic (*G* < 0) and worst of all (*G* = min *G*) performance.

### Complementary attributes of analyzed methods

```{image} images/radar_plot_indago_optimizers.png
:alt: Indago methods complementary attributes
:width: 100%
:align: center
```

Here you can see the weighted *G*-scores in terms of three attribute-axes corresponding to simulation scenario type: global-local search, high-low dimensionality, and fast-exhaustive search. The grey shadows indicates the increase of performance in a 10-run optimization batch. 

### Method sensitivity to problem multimodality

```{image} images/post_analysis_indago_G_M.png
:alt: Method score as related to problem multimodality
:width: 45%
:align: center
```

In this figure you can see the degradation of the *G*-score (y-axis) as related to the increase of the problem multimodality (x-axis). The dotted line stands for *G* = 0. This indicates how well a method is expected to perform on highly global problems.

### Overview of method features

```{image} images/post_analysis_indago_method_features.png
:alt: Indago methods features
:width: 100%
:align: center
```

This figure gives encompassing information on various properties of Indago methods.

### Best performing method ensembles

```{image} images/post_analysis_indago_method_sets.png
:alt: Best ensembles of Indago methods
:width: 35%
:align: center
```
These are the best performing method ensembles, according to IndagoBench25 testing. Note that some of these methods (GWO, SSA, BA), even otherwise not particularly outstanding, can still be useful when using several optimization methods in a wider analysis. However, there are no guarantees, and for any given problem these methods could show to be insufficient.

## Guidelines

According to the conducted research, a number of guidelines can be given. Note that these are all statistically based, and could be fairly ineffective for your particular problem. Still, it is often useful to have some kind of direction, no matter how non-definitive it may be.

So, here are some recommendations on which methods to use and when:

- If you suspect the problem is highly multimodal (i.e. global), your first choice should be LSHADE, PSO, or EFO.
- If you believe the problem is local (i.e. near-convex), use LSHADE, MSGD, or NM.
- If your goal function is very computationally expensive, use PSO or FWA.
- If your goal function is relatively cheap, use LSHADE or EFO.
- If you cannot afford very many optimization runs, use LSHADE or PSO.
- If global methods (LSHADE, EFO, PSO, FWA, etc.) failed you, try using local methods (NM, MSGD) across many repeated runs.
- If nothing else seemed to work, try with FWA or GWO.

Happy optimizing!
