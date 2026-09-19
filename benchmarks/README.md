# Numerical and performance benchmarks

This directory contains the scripts used to evaluate numerical agreement and
computational performance for MixSIAPy. Generated posterior files and benchmark
working directories are intentionally excluded from the repository because they
are large and can be recreated from the distributed example data.

The numerical-agreement workflow is driven by `run_numerical_agreement.py`, with
model definitions in `numerical_configs.py` and implementation-specific runners
in `numerical_runner_python.py` and `numerical_runner_r.R`.

The performance workflow uses `run_performance_benchmarks.py` and
`performance_runner.py`. GPU chain-scaling experiments are implemented in
`run_gpu_chain_scaling.py`. Summary and plotting scripts operate on the generated
benchmark tables.

Run these scripts from the repository root in an environment containing the
required optional inference backends. R comparisons additionally require R,
MixSIAR, and JAGS. Hardware, software versions, sampling schedules, and diagnostic
criteria should be recorded with every benchmark run.
