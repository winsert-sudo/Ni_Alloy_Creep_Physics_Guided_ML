# Paper 1 Software Versions

## Reconstructed publication/reproduction environment

| Component | Version |
|---|---|
| Environment | `paper1-creep-env` |
| Python | 3.12.14 |
| NumPy | 2.5.3 |
| pandas | 3.0.5 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1, CPU build |
| Matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| joblib | 1.6.0 |
| PyYAML | 6.0.3 |
| tabulate | 0.10.0 |

## Computational platform

| Item | Recorded value |
|---|---|
| OS/kernel | Linux 4.18.0-477.10.1.el8_8 |
| Architecture | x86_64 |
| libc | glibc 2.28 |
| BLAS | OpenBLAS 0.3.34, pthreads build |
| XGBoost accelerator | CPU |
| Conda channel | conda-forge |

## Historical inference

Frozen TASK 04.7-B2 and TASK 04.7-C0 run configurations recorded Python 3.12.7 on the same Linux/glibc platform, and the project contains CPython 3.12 bytecode. Exact historical library versions were not recorded. TASK 05.7-E therefore reconstructs a new tested Python 3.12 environment; it does not claim recovery of the original package set.

## Verification

The unmodified `train_stress_temperature_physics_v1.py` workflow was executed in an isolated copy of the required inputs. The reconstructed environment reproduced all 220 headline-model OOF predictions exactly and reproduced the stored full-precision R², MAE, and RMSE with zero difference. This is a prior internal verification record, not a rerun performed by this candidate construction. The restricted inputs and historical verification report are not bundled.


This is the reconstructed publication/reproduction environment, not a recovered historical training environment. Installing it alone does not enable exact numerical reproduction. No installation or training was performed during candidate construction.
