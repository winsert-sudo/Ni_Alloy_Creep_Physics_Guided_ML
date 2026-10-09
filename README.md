# Physics-Guided ML for Ni-Based Alloy Creep

Public source code is available at [winsert-sudo/Ni_Alloy_Creep_Physics_Guided_ML](https://github.com/winsert-sudo/Ni_Alloy_Creep_Physics_Guided_ML) under BSD-3-Clause. The initial publication commit `bb99d5d72211c8962c55fef499e3cb2be0dfa4c1` contains 56 files. Version `v1.0.0` is proposed; no version tag or GitHub Release has been created.

This code supports source-aware cross-alloy creep rupture prediction using target-free physics coordinates. The target is log10(creep rupture life / h). The unpublished paper title and verified author order are in [CITATION.cff](CITATION.cff); no DOI or publication date is assigned.

Frozen documentation reference: Tier 1 n=220, source-aware five-fold pooled OOF, P3-replace XGBoost; R²=0.7497006307687231, MAE=0.3592029929766931, RMSE=0.4930923099654113. These are screened aggregate references, not a newly executed benchmark.

Install dependencies in a separate environment using `python -m pip install -r requirements.txt`. See environment/software_versions.md for the reconstructed environment and its historical-version limitations. Installation was not repeated during packaging.

| Directory | Contents |
|---|---|
| src/ | Audited preprocessing, modeling, evaluation, figure code and operational preflight |
| examples/ | Help and lawful-input preflight instructions; no experimental sample rows |
| docs/ | Methods, feature definitions, validation, schemas and execution guide |
| metadata/ | Bibliographic/service locators; no experimental reference/heat keys |
| tests/ | Standard-library package integrity and syntax checks |
| results_summary/ | Previously screened aggregate results and configurations |
| environment/ | Dependency/environment records |
| manifests/ | Complete content inventory and checksums |

Run `python -B tests/check_package.py` for non-scientific integrity checks from a clean source archive/export without `.git`. The existing verifier inventories every file, including Git administrative files when run directly in a clone; see the [execution guide](docs/CODE_EXECUTION_GUIDE.md) for export-based verification. Run `python -B -S src/modeling/train_stress_temperature_physics_v1.py --help` for CLI help. Scientific execution requires lawful compatible inputs in a private disposable workspace outside this tree; see [execution guide](docs/CODE_EXECUTION_GUIDE.md) and [data access](DATA_ACCESS.md).

Supported code includes relational preprocessing, chemistry features, grouped splits, physics-coordinate ablations, transfer and classical/state diagnostics. Exact frozen inputs, folds and some source-specific ancillary construction are unavailable here. New splits on independently reconstructed data may differ. [Reproducibility](REPRODUCIBILITY.md) distinguishes code inspection from exact benchmark reproduction.

No raw workbooks, observation/feature tables, exact fold assignments, OOF rows, row-bearing figures or trained model artifacts are bundled. All future numerical outputs must remain private pending applicable rights clearance. The corresponding author reports institutional authorization, BSD-3-Clause approval and approval of the R1 payload. See [LICENSE](LICENSE) and [license scope](LICENSE_SCOPE.md). The public source-code repository was published under the author’s explicit authorization. The software license grants no rights over third-party data. See [release status and data restrictions](RELEASE_STATUS.md).

Code availability: the BSD-3-Clause source code is public at https://github.com/winsert-sudo/Ni_Alloy_Creep_Physics_Guided_ML. No archival DOI or versioned GitHub Release has been assigned. Exact public-only benchmark reproduction remains unavailable because restricted inputs and frozen assignments are withheld. These documentation updates prepare the proposed `v1.0.0` release locally; they have not been pushed.
