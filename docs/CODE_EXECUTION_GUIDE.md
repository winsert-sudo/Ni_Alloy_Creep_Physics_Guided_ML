# Code execution and verification guide

These instructions distinguish non-scientific candidate verification from possible future execution with lawful private inputs. No model, figure or manuscript pipeline was run during remediation. Original algorithms and outputs were retained in reviewed copies; all future generated outputs remain private.

From the candidate root, summary/integrity verification requires only Python's standard library:

```bash
python -B src/evaluation/verify_public_summaries.py
```

This checks candidate SHA-256 inventory and three aggregate headline alignments. It does not recompute metrics from observations or run a model.

For help without scientific dependencies or inputs:

```bash
python -B -S src/modeling/train_stress_temperature_physics_v1.py --help
python -B -S src/preprocessing/build_ml_dataset_v1.py --help
```

The common operational wrapper accepts --work-root and --check-inputs. Every entry's [input contract](../src/CODE_INPUTS.json) lists exact relative inputs, output destinations, peer modules and retained original CLI options. A work root is an existing, private, disposable directory supplied by the researcher, outside this candidate and frozen/submission archives. The wrapper never creates missing inputs, invents values or changes computations. Missing inputs yield a clear error before scientific imports/writes; existing output destinations are refused.

Only after lawful access and source reconstruction, a user may check presence without reading experimental rows:

```bash
python -B -S src/modeling/train_stress_temperature_physics_v1.py --work-root "$PAPER1_PRIVATE_WORKSPACE" --check-inputs
```

PAPER1_PRIVATE_WORKSPACE must be explicitly set to the user's own private disposable directory; no real directory or source row is supplied by this package. Presence success does not validate schemas, frozen hashes/folds, scientific results or access rights.

The source-to-model chain is relational inputs → build_ml_dataset_v1.py → build_chemistry_features_v1p1.py → create_splits_v1.py → train_stress_temperature_physics_v1.py. The headline full script additionally requires lawful row-linked oof_error_table.csv and temperature_stress_coverage.csv for its original diagnostics. Their creation is not fully supplied here: the original source-specific audit is excluded. For exact reproduction, use the exact immutable split map; newly generated folds on reconstructed rows/ordering may differ.

The remaining reviewed scripts cover Tier1 and transfer baselines, classical/state diagnostics, unified comparisons, and original figure/manuscript/table assembly. Peer modules live together in src/modeling, including the original NumPy NN dependency; no PyTorch dependency is required by that implementation. Dependencies are those in the reconstructed environment record, not a claim of recovered historical versions. Figure/manuscript assembly executes original top-level code and must be invoked as a standalone script; import-only testing is inappropriate for those entries.

A lawful clean run has not been demonstrated by this task. Help/missing-input checks, syntax/import review and whole-module AST comparison provide operational/equivalence evidence without numerical reruns. Do not call the public-only candidate fully reproducible. Do not copy private features, predictions/residuals, state parameters, figures/SVGs, logs or model outputs into the public tree.
