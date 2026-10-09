# Safe entry-point examples

From the repository root:

```bash
python -B -S src/modeling/train_stress_temperature_physics_v1.py --help
python -B -S src/preprocessing/build_ml_dataset_v1.py --help
```

After independently obtaining lawful inputs, set PAPER1_PRIVATE_WORKSPACE to your own existing private disposable directory and run:

```bash
python -B -S src/modeling/train_stress_temperature_physics_v1.py --work-root "$PAPER1_PRIVATE_WORKSPACE" --check-inputs
```

This verifies presence and output safety only. It does not parse observations, download data, train or validate scientific results. No example experimental records are bundled.
