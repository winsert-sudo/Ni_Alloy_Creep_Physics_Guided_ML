#!/usr/bin/env python3
"""Create frozen source-aware and random-reference splits for creep modeling."""
from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split


PROJECT_ROOT = _CANDIDATE_ROOT
REQUIRED_COLUMNS = [
    "creep_id", "source_id", "material_state_id", "experiment_family_id",
    "transfer_tier", "mechanism_class", "alloy_family",
]
TIERS = ["Tier 1", "Tier 2", "Tier 3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path", type=Path,
        default=PROJECT_ROOT / "03_MODELING/data/baseline_creep_dataset_v1p1.csv",
        help="Input ML dataset CSV.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "03_MODELING/splits",
        help="Directory for split assignments, summaries, and metadata.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of source-aware GroupKFold folds.")
    parser.add_argument("--random-seed", type=int, default=42, help="Fixed seed for the random reference split.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.fillna("").to_markdown(index=False)


def domain_distribution(frame: pd.DataFrame, fold_id: int, split: str) -> dict[str, object]:
    counts = frame["transfer_tier"].value_counts()
    return {
        "fold_id": fold_id,
        "split": split,
        "Tier 1": int(counts.get("Tier 1", 0)),
        "Tier 2": int(counts.get("Tier 2", 0)),
        "Tier 3": int(counts.get("Tier 3", 0)),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Input dataset is missing required columns: {missing_columns}")
    if data["creep_id"].isna().any() or not data["creep_id"].is_unique:
        raise ValueError("creep_id must be non-missing and unique.")
    if data["source_id"].isna().any() or data["source_id"].eq("").any():
        raise ValueError("source_id must be populated for every record.")
    source_count = int(data["source_id"].nunique())
    if source_count < args.n_splits:
        raise ValueError(f"{source_count} source groups cannot support {args.n_splits} folds.")
    logging.info("Loaded %d samples from %d source groups.", len(data), source_count)

    # Primary split: a source is assigned to exactly one validation fold.
    assignments = pd.DataFrame({"fold_id": -1}, index=data.index)
    group_kfold = GroupKFold(n_splits=args.n_splits)
    fold_rows, domain_rows, alloy_rows = [], [], []
    for fold_id, (train_index, test_index) in enumerate(group_kfold.split(data, groups=data["source_id"])):
        assignments.loc[test_index, "fold_id"] = fold_id
        train, test = data.loc[train_index], data.loc[test_index]
        train_sources, test_sources = set(train["source_id"]), set(test["source_id"])
        overlap = train_sources & test_sources
        if overlap:
            raise ValueError(f"Source leakage in GroupKFold fold {fold_id}: {sorted(overlap)}")
        fold_rows.append({
            "fold_id": fold_id,
            "train_samples": len(train),
            "test_samples": len(test),
            "train_sources": len(train_sources),
            "test_sources": len(test_sources),
            "shared_sources": len(overlap),
        })
        domain_rows.extend([domain_distribution(train, fold_id, "train"), domain_distribution(test, fold_id, "test")])
        for split, frame in (("train", train), ("test", test)):
            for alloy_family, count in frame["alloy_family"].value_counts().sort_index().items():
                alloy_rows.append({"fold_id": fold_id, "split": split, "alloy_family": alloy_family, "sample_count": int(count)})

    if assignments["fold_id"].lt(0).any() or not assignments["fold_id"].between(0, args.n_splits - 1).all():
        raise ValueError("GroupKFold did not assign each sample to exactly one validation fold.")
    group_assignment = data[["creep_id", "source_id", "material_state_id"]].copy()
    group_assignment["fold_id"] = assignments["fold_id"].astype(int)
    group_assignment.to_csv(args.output_dir / "source_groupkfold_5fold_assignment.csv", index=False)
    fold_statistics = pd.DataFrame(fold_rows)
    domain_statistics = pd.DataFrame(domain_rows)
    alloy_statistics = pd.DataFrame(alloy_rows)

    # Reference only: deliberately allows a source to occur in both partitions.
    train_index, test_index = train_test_split(
        data.index, test_size=0.2, random_state=args.random_seed, shuffle=True,
    )
    random_assignment = data[["creep_id", "source_id"]].copy()
    random_assignment["random_split"] = "train"
    random_assignment.loc[test_index, "random_split"] = "test"
    random_assignment.to_csv(args.output_dir / "random_80_20_assignment.csv", index=False)
    random_train = data.loc[train_index]
    random_test = data.loc[test_index]
    shared_random_sources = set(random_train["source_id"]) & set(random_test["source_id"])
    if len(random_assignment) != len(data) or random_assignment["random_split"].isna().any():
        raise ValueError("Random split did not assign every sample.")

    source_summary = data.groupby("source_id", sort=True).size().rename("sample_count").reset_index()
    source_group_summary = f"""# Source-aware GroupKFold (5-fold) Summary

## Overall

- Dataset: `{args.input_path.name}`
- Total samples: **{len(data)}**
- Total sources: **{source_count}**
- Number of folds: **{args.n_splits}**
- Grouping column: `source_id`
- Validation convention: each record is validation data exactly once; training is the remaining four folds.

## Fold statistics

{markdown_table(fold_statistics)}

All `shared_sources` counts are zero: source leakage validation **PASS** for every fold.

## Domain distribution

{markdown_table(domain_statistics)}

## Alloy-family distribution

{markdown_table(alloy_statistics)}

## Source-group sizes

{markdown_table(source_summary)}

## Interpretation

`GroupKFold` balances sample counts as far as its whole-source constraint permits; it does not stratify transfer tiers, mechanism classes, or alloy families. Folds with sparse Tier 1 or Tier 3 coverage must be reported alongside aggregate cross-validation results rather than interpreted as interchangeable domains.
"""
    (args.output_dir / "source_groupkfold_5fold_summary.md").write_text(source_group_summary, encoding="utf-8")

    random_summary = pd.DataFrame([
        {"random_split": "train", "samples": len(random_train), "sources": random_train["source_id"].nunique()},
        {"random_split": "test", "samples": len(random_test), "sources": random_test["source_id"].nunique()},
    ])
    random_split_summary = f"""# Random 80/20 Reference Split Summary

## Configuration

- Dataset: `{args.input_path.name}`
- Implementation: `sklearn.model_selection.train_test_split`
- Test fraction: **0.2**
- Random seed: **{args.random_seed}**

## Partition sizes

{markdown_table(random_summary)}

## Source leakage (expected for reference split)

- Shared source IDs across random train/test: **{len(shared_random_sources)}** of {source_count}
- Shared source IDs: `{", ".join(sorted(shared_random_sources))}`

This split is a deliberately optimistic reference because observations from the same source can occur in both train and test. It must not replace the source-aware GroupKFold protocol for the scientific primary result.
"""
    (args.output_dir / "random_split_summary.md").write_text(random_split_summary, encoding="utf-8")

    metadata = {
        "dataset": args.input_path.stem,
        "dataset_sha256": sha256(args.input_path),
        "group_column": "source_id",
        "group_kfold": args.n_splits,
        "group_kfold_implementation": "sklearn.model_selection.GroupKFold",
        "random_split_implementation": "sklearn.model_selection.train_test_split",
        "random_test_size": 0.2,
        "random_seed": args.random_seed,
        "n_samples": len(data),
        "n_source_groups": source_count,
        "groupkfold_source_leakage": False,
        "random_shared_source_count": len(shared_random_sources),
        "required_metadata_columns": REQUIRED_COLUMNS,
        "assignment_files": {
            "source_groupkfold": "source_groupkfold_5fold_assignment.csv",
            "random_reference": "random_80_20_assignment.csv",
        },
        "construction_script": "03_MODELING/scripts/create_splits_v1.py",
    }
    (args.output_dir / "split_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote split assignments, summaries, and metadata to %s", args.output_dir)


if __name__ == "__main__":
    main()
