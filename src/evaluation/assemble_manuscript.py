#!/usr/bin/env python3
"""Assemble manuscript from frozen Markdown components."""

from __future__ import annotations

# Operational entry-point preflight; scientific functions below are unchanged.
import sys as _candidate_sys
from pathlib import Path as _CandidatePath
_candidate_sys.path.insert(0, str(_CandidatePath(__file__).resolve().parents[1]))
from _candidate_paths import configure as _candidate_configure
_CANDIDATE_ROOT = _candidate_configure(__file__, run_as_main=(__name__ == '__main__'))

import re
from pathlib import Path


BASE = _CANDIDATE_ROOT / "05_PAPER_CONVERGENCE"
OUT = BASE / "MANUSCRIPT_FINAL_DRAFT_V1.md"


def read(name: str) -> str:
    return (BASE / name).read_text(encoding="utf-8")


def between(text: str, start: str, end: str | None = None) -> str:
    part = text.split(start, 1)[1]
    if end is not None:
        part = part.split(end, 1)[0]
    return part.strip()


def split_sections(text: str, heading_level: int) -> list[str]:
    marker = "#" * heading_level
    return [part.strip() for part in re.split(rf"(?=^{marker} )", text, flags=re.M) if part.strip()]


def table_block(filename: str) -> str:
    lines = read(f"manuscript_tables/{filename}").strip().splitlines()
    title = lines[0].removeprefix("# ")
    body = "\n".join(line for line in lines[1:] if not line.startswith("Source: ")).strip()
    return f"#### {title}\n\n{body}"


def caption_map() -> dict[str, tuple[str, str]]:
    text = read("FIGURE_CAPTIONS.md")
    matches = re.finditer(
        r"^## (Figure (S?\d+)\. ([^\n]+))\n\n(.*?)(?=^## |\Z)",
        text,
        flags=re.M | re.S,
    )
    return {m.group(2): (m.group(1), m.group(4).strip()) for m in matches}


CAPTIONS = caption_map()
FIGURE_STEMS = {
    "1": "figures/main/Fig1_database_validation",
    "2": "figures/main/Fig2_ml_benchmark",
    "3": "figures/main/Fig3_transfer_limitation",
    "4": "figures/main/Fig4_state_dependent_physics",
    "5": "figures/main/Fig5_state_parameter_recovery",
    "6": "figures/main/Fig6_scientific_synthesis",
    "S1": "figures/supplementary/FigS1_dataset_distributions",
    "S2": "figures/supplementary/FigS2_split_comparison",
    "S3": "figures/supplementary/FigS3_residual_comparison",
    "S4": "figures/supplementary/FigS4_nn_benchmark",
    "S5": "figures/supplementary/FigS5_feature_interpretation",
}


def figure_block(key: str) -> str:
    title, caption = CAPTIONS[key]
    stem = FIGURE_STEMS[key]
    return (
        f"> **{title.split('.', 1)[0]} placement:** "
        f"[PDF]({stem}.pdf) | [SVG]({stem}.svg) | [PNG]({stem}.png)\n\n"
        f"**{title}**\n\n{caption}"
    )


title_text = re.search(r"\*\*(.+?)\*\*", between(read("TITLE_FINAL.md"), "## Final title", "## Running title")).group(1)

abstract = between(read("ABSTRACT_DRAFT_V1.md"), "## Structured abstract", "## Word count")
abstract = re.sub(r"^### (.+)$", r"**\1**", abstract, flags=re.M)

introduction = between(read("INTRODUCTION_DRAFT_V1.md"), "## 1. Introduction", "## Draft metadata")

methods = between(read("METHODS_DRAFT_V1.md"), "## 2. Methods", "## Draft metadata")
method_parts = split_sections(methods, 3)
assembled_methods: list[str] = []
for part in method_parts:
    assembled_methods.append(part)
    if part.startswith("### 2.1 "):
        assembled_methods.append(table_block("Table_1_dataset_summary.md"))
        assembled_methods.append(
            "**Table 1 descriptor note.** Available descriptor groups are test conditions "
            "(temperature and stress), alloy and source identifiers, composition, mechanism class, "
            "heat-treatment identifier, and chemistry missingness/completeness metadata."
        )
    if part.startswith("### 2.2 "):
        assembled_methods.append(figure_block("1"))

results = between(read("RESULTS_DRAFT_V1.md"), "# 3. Results")
result_parts = split_sections(results, 2)
result_assets = {
    "## 3.1 ": ("Table_2_model_benchmark.md", "2"),
    "## 3.2 ": ("Table_3_transfer_learning.md", "3"),
    "## 3.3 ": ("Table_4_physics_models.md", "4"),
    "## 3.4 ": ("Table_5_parameter_learning.md", "5"),
}
assembled_results: list[str] = []
for part in result_parts:
    assembled_results.append(re.sub(r"^## ", "### ", part, count=1))
    for prefix, (table_name, figure_key) in result_assets.items():
        if part.startswith(prefix):
            assembled_results.append(table_block(table_name))
            assembled_results.append(figure_block(figure_key))
            break

discussion = between(read("DISCUSSION_DRAFT_V1.md"), "# 4. Discussion")
discussion_parts = split_sections(discussion, 2)
assembled_discussion: list[str] = []
for part in discussion_parts:
    assembled_discussion.append(re.sub(r"^## ", "### ", part, count=1))
    if part.startswith("## 4.3 "):
        assembled_discussion.append(figure_block("6"))

conclusions = between(read("CONCLUSIONS_DRAFT_V1.md"), "# 5. Conclusions")

supplementary = [
    ("S1 Dataset distribution", "S1"),
    ("S2 Validation comparison", "S2"),
    ("S3 Additional model results", "S3"),
    ("S4 Neural network benchmark", "S4"),
    ("S5 Feature interpretation", "S5"),
]
supplementary_text = "\n\n".join(f"### {heading}\n\n{figure_block(key)}" for heading, key in supplementary)
assembled_method_text = "\n\n".join(assembled_methods)
assembled_result_text = "\n\n".join(assembled_results)
assembled_discussion_text = "\n\n".join(assembled_discussion)

document = f"""# {title_text}

## Authors

[Authors, affiliations, corresponding-author details, and ORCID identifiers to be inserted]

## Abstract

{abstract}

## Keywords

Creep rupture; physics-guided machine learning; source-aware validation; cross-alloy generalization; material-state representation; transfer learning

## 1. Introduction

{introduction}

## 2. Database and Source-Aware Validation

{assembled_method_text}

## 3. Results

{assembled_result_text}

## 4. Discussion

{assembled_discussion_text}

## 5. Conclusions

{conclusions}

## References

[To be populated according to `REFERENCE_ARCHITECTURE.md` and `REFERENCE_PLACEMENT_MATRIX.csv`.]

## Data Availability

[To be finalized after the source-data access and licensing audit. The submission statement must distinguish distributable processed data from any source material subject to access restrictions.]

## Code Availability

[To be finalized after the reproducibility package and executable environment are frozen.]

## Supplementary Information

{supplementary_text}
"""

OUT.write_text(document.rstrip() + "\n", encoding="utf-8")

# Production checks: structure, assets, frozen wording, and headline values.
assembled = OUT.read_text(encoding="utf-8")
required_headings = [
    "## Authors",
    "## Abstract",
    "## Keywords",
    "## 1. Introduction",
    "## 2. Database and Source-Aware Validation",
    "## 3. Results",
    "### 3.1 Physics-Guided ML Benchmark",
    "### 3.2 Cross-Alloy Transfer Analysis",
    "### 3.3 Classical Empirical Physics Interpretation",
    "### 3.4 Material-State Representation Limitation",
    "## 4. Discussion",
    "## 5. Conclusions",
    "## References",
    "## Data Availability",
    "## Code Availability",
    "## Supplementary Information",
]
positions = [assembled.index(heading) for heading in required_headings]
assert positions == sorted(positions)
assert all(assembled.count(f"#### Table {i}.") == 1 for i in range(1, 6))
assert all(assembled.count(f"**Figure {i}.") == 1 for i in range(1, 7))
assert all(assembled.count(f"**Figure S{i}.") == 1 for i in range(1, 6))
assert assembled.index("**Figure 1.") > assembled.index("## 1. Introduction")
for stem in FIGURE_STEMS.values():
    assert all((BASE / f"{stem}.{suffix}").exists() for suffix in ("pdf", "svg", "png"))
for required_term in (
    "unresolved material-state-related variability",
    "state-associated effective physics parameters",
    "source-aware validation",
):
    assert required_term in assembled
for forbidden_term in ("hidden state", "latent state", "universal law", "discovered mechanism"):
    assert forbidden_term not in assembled.lower()
assert "does not establish intrinsic parameter variation" in assembled
for frozen_value in (
    "0.7497", "0.3592", "0.4931", "0.7254", "0.3830", "0.6893", "0.4202",
    "0.7008", "0.4165", "0.4802", "0.8428", "0.9301", "−447.2817",
):
    assert frozen_value in assembled
assert "[To be populated according to `REFERENCE_ARCHITECTURE.md`" in assembled
assert "# TASK" not in assembled and "Draft metadata" not in assembled

print(f"Assembled {OUT}")
print("PASS: manuscript structure, 5 tables, 11 figure blocks/assets, terminology, and headline-value presence")
