"""Verify only candidate checksums and frozen aggregate headline alignment."""
import argparse
import csv
import hashlib
import math
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.candidate_root.resolve()
    listed = {}
    for line in (root / 'manifests/CHECKSUMS_SHA256.txt').read_text().splitlines():
        expected, rel = line.split('  ', 1)
        p = (root / rel).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise SystemExit('Invalid candidate checksum path')
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise SystemExit('Candidate checksum mismatch: ' + rel)
        listed[str(p.relative_to(root))] = expected
    files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if set(listed) != files - {'manifests/CHECKSUMS_SHA256.txt'}:
        raise SystemExit('Candidate checksum inventory does not cover the exact file set')
    with (root / 'results_summary/METRIC_REFERENCE_TABLE.csv').open() as f:
        refs = list(csv.DictReader(f))
    with (root / 'results_summary/FINAL_MODEL_COMPARISON_TABLE.csv').open() as f:
        matches = [r for r in csv.DictReader(f) if r['model_name'] == 'P3-replace XGBoost']
    if len(matches) != 1 or len(refs) != 3 or {r['metric'] for r in refs} != {'R2', 'MAE', 'RMSE'}:
        raise SystemExit('Aggregate headline reference is incomplete')
    for ref in refs:
        if ref['n_oof'] != '220' or not math.isclose(float(ref['frozen_full_precision']), float(matches[0][ref['metric']]), rel_tol=0, abs_tol=5e-10):
            raise SystemExit('Frozen aggregate headline alignment failed')
    print(f'PASS: {len(listed)} candidate checksums and 3 frozen aggregate headline alignments. No OOF values recomputed; no restricted inputs read.')

if __name__ == '__main__':
    main()
