"""Operational input/path preflight only; no scientific computations or datasets."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
CANDIDATE_DIR = CODE_DIR.parent

def _inside(path, parent):
    return path == parent or parent in path.parents

def _original_option(argv, flag):
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
        if value.startswith(flag + '='):
            return Path(value.split('=', 1)[1]).expanduser().resolve()
    return None

def configure(entry_file, run_as_main=False):
    """Bind original relative paths to a private workspace; preflight before imports."""
    name = Path(entry_file).name
    inherited = os.environ.get('PAPER1_WORK_ROOT')
    if not run_as_main:
        # Library import alone does not read private inputs or invoke a main routine.
        return Path(inherited).expanduser().resolve() if inherited else Path('__PAPER1_INPUTS_NOT_SUPPLIED__').resolve()
    spec = json.loads((CODE_DIR / 'CODE_INPUTS.json').read_text(encoding='utf-8'))[name]
    parser = argparse.ArgumentParser(
        description=spec['role'] + '. Local pre-release candidate; lawful external inputs required.',
        epilog='Original options: ' + spec['original_options'] + '. No exact public-only reproduction is promised.',
    )
    parser.add_argument('--work-root', type=Path, help='Private disposable workspace with original relative input layout; or PAPER1_WORK_ROOT.')
    parser.add_argument('--check-inputs', action='store_true', help='Check file presence and protected/output paths, then exit; no imports, parsing input rows or scientific execution.')
    args, remaining = parser.parse_known_args()
    root_arg = args.work_root or inherited
    if not root_arg:
        parser.error('No lawful input workspace supplied. Use --work-root or PAPER1_WORK_ROOT; restricted inputs are not bundled.')
    root = Path(root_arg).expanduser().resolve()
    if _inside(root, CANDIDATE_DIR) or any(part in {'Paper1_Final_Release_v1.0', 'submission_package_CMS'} for part in root.parts):
        parser.error('Use a separate private disposable workspace outside the candidate and frozen/submission archives.')
    if not root.is_dir():
        parser.error('The private input workspace does not exist; no files or directories were created.')
    inputs = [root / p for p in spec['inputs']]
    outputs = [root / p for p in spec['outputs']]
    # Respect existing preprocessing CLI options without changing its parser or computations.
    if name == 'build_ml_dataset_v1.py':
        db = _original_option(remaining, '--db-dir')
        if db:
            inputs = [db / Path(p).name for p in spec['inputs']]
        out = _original_option(remaining, '--output-dir')
        if out:
            outputs = [out / Path(p).name for p in spec['outputs']]
    elif name == 'create_splits_v1.py':
        supplied = _original_option(remaining, '--input-path')
        if supplied:
            inputs = [supplied]
        out = _original_option(remaining, '--output-dir')
        if out:
            outputs = [out / Path(p).name for p in spec['outputs']]
    report = _original_option(remaining, '--report-dir')
    protected = [p.resolve() for p in inputs + outputs + ([report] if report else [])]
    if any(_inside(p, CANDIDATE_DIR) or any(part in {'Paper1_Final_Release_v1.0', 'submission_package_CMS'} for part in p.parts) for p in protected):
        parser.error('Input/output overrides must remain outside the candidate and frozen/submission archives.')
    if any(not p.is_file() for p in inputs):
        missing = [str(p) for p in inputs if not p.is_file()]
        parser.error('Required lawful external inputs are absent: ' + ', '.join(missing) + '. Code and metadata do not supply restricted experimental rows.')
    if any(p.exists() for p in outputs):
        parser.error('An output destination already exists. Use a clean disposable workspace; existing results will not be overwritten.')
    if args.check_inputs:
        print('INPUT PRESENCE CHECK PASS; no row content read, scientific imports or model/figure execution performed.')
        raise SystemExit(0)
    os.environ['PAPER1_WORK_ROOT'] = str(root)
    sys.argv = [sys.argv[0]] + remaining
    return root
