"""Read-only packaging checks; no scientific execution or external access."""
import ast
import runpy
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for path in (root / 'src').rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
runpy.run_path(str(root / 'src/evaluation/verify_public_summaries.py'), run_name='__main__')
