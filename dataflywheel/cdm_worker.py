"""Isolated bridge into the official OmniDocBench CDM environment.

Invoked as a script by metrics.py; deliberately avoids importing this package.
"""
import json
from pathlib import Path
import shutil
import sys


def main():
    repo, inp, out = sys.argv[1:]
    sys.path.insert(0, repo)
    result = {}
    try:
        for binary in ("pdflatex", "gs"):
            if not shutil.which(binary):
                raise RuntimeError(f"missing CDM dependency: {binary}")
        from src.metrics.cdm.cdm import cdm_metrics
        data = json.loads(Path(inp).read_text())
        result = cdm_metrics(data["gt"], data["pred"], save_vis=False, tmp_dir=str(Path(inp).parent / "render"))
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    Path(out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
