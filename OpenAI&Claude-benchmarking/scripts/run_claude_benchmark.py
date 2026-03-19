from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = SCRIPT_DIR / 'run_benchmark.py'

if __name__ == '__main__':
    args = [sys.executable, str(TARGET), '--provider', 'claude', *sys.argv[1:]]
    raise SystemExit(subprocess.call(args))
