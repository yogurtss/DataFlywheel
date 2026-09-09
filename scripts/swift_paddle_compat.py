"""Loaded by swift --external_plugins in each training process."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataflywheel.paddle_compat import install_patch

install_patch()
