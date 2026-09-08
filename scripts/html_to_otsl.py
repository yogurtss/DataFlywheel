"""Independent batch conversion entry point using the unified importer."""
import sys
import _bootstrap  # noqa: F401
from dataflywheel.cli import main

if __name__ == "__main__":
    main(["convert", *sys.argv[1:]])
