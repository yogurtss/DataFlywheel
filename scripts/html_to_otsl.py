"""Independent batch conversion entry point using the unified importer."""
import sys
from dataflywheel.cli import main

if __name__ == "__main__":
    main(["convert", *sys.argv[1:]])
