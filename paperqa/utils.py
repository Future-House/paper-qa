import os
import sys


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self):
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_):
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout
