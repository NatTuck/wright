"""Serve one registry model on this host through Resin's HTTP APIs, by preset name.

For Gaudi, use the `wright` command, which finds the preset from the checkpoint. This script stays
for presets that `wright` does not cover.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Direct script execution must also make the repository packages importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.serving import argument_parser, main  # noqa: F401,E402

if __name__ == "__main__":
    main()
