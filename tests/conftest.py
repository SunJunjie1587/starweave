from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY_SRC = ROOT / "plugins" / "first_party" / "src"
MANIFEST = ROOT / "plugins" / "first_party" / "entry_points.json"

# Local tests run without installing the first-party plugin distribution.
# Installed distributions still resolve through entry points first.
if str(FIRST_PARTY_SRC) not in sys.path:
    sys.path.insert(0, str(FIRST_PARTY_SRC))
os.environ.setdefault("STARWEAVE_PLUGIN_MANIFEST", str(MANIFEST))
os.environ.setdefault("STARWEAVE_PLUGIN_MANIFEST_FIRST", "1")
