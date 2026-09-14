"""Start the local FPL dashboard without opening a console window."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from fplbot.webapp import serve  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        serve(
            PROJECT_DIR / "docs",
            PROJECT_DIR,
            port=8765,
            open_browser=True,
        )
    )
