"""Opt-in diagnostics for CSV validation failures (set REVIEWLENS_DEBUG_CSV=1).

Writes one line to stderr and to reports/csv_diagnostics.log just before a CSV validation error is raised, so a
failure can be traced to the exact function, file and parser settings that produced it.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "reports" / "csv_diagnostics.log"


def log_csv_failure(function: str, source, settings: dict, columns) -> None:
    if os.environ.get("REVIEWLENS_DEBUG_CSV") != "1":
        return
    if isinstance(source, (bytes, bytearray)):
        desc = f"<in-memory upload, {len(source):,} bytes; no disk path>"
    elif isinstance(source, str) and source.startswith("upload:"):
        desc = f"<in-memory {source}; no disk path>"
    else:
        desc = str(Path(str(source)).resolve()) if isinstance(source, (str, Path)) else repr(source)
    line = (f"{datetime.now().isoformat(timespec='seconds')} | function={function} | source={desc} | "
            f"settings={settings} | first5_columns={[str(c)[:40] for c in list(columns)[:5]]}")
    print("[CSV-DIAG]", line, file=sys.stderr, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
