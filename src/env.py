"""Tiny .env loader (no extra dependency).

Reads KEY=VALUE lines from <repo>/.env into os.environ. A variable that is ALREADY set in the
terminal / Windows wins over .env; when that happens we print a note (names only, never secrets),
because it is the most common reason "I fixed .env but nothing changed".

The .env file holds personal tokens: it is git-ignored and must NEVER be committed.
"""
import os
import sys
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_env(path: Path = ENV_PATH) -> bool:
    if not path.exists():
        return False
    overridden = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if key in os.environ and os.environ[key] != value:
            overridden.append(key)
        else:
            os.environ.setdefault(key, value)
    if overridden:
        print(f"[env] NOTE: {', '.join(overridden)} is already set in your terminal/Windows and "
              f"OVERRIDES your .env file.\n"
              f"      Clear it:  Remove-Item Env:{overridden[0]}   (then open a NEW terminal)\n",
              file=sys.stderr)
    return True
