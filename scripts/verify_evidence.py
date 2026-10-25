"""Check a signed evidence bundle exported from the console (Alert -> Evidence bundle).

    python scripts/verify_evidence.py evidence-<id>.zip

Uses the site secret (EDGECOPILOT_SECRET or data/.secret), like the server. Exit code 0 = intact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from edgecopilot.evidence import verify_bundle  # noqa: E402
from edgecopilot.security import load_secret  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--secret-file", default=os.path.join(ROOT, "data", ".secret"))
    a = ap.parse_args()
    with open(a.bundle, "rb") as f:
        res = verify_bundle(f.read(), load_secret(path=a.secret_file))
    print(json.dumps(res, indent=1))
    print("INTACT" if res["ok"] else "NOT INTACT")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
