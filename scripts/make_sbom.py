"""Generate a CycloneDX 1.5 SBOM + ML-BOM (Python packages and ONNX models with hashes).

    python scripts/make_sbom.py > sbom.cdx.json
    python scripts/make_sbom.py --pin-config configs/campus.yaml   # also print the sha256 pins for the config
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys
import uuid
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_INFO = {
    "yolo11n-pose": {"licence": "AGPL-3.0", "supplier": "Ultralytics", "pedigree": "COCO-pose pretrained"},
    "fire_smoke": {"licence": "AGPL-3.0 (weights) + CC-BY-NC-4.0 (Home-fire training data)", "supplier": "team local maxima",
                   "pedigree": "YOLO11n D-Fire (sayedgamal99, MIT) fine-tuned on Home-fire + DBA-Fire + DeepQuest negatives"},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_requirements():
    from edgecopilot import __version__  # noqa: F401
    reqs = md.requires("edgecopilot") or []
    names = set()
    for r in reqs:
        if "extra ==" in r and "mqtt" not in r:
            continue
        names.add(r.split(";")[0].split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower())
    # transitive closure
    todo, seen = list(names), set()
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        try:
            for r in md.requires(n) or []:
                if "extra ==" in r:
                    continue
                todo.append(r.split(";")[0].split("[")[0].split(">")[0].split("=")[0].split("<")[0].split("!")[0].split("~")[0].strip().lower())
        except md.PackageNotFoundError:
            pass
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin-config", default=None)
    args = ap.parse_args()
    from edgecopilot import __version__
    comps = []
    for n in runtime_requirements():
        try:
            d = md.distribution(n)
        except md.PackageNotFoundError:
            continue
        lic = d.metadata.get("License-Expression") or d.metadata.get("License") or ""
        comps.append({"type": "library", "name": d.metadata["Name"], "version": d.version,
                      "purl": f"pkg:pypi/{d.metadata['Name'].lower()}@{d.version}",
                      **({"licenses": [{"license": {"name": lic[:80]}}]} if lic else {})})
    pins = {}
    for fn in sorted(os.listdir(os.path.join(ROOT, "models"))):
        if not fn.endswith(".onnx"):
            continue
        p = os.path.join(ROOT, "models", fn)
        h = sha256(p)
        pins[f"models/{fn}"] = h
        key = next((k for k in MODEL_INFO if fn.startswith(k)), None)
        info = MODEL_INFO.get(key, {})
        comps.append({"type": "machine-learning-model", "name": fn, "version": __version__,
                      "hashes": [{"alg": "SHA-256", "content": h}],
                      "supplier": {"name": info.get("supplier", "unknown")},
                      "licenses": [{"license": {"name": info.get("licence", "unknown")}}],
                      "pedigree": {"notes": info.get("pedigree", "")}})
    bom = {"bomFormat": "CycloneDX", "specVersion": "1.5", "serialNumber": f"urn:uuid:{uuid.uuid4()}", "version": 1,
           "metadata": {"timestamp": datetime.now(timezone.utc).isoformat(),
                        "component": {"type": "application", "name": "edgecopilot", "version": __version__}},
           "components": comps}
    if args.pin_config:
        print("# add under models.<name> in", args.pin_config, file=sys.stderr)
        for k, v in pins.items():
            print(f"#   {k}: sha256: {v}", file=sys.stderr)
    json.dump(bom, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
