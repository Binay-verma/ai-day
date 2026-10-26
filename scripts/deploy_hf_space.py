"""Deploy the public demo to a Hugging Face Space in one command (run on your own computer).

    pip install huggingface_hub
    hf auth login                      # or: huggingface-cli login   (token with "write" access)
    python scripts/deploy_hf_space.py --space <your-hf-username>/edge-copilot

It asks for the judge password (>= 10 characters), creates the Space (Docker, public, free CPU), stores the
password and a server secret as Space *secrets* (never in the files), uploads the code, the two 416 px models
and the demo reel, and prints the address to give the judges. Re-run it to update the Space.
Building takes ~5 minutes the first time; watch the "Logs" tab of the Space.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import secrets
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDED = ["models/yolo11n-pose_416.onnx", "models/fire_smoke_416.onnx", "demo/demo_campus.mp4", "demo/demo_campus_sensors.csv"]


def space_url(repo_id: str) -> str:
    owner, name = repo_id.split("/", 1)
    sub = re.sub(r"[^a-z0-9]+", "-", f"{owner}-{name}".lower()).strip("-")
    return f"https://{sub}.hf.space"


def stage(dst: str) -> None:
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "_cloud_runtime.yaml")
    shutil.copytree(os.path.join(ROOT, "edgecopilot"), os.path.join(dst, "edgecopilot"), ignore=ign)
    shutil.copytree(os.path.join(ROOT, "configs"), os.path.join(dst, "configs"), ignore=ign)
    for rel in NEEDED + ["pyproject.toml", "deploy/hf_space/start.py"]:
        os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), os.path.join(dst, rel))
    shutil.copy2(os.path.join(ROOT, "deploy", "hf_space", "Dockerfile"), os.path.join(dst, "Dockerfile"))
    shutil.copy2(os.path.join(ROOT, "deploy", "hf_space", "SPACE_README.md"), os.path.join(dst, "README.md"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", required=True, help="<hf-username>/<space-name>, e.g. binay/edge-copilot")
    ap.add_argument("--password", default=None, help="judge password (asked for if omitted)")
    ap.add_argument("--admin-password", default=None, help="optional admin account password")
    ap.add_argument("--private", action="store_true", help="private Space (judges would need a Hugging Face login)")
    ap.add_argument("--dry-run", action="store_true", help="only build the upload folder and print its path")
    a = ap.parse_args()
    if "/" not in a.space:
        sys.exit("--space must look like <hf-username>/<space-name>")
    missing = [p for p in NEEDED if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        sys.exit("Missing files (unzip edge-copilot-models-and-demo.zip into the project folder first): " + ", ".join(missing))
    dst = tempfile.mkdtemp(prefix="hf_space_")
    stage(dst)
    if a.dry_run:
        print(f"staged in {dst}")
        return 0
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub   (then: hf auth login)")
    pw = a.password or getpass.getpass("Judge password (>= 10 characters): ")
    if len(pw) < 10:
        sys.exit("the judge password must be at least 10 characters")
    api = HfApi()
    who = api.whoami()["name"]
    print(f"Signed in to Hugging Face as {who}")
    first = not api.repo_exists(a.space, repo_type="space")
    api.create_repo(a.space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    api.add_space_secret(a.space, "DEMO_PASSWORD", pw)
    if a.admin_password:
        api.add_space_secret(a.space, "ADMIN_PASSWORD", a.admin_password)
    # a stable server secret, set once: sessions and the audit chain survive restarts and re-deploys
    if first:
        api.add_space_secret(a.space, "EDGECOPILOT_SECRET", secrets.token_urlsafe(48))
    print("Uploading code, models and the demo reel (about 40 MB) ...")
    api.upload_folder(folder_path=dst, repo_id=a.space, repo_type="space",
                      commit_message="Deploy Explainable Edge AI Copilot public demo")
    shutil.rmtree(dst, ignore_errors=True)
    print("\nDone. The Space is building (about 5 minutes the first time):")
    print(f"  Space page:      https://huggingface.co/spaces/{a.space}")
    print(f"  Give judges:     {space_url(a.space)}   (user: judge)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
