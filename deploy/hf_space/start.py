"""Start the hosted public demo (Hugging Face Space or any container host).

Accounts come from environment secrets, never from files in the repository:
  DEMO_PASSWORD   password of the shared "judge" account (operator: can confirm / dismiss, export, run the scenario)
  ADMIN_PASSWORD  optional "admin" account (audit log, legal hold)
If DEMO_PASSWORD is missing, a random one is generated and printed to the container log (visible to the
Space owner only). Writes configs/_cloud_runtime.yaml (extends cloud_demo.yaml) and starts the server.
"""
from __future__ import annotations

import os
import secrets
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from edgecopilot.security import hash_password  # noqa: E402


def main() -> None:
    os.chdir(ROOT)
    demo_pw = os.environ.get("DEMO_PASSWORD", "")
    if len(demo_pw) < 10:
        demo_pw = secrets.token_urlsafe(12)
        print(f"[start] DEMO_PASSWORD not set (or < 10 chars): generated one for user 'judge': {demo_pw}", flush=True)
    users = [{"username": "judge", "role": "operator", "password_hash": hash_password(demo_pw)}]
    admin_pw = os.environ.get("ADMIN_PASSWORD", "")
    if len(admin_pw) >= 10:
        users.append({"username": "admin", "role": "admin", "password_hash": hash_password(admin_pw)})
    lines = ["# generated at start-up by deploy/hf_space/start.py - do not commit", "extends: cloud_demo.yaml",
             "server:", "  auth:", "    users:"]
    for u in users:
        lines.append(f'      - {{username: {u["username"]}, role: {u["role"]}, password_hash: "{u["password_hash"]}"}}')
    path = os.path.join(ROOT, "configs", "_cloud_runtime.yaml")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    # behind the platform's HTTPS proxy: trust its X-Forwarded-* so lockout / rate limits see real clients
    os.environ.setdefault("EDGECOPILOT_TRUSTED_PROXIES", "*")
    port = os.environ.get("PORT", "7860")
    print(f"[start] users: {', '.join(u['username'] for u in users)}; listening on 0.0.0.0:{port}", flush=True)
    # container entry point: the platform's proxy is the only way in, and login is always on (public_demo config)
    os.execvp(sys.executable, [sys.executable, "-m", "edgecopilot.server.app", "--config", path,  # noqa: S606
                               "--host", "0.0.0.0", "--port", port])  # noqa: S104


if __name__ == "__main__":
    main()
