#!/usr/bin/env bash
# Run the public cloud demo inside a GitHub Codespace and make it reachable by the judges.
#   bash scripts/codespace_demo.sh
# Needs: the models + demo reel (upload edge-copilot-models-and-demo.zip to the project folder once) and a judge
# password: either a Codespaces secret DEMO_PASSWORD (github.com/settings/codespaces) or typed when asked.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f models/yolo11n-pose_416.onnx ] || [ ! -f demo/demo_campus.mp4 ]; then
  zip=$(ls -1 edge-copilot-models-and-demo*.zip 2>/dev/null | head -1 || true)
  if [ -n "$zip" ]; then echo "Unzipping $zip ..."; unzip -oq "$zip"; fi
fi
for f in models/yolo11n-pose_416.onnx models/fire_smoke_416.onnx demo/demo_campus.mp4; do
  [ -f "$f" ] || { echo "Missing $f: drag edge-copilot-models-and-demo.zip into the file explorer (project root) and run again."; exit 1; }
done
if [ -z "${DEMO_PASSWORD:-}" ]; then read -r -s -p "Judge password (>= 10 characters): " DEMO_PASSWORD; echo; export DEMO_PASSWORD; fi
[ "${#DEMO_PASSWORD}" -ge 10 ] || { echo "The judge password must be at least 10 characters"; exit 1; }
export PORT=7860
python deploy/hf_space/start.py &
SERVER=$!
for _ in $(seq 1 90); do curl -fs "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break; sleep 1; done
if [ -n "${CODESPACE_NAME:-}" ]; then
  gh codespace ports visibility "$PORT:public" -c "$CODESPACE_NAME" >/dev/null 2>&1 || \
    echo "Could not make the port public automatically: Ports tab -> right-click 7860 -> Port Visibility -> Public"
  DOMAIN=${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}
  echo
  echo "=================================================================="
  echo " Give the judges:  https://${CODESPACE_NAME}-${PORT}.${DOMAIN}"
  echo " Login: judge / the password you set.  Keep this terminal running."
  echo "=================================================================="
fi
wait "$SERVER"
