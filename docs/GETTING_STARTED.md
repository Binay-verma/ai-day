# Getting started: install, run, test, demo

A step-by-step guide for teammates and evaluators. It works on **Windows, macOS and Linux**
and needs no GPU. Total time: about 15 minutes.

---

## 1. What you need

| | |
|---|---|
| Python | 3.10, 3.11, 3.12 or 3.13 (`python --version`) |
| Git | any recent version |
| Disk / RAM | ~1.5 GB free, 4 GB RAM or more |
| Files not in git | `edge-copilot-models-and-demo.zip` (models + demo video, ~90 MB): from the repository's **Releases** page or from a teammate |

> Why a separate zip? The ONNX models and the demo video are too large for git, and the YOLO
> weights (AGPL-3.0) and the fire training data (CC BY-NC) must not be redistributed publicly.
> Keep the repository private.

---

## 2. Get the code

```bash
git clone https://github.com/Binay-verma/ai-day.git
cd ai-day
```

(Or download the ZIP from GitHub and unzip it: the folder is then called `ai-day-main`.)

---

## 3. Create a virtual environment and install

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If PowerShell refuses to run `Activate.ps1`, run this once, then activate again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

You should see `Successfully installed ... edgecopilot-1.1.0 ...` at the end. Every time you
open a new terminal, activate the environment again (the prompt then starts with `(.venv)`).

---

## 4. Add the models and the demo video

Put `edge-copilot-models-and-demo.zip` in the folder **above** the project folder, then:

**Windows:** `Expand-Archive ..\edge-copilot-models-and-demo.zip -DestinationPath . -Force`

**macOS / Linux:** `unzip -o ../edge-copilot-models-and-demo.zip`

Check that these exist:

```
models/yolo11n-pose.onnx       models/fire_smoke.onnx       models/yolo11s-pose.onnx
models/yolo11n-pose_416.onnx   models/fire_smoke_416.onnx   (+ *.int8.onnx)
demo/demo_campus.mp4           demo/demo_campus_sensors.csv
```

(Building the models yourself instead needs PyTorch: `pip install -e ".[train]"` and then
`python scripts/setup_models.py`.)

---

## 5. Run the demo

```bash
python -m edgecopilot.server.app --source demo/demo_campus.mp4 --demo
```

Open **http://127.0.0.1:8000** in a browser. `--demo` skips the login and only listens on your
own computer. The 69 s demo video loops: walkway (normal), lying down on a bed (a hard
negative), a real fall, a small fire with an MQ-2 sensor trace, and a crowd dispersal.

### What you are looking at

| Part of the console | What it shows |
|---|---|
| Left: live view | People with skeletons, faces pixelated (privacy filter) |
| Risk right now | Current probability per hazard against its alert line |
| Right: Alerts | One card per incident: what happened, **why** (the evidence bars), what would have changed the decision, the recommended actions, an anonymised snapshot and heat map |
| "Explanation checked…" | The grounding verifier: every number and claim in the text was checked against the evidence |
| **What if?** | Drag a reading (e.g. torso angle) and see the exact new score and decision |
| Confirm / Dismiss | Records the operator's decision (audited); a dismissal tunes that camera, within safe limits |
| Ask | Questions such as "What would change it?", answered from the evidence |
| Header | **Sound off/on** (alarm tones), **Self-test** status, **Handover** report (operators) |

Stop the server with **Ctrl + C**.

---

## 6. Use your own webcam

```bash
python -m edgecopilot.server.app --config configs/examples/webcam.yaml --demo
```

- **macOS**: allow camera access for Terminal / VS Code when asked (or System Settings →
  Privacy & Security → Camera), then restart the terminal.
- Wrong camera (e.g. an iPhone via Continuity Camera)? Edit `source: "0"` to `"1"` in
  `configs/examples/webcam.yaml`.
- Quick version without the config: `--source 0`.

Place the laptop 3–4 m away so it sees your **whole body and the floor**. Then try:

| Test | Do this | Expect |
|---|---|---|
| Privacy | Sit in front of the camera | Your face is pixelated |
| Fall | Walk in, drop quickly onto a mat / cushion, stay down | Fall alert within a few seconds, with its reasons |
| Long lie | Stay down ~20 s (the webcam config shortens 60 s to 20 s) | Alert escalates to CRITICAL "STILL DOWN" |
| Not a fall | Lie down slowly and deliberately (sit first, then lie back) | No fall alert. The fall gauge says "held below the line: no fast, uncontrolled descent was seen (lying down, not falling)". If you stay down 60 s (20 s in the webcam config) it alerts anyway as a possible long lie |
| Not a fall | Crouch or bend down fast, then stand up | No fall alert ("the person did not end up lying down") |
| Intrusion | Stand in the right third of the image ~25 s | "Restricted-area intrusion" alert with a patrol action |
| Crowd | 3+ people walk in, then all run out of view at once (or uncomment the `zones` test lines in the webcam config) | Crowd measures abnormal *movement* (sudden dispersal, panic) and over-capacity, not a head count. 2 people standing in view correctly stays near 0 % (Corridor B holds 30) |
| Fire | Hold a phone playing a fire video close to the camera | May raise smoke / fire; without the MQ-2 sensor it may stay VERIFY. **Never use a real flame indoors** |
| Camera trouble | Cover the lens ~5 s | "Camera blackout" trouble alert |
| What if? | Open it on any alert, drag a slider | Score and decision update |

How it tells a fall from lying down: a fall alert needs *both* an uncontrolled descent (fast
drop of the hips, measured in body-heights) *and* a lying posture at the end. Slowly lying down
only gives the second. Push-ups and throwing yourself onto a bed are known look-alikes.

Use `threads: 4` (up to 6 on a laptop with 8+ cores; the webcam config ships with 4). With 2
threads the first webcam test ran at ~7 FPS, and a fast fall then spans only a few frames.

A laptop camera at table height is harder for fall detection than a ceiling camera (the body is
often cut off or seen end-on). For better fall detection add
`pose: {path: models/yolo11s-pose.onnx}` under the camera in the webcam config (uses ~3x CPU).

### Where your test is saved, and how to share it for review

Every alert (explanation, evidence, decision, your confirm / dismiss) is stored in
`data/edgecopilot.db`; privacy-filtered snapshots and heat maps are stored encrypted in
`data/media/`. To export a session as one file:

```bash
python scripts/export_session.py --hours 6                 # alerts, evidence, audit, machine info
python scripts/export_session.py --hours 6 --with-images   # + the pixelated snapshots / heat maps
```

It writes `outputs/session_export.zip`. Nothing leaves the computer unless you send it. Tip:
note down what you did and when (e.g. "15:02 fell on the mat, 15:05 lay down slowly"), so each
alert, or each missing alert, can be matched to what really happened.

---

## 7. Other ways to run

```bash
# headless: print explained alerts in the terminal
python -m edgecopilot.cli --source demo/demo_campus.mp4

# save an annotated, privacy-filtered video
python -m edgecopilot.cli --source demo/demo_campus.mp4 --save outputs/annotated.mp4

# a different port if 8000 is busy
python -m edgecopilot.server.app --source demo/demo_campus.mp4 --demo --port 8001

# faster on weak laptops (416 px models, ~12 FPS on 2 cores)
python -m edgecopilot.server.app --config configs/production.example.yaml --source demo/demo_campus.mp4 --demo --host 127.0.0.1
```

---

## 8. Check that everything works (tests)

```bash
pytest -q
```

Expected: all tests pass (a few are skipped if the model files are missing). The same checks
run on GitHub for every push (lint, type check, security scan, dependency audit, tests).

The benchmarks behind every number in `docs/RESULTS.md` are in `benchmarks/`; most need the
public datasets, which are not distributed with the project (see `docs/RESULTS.md`,
"Reproduce").

---

## 9. A 5-minute demo script (for presentations)

1. **Open the console** on the demo video. Point at the pixelated faces: privacy by design, no
   raw video leaves the box.
2. **Fall alert (~27 s into the video):** read the explanation aloud, show the evidence bars and
   the "would not have fired if…" line. "The explanation *is* the decision: these bars add up
   exactly to the score."
3. **What if?** Drag torso angle down: the alert would not have fired. Then show "Explanation
   checked against the evidence: 100 %": the verifier rejects any made-up number or cause.
4. **The false alert on the bed (~19 s):** routed as VERIFY, low margin. Honest about weak spots;
   a human decides.
5. **Fire (~47 s):** several signals agree (detector, flicker, MQ-2 sensor), so ALARM, with the
   zone's evacuation route and extinguisher in the actions.
6. **Operations:** Sound on (alarm tones), Self-test, Handover report, and one camera fault
   (cover the webcam) to show that a broken camera is itself an alert.
7. **Close with the numbers** from `docs/RESULTS.md` and the verdict in `docs/AUDIT.md`,
   including what is *not* solved yet. Decision support only: it never replaces the certified
   fire alarm.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `Video file not found: demo/demo_campus.mp4` | Step 4: unzip the models + demo zip into the project folder |
| `model pose not found ... running on weaker cues` | Same: the `models/` files are missing |
| `ModuleNotFoundError: edgecopilot` | Activate the environment (`.venv`) and run `pip install -e ".[dev]"` from the project folder |
| Webcam: black screen / "camera offline" | macOS camera permission for the terminal; close other apps using the camera (Zoom, Teams); try `source: "1"` |
| `address already in use` | Another server is running: stop it, or use `--port 8001` |
| `No users configured` | You ran without `--demo`. For a real deployment add users (`python -m edgecopilot.security hash`), see `docs/OPERATIONS.md` |
| Very low FPS | Close other heavy apps, or use the 416 px models (step 7, last command) |
| PowerShell will not activate the venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

For a real deployment (cameras over RTSP, logins, TLS, notifications, monitoring) follow
`docs/OPERATIONS.md` and `docs/SECURITY.md`.
