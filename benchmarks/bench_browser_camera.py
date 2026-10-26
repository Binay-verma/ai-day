"""Server-limited frame rate of one "Use my camera" session.

A thin client replays a video through /ws/camera exactly like the console does (JPEG, capture time, one frame
in flight, at most 15 FPS) and reports the frames per second the server sustains. Alerts it raises are in the
console / database under the browser-... camera id.

    DEMO_PASSWORD=... PORT=7861 python deploy/hf_space/start.py &          # the cloud demo configuration
    python benchmarks/bench_browser_camera.py 127.0.0.1:7861 <judge password> <video.mp4> 70

Test clip used in RESULTS section 17 (GMDCSA24: lie down on a bed, then walk and fall; each held still):
    ffmpeg -i "Subject 2/ADL/03.mp4" -i "Subject 1/Fall/05.mp4" -filter_complex "[0:v]scale=480:360,fps=15,
    setsar=1,tpad=stop_mode=clone:stop_duration=6[a];[1:v]scale=480:360,fps=15,setsar=1,tpad=stop_mode=clone:
    stop_duration=10[b];[a][b]concat=n=2:v=1[v]" -map "[v]" -pix_fmt yuv420p fakecam.y4m
"""
import asyncio
import json
import struct
import sys
import time
import urllib.request

import cv2
import websockets

URL, PW, VIDEO, SECS = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])


def login():
    req = urllib.request.Request(f"http://{URL}/api/login", data=json.dumps({"username": "judge", "password": PW}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["token"]


async def main():
    cap = cv2.VideoCapture(VIDEO)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes())
    tok = login()
    async with websockets.connect(f"ws://{URL}/ws/camera?token={tok}", max_size=2**24) as ws:
        print(json.loads(await ws.recv()))
        t0 = time.perf_counter()
        n, skips, ms = 0, 0, []
        while time.perf_counter() - t0 < SECS:
            el = time.perf_counter() - t0
            idx = int(el * 15) % len(frames)          # the camera's own 15 FPS clock: send the current frame
            s = time.perf_counter()
            await ws.send(struct.pack("<d", el) + frames[idx])
            m = json.loads(await ws.recv())
            if m["type"] == "frame":
                n += 1
                ms.append((time.perf_counter() - s) * 1e3)
            elif m["type"] == "skip":
                skips += 1
            await asyncio.sleep(max(0.0, 1 / 15 - (time.perf_counter() - s)))
        dt = time.perf_counter() - t0
        ms.sort()
        print(json.dumps({"fps": round(n / dt, 2), "frames": n, "skips": skips, "rtt_median_ms": round(ms[len(ms) // 2], 1),
                          "rtt_p90_ms": round(ms[int(len(ms) * 0.9)], 1)}))
        await ws.send('{"type": "stop"}')


asyncio.run(main())
