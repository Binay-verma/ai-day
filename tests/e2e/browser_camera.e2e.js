// Real-browser end-to-end test of "Use my camera" (not part of pytest): Chromium with a video file as its webcam
// (--use-file-for-fake-video-capture; build fakecam.y4m as in benchmarks/bench_browser_camera.py). Signs in as
// judge, clicks "Use my camera", samples the console every 2 s, takes screenshots, stops the camera.
// usage: node tests/e2e/browser_camera.e2e.js http://127.0.0.1:7861 <judge password> fakecam.y4m 72 <outdir>
const { chromium } = require("playwright");
const [url, pw, cam, secs, out] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch({ args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    `--use-file-for-fake-video-capture=${cam}`] });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await ctx.grantPermissions(["camera"], { origin: url });
  const page = await ctx.newPage();
  const logs = [];
  page.on("console", m => logs.push(`${m.type()}: ${m.text()}`));
  page.on("pageerror", e => logs.push(`pageerror: ${e.message}`));
  await page.goto(url);
  await page.fill("#lu", "judge"); await page.fill("#lp", pw);
  await page.click("#loginform button[type=submit]");
  await page.waitForSelector("#mycam:not([hidden])", { timeout: 15000 });
  await page.screenshot({ path: `${out}/0_before.png` });
  const t0 = Date.now();
  await page.click("#mycam");
  await page.waitForSelector("#mycambar.live", { timeout: 15000 });
  const samples = [];
  let shot = 1;
  while (Date.now() - t0 < secs * 1000) {
    await page.waitForTimeout(2000);
    const s = await page.evaluate(() => ({
      t: performance.now(), fps: document.querySelector("#fps").textContent, ppl: document.querySelector("#ppl").textContent,
      cam: document.querySelector("#camsel").value, bar: document.querySelector("#mycamtext").textContent,
      time: document.querySelector("#mycamtime").textContent,
      alerts: [...document.querySelectorAll(".alert .title")].map(x => x.textContent),
      fall: (document.querySelector(".meter .val") || {}).textContent }));
    s.elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    samples.push(s);
    console.log(JSON.stringify({ e: s.elapsed, fps: s.fps, ppl: s.ppl, fall: s.fall, n_alerts: s.alerts.length, time: s.time }));
    if ([8, 20, 30].includes(Math.round(s.elapsed / 2) * 2) && shot < 4) await page.screenshot({ path: `${out}/${shot++}_live.png` });
  }
  await page.screenshot({ path: `${out}/9_end.png`, fullPage: true });
  await page.click("#mycamstop");
  await page.waitForTimeout(1500);
  const after = await page.evaluate(() => ({ bar: document.querySelector("#mycamtext").textContent,
    pressed: document.querySelector("#mycam").getAttribute("aria-pressed"), cam: document.querySelector("#camsel").value,
    options: [...document.querySelectorAll("#camsel option")].map(o => o.value) }));
  console.log("AFTER", JSON.stringify(after));
  console.log("ALERTS", JSON.stringify(samples[samples.length - 1].alerts));
  console.log("LOGS", JSON.stringify(logs.filter(l => !l.startsWith("log:")).slice(0, 20)));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
