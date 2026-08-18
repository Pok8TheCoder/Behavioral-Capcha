(async () => {
  const meta = { pid: 1, pt: "mouse", src: "canvas" };
  const seed = (Date.now() ^ (Math.random() * 0xffffffff)) >>> 0;
  let s = seed || 1;
  const rand = () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17; s >>>= 0;
    s ^= s << 5; s >>>= 0;
    return (s >>> 0) / 4294967296;
  };
  const nrand = () => {
    const u = Math.max(1e-9, 1 - rand());
    const v = 1 - rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const resp = await fetch("http://127.0.0.1:8010/api/challenge/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ challenge_type: "button_sequence" }),
  });
  const data = await resp.json();
  challenge = data;
  isCapturing = true;
  submitting = false;
  isBotRunning = false;
  sequenceProgress = 0;
  clientInteractionStartMs = performance.now();
  document.getElementById("challenge-title").textContent = data.title;
  document.getElementById("challenge-instructions").textContent = data.instructions;
  document.getElementById("meta-type").textContent = data.challenge_type;
  document.getElementById("meta-seed").textContent = String(data.seed);
  document.getElementById("meta-participant").textContent = "—";
  resetAssessmentUI("IN PROGRESS", data.instructions);
  document.getElementById("btn-submit").style.display = "inline-block";
  if (data.spec.buttons) data.spec.buttons.forEach((b) => { b.clicked = false; });
  drawChallenge();

  const buttons = [...data.spec.buttons].sort((a, b) => a.id - b.id);
  const W = 960, H = 640;
  let t = performance.now() + 180 + rand() * 420;
  let x = 28 + rand() * 90;
  let y = 22 + rand() * 70;
  const evs = [];

  function push(px, py, type) {
    const rx = Math.round(px);
    const ry = Math.round(py);
    if (type === "mousemove" && evs.length) {
      const prev = evs[evs.length - 1];
      if (prev.type === "mousemove" && prev.x === rx && prev.y === ry) return;
    }
    evs.push({
      x: rx,
      y: ry,
      t: Math.round(t * 100) / 100,
      type,
      pid: meta.pid,
      pt: meta.pt,
      src: meta.src,
    });
  }

  function jerk(tau) {
    const a = 9.4 + rand() * 1.4;
    const b = 14.2 + rand() * 1.8;
    const c = a * 0.5;
    return a * tau ** 3 - b * tau ** 4 + (b - a + c * 0) * 0 + (6 + rand() * 0.6) * tau ** 5;
  }
  function minJerk(tau) {
    const raw = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5;
    return clamp(raw + nrand() * 0.012, 0, 1);
  }

  function moveTo(tx, ty, dur) {
    const x0 = x, y0 = y;
    const dx = tx - x0, dy = ty - y0;
    const dist = Math.hypot(dx, dy) || 1;
    const nx = -dy / dist, ny = dx / dist;
    const curve = (0.06 + rand() * 0.22) * dist * (rand() < 0.5 ? -1 : 1);
    const cx = x0 + dx * (0.42 + rand() * 0.18) + nx * curve;
    const cy = y0 + dy * (0.38 + rand() * 0.22) + ny * curve;
    let ox = 0, oy = 0;
    const meanDt = 7.2 + rand() * 3.4;
    const steps = Math.max(10, Math.round(dur / meanDt));
    const pauseAt = rand() < 0.28 ? 0.35 + rand() * 0.3 : -1;
    for (let i = 1; i <= steps; i++) {
      let tau = i / steps;
      if (pauseAt > 0 && Math.abs(tau - pauseAt) < 0.04) {
        t += 18 + rand() * 42;
      }
      const s = minJerk(tau);
      const u = 1 - s;
      const bx = u * u * x0 + 2 * u * s * cx + s * s * tx;
      const by = u * u * y0 + 2 * u * s * cy + s * s * ty;
      ox = ox * 0.62 + nrand() * 0.55;
      oy = oy * 0.62 + nrand() * 0.55;
      x = clamp(bx + ox, 4, W - 4);
      y = clamp(by + oy, 4, H - 4);
      t += meanDt + nrand() * 1.8 + (rand() < 0.07 ? 9 + rand() * 10 : 0);
      if (rand() < 0.04) t += 12 + rand() * 20;
      push(x, y, "mousemove");
    }
    x = tx;
    y = ty;
  }

  function clickAt(tx, ty) {
    const landx = tx + nrand() * 7.5;
    const landy = ty + nrand() * 7.5;
    const dist = Math.hypot(landx - x, landy - y);
    const id = Math.log2(dist / 52 + 1);
    let dur = 85 + 148 * id + nrand() * 40;
    dur = clamp(dur, 170, 980);
    if (rand() < 0.42) {
      const over = 0.06 + rand() * 0.11;
      moveTo(landx + (landx - x) * over + nrand() * 7, landy + (landy - y) * over + nrand() * 7, dur * 0.76);
      t += 20 + rand() * 55;
      moveTo(landx, landy, 80 + rand() * 95);
    } else {
      moveTo(landx, landy, dur);
    }
    const dwellEnd = t + 48 + rand() * 130;
    while (t < dwellEnd) {
      t += 7.5 + rand() * 6;
      x += nrand() * 0.4;
      y += nrand() * 0.4;
      push(x, y, "mousemove");
    }
    t += 1 + rand() * 6;
    push(x, y, "mousedown");
    const holdEnd = t + 62 + rand() * 95;
    while (t < holdEnd) {
      t += 7.5 + rand() * 6;
      x += nrand() * 0.28;
      y += nrand() * 0.28;
      push(x, y, "mousemove");
    }
    push(x, y, "mouseup");
    t += 35 + rand() * 110;
  }

  const readEnd = t + 380 + rand() * 820;
  while (t < readEnd) {
    t += 8 + rand() * 9;
    x = clamp(x + nrand() * 0.9, 8, W - 8);
    y = clamp(y + nrand() * 0.7, 8, H - 8);
    push(x, y, "mousemove");
  }

  for (const b of buttons) clickAt(b.x, b.y);

  events = evs;
  trail = evs.filter((e) => e.type === "mousemove").slice(-800).map((e) => ({ x: e.x, y: e.y, isBot: false }));
  document.getElementById("meta-events").textContent = String(events.length);
  drawChallenge();

  const wait = Math.max(40, t - performance.now() + 50);
  await new Promise((r) => setTimeout(r, Math.min(wait, 12000)));
  submitAttempt("human");
  await new Promise((r) => setTimeout(r, 1500));
  const scoreText = document.getElementById("human-confidence")
    ? document.getElementById("human-confidence").textContent
    : document.body.innerText;
  return {
    seed,
    session: data.session_id,
    challenge_seed: data.seed,
    n_events: evs.length,
    wait,
    scoreText: (document.querySelector("#confidence-value, .confidence, #human-score") || {}).textContent || null,
    overlay: document.getElementById("result-overlay") && document.getElementById("result-overlay").innerText,
    bodySlice: document.body.innerText.slice(0, 1800),
  };
})()
