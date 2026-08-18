(async () => {
  function mulberry(seed) {
    let s = seed >>> 0 || 1;
    return () => {
      s += 0x6D2B79F5;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rand = mulberry((Date.now() ^ (Math.random() * 1e9)) >>> 0);
  const nrand = () => {
    const u = Math.max(1e-9, 1 - rand());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * (1 - rand()));
  };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const typePick = ["button_sequence", "pin_entry", "target_hunt"][Math.floor(rand() * 3)];
  const resp = await fetch("http://127.0.0.1:8010/api/challenge/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ challenge_type: typePick }),
  });
  const data = await resp.json();
  challenge = data;
  isCapturing = true;
  submitting = false;
  isBotRunning = false;
  sequenceProgress = 0;
  pinProgress = 0;
  targetHits = 0;
  clientInteractionStartMs = performance.now();
  document.getElementById("challenge-title").textContent = data.title;
  document.getElementById("challenge-instructions").textContent = data.instructions;
  document.getElementById("meta-type").textContent = data.challenge_type;
  document.getElementById("meta-seed").textContent = String(data.seed);
  document.getElementById("meta-participant").textContent = "—";
  resetAssessmentUI("IN PROGRESS", data.instructions);
  document.getElementById("btn-submit").style.display = "inline-block";
  if (data.spec.buttons) data.spec.buttons.forEach((b) => { b.clicked = false; });
  if (data.spec.keys) data.spec.keys.forEach((k) => { k.pressed = false; });
  if (data.spec.targets) data.spec.targets.forEach((t) => { t.clicked = false; });
  drawChallenge();

  const W = 960, H = 640;
  let t = performance.now();
  let x = 20 + rand() * 120;
  let y = 18 + rand() * 90;
  const evs = [];
  const stampMove = () => t; // mousemove-like clock
  let moveClockSkew = 0.4 + rand() * 1.8;

  function push(px, py, type, usePerf) {
    const rx = Math.round(px);
    const ry = Math.round(py);
    if (type === "mousemove" && evs.length) {
      const prev = evs[evs.length - 1];
      if (prev.type === "mousemove" && prev.x === rx && prev.y === ry) return false;
    }
    const tt = usePerf ? performance.now() : t;
    evs.push({
      x: rx, y: ry,
      t: Math.round(tt * 100) / 100,
      type, pid: 1, pt: "mouse", src: "canvas",
    });
    return true;
  }

  function advance(dt) {
    t += dt;
  }

  function burstMove(nx, ny) {
    const n = 1 + Math.floor(rand() * 4);
    for (let i = 0; i < n; i++) {
      const px = nx + (i ? nrand() * 0.35 : 0);
      const py = ny + (i ? nrand() * 0.35 : 0);
      t += 0.2 + rand() * 1.4;
      push(px, py, "mousemove", false);
    }
    x = nx; y = ny;
    t += 5.5 + rand() * 11 + (rand() < 0.12 ? 10 + rand() * 18 : 0);
  }

  function homeTo(tx, ty) {
    if (rand() < 0.22) {
      const wrong = Math.atan2(ty - y, tx - x) + (rand() < 0.5 ? 1 : -1) * (0.4 + rand() * 0.9);
      const d = 18 + rand() * 55;
      const hx = clamp(x + Math.cos(wrong) * d, 6, W - 6);
      const hy = clamp(y + Math.sin(wrong) * d, 6, H - 6);
      const hops = 3 + Math.floor(rand() * 5);
      for (let i = 1; i <= hops; i++) {
        burstMove(x + (hx - x) * (i / hops) + nrand() * 1.2, y + (hy - y) * (i / hops) + nrand() * 1.2);
      }
    }
    let guard = 0;
    while (guard++ < 400) {
      const dx = tx - x, dy = ty - y;
      const dist = Math.hypot(dx, dy);
      if (dist < 2.4) break;
      const heading = Math.atan2(dy, dx) + nrand() * (0.18 + 0.35 * Math.min(1, dist / 280));
      const ballistic = dist > 46;
      const step = ballistic
        ? clamp(dist * (0.12 + rand() * 0.16) + 3 + nrand() * 2.2, 2.5, 22)
        : clamp(1.1 + dist * (0.18 + rand() * 0.2) + nrand() * 0.8, 0.6, 7.5);
      let nx = clamp(x + Math.cos(heading) * step, 5, W - 5);
      let ny = clamp(y + Math.sin(heading) * step, 5, H - 5);
      if (rand() < 0.08 && ballistic) {
        nx += nrand() * 4;
        ny += nrand() * 4;
      }
      burstMove(nx, ny);
      if (rand() < 0.07 && dist > 30) t += 16 + rand() * 40;
    }
    burstMove(tx + nrand() * 1.1, ty + nrand() * 1.1);
  }

  function clickAt(tx, ty) {
    homeTo(tx + nrand() * 5.5, ty + nrand() * 5.5);
    const hoverEnd = t + 40 + rand() * 160;
    while (t < hoverEnd) {
      t += 8 + rand() * 10;
      push(x + nrand() * 0.7, y + nrand() * 0.7, "mousemove", false);
    }
    push(x, y, "mousedown", true);
    const holdEnd = t + 48 + rand() * 110;
    while (t < holdEnd) {
      t += 8 + rand() * 8;
      x += nrand() * 0.22;
      y += nrand() * 0.22;
      push(x, y, "mousemove", false);
    }
    push(x, y, "mouseup", false);
    t += 25 + rand() * 140;
  }

  const targets = [];
  if (data.challenge_type === "button_sequence") {
    [...data.spec.buttons].sort((a, b) => a.id - b.id).forEach((b) => targets.push([b.x, b.y]));
  } else if (data.challenge_type === "pin_entry") {
    const code = String(data.spec.code);
    const keys = data.spec.keys;
    for (const ch of code) {
      const key = keys.find((k) => String(k.label) === ch || String(k.value) === ch || String(k.id) === ch);
      if (key) targets.push([key.x, key.y]);
    }
  } else if (data.challenge_type === "target_hunt") {
    data.spec.targets.forEach((tg) => targets.push([tg.x, tg.y]));
  }

  const thinkEnd = t + 280 + rand() * 900;
  while (t < thinkEnd) {
    t += 9 + rand() * 12;
    x = clamp(x + nrand() * 1.1, 8, W - 8);
    y = clamp(y + nrand() * 0.9, 8, H - 8);
    push(x, y, "mousemove", false);
  }

  for (const [tx, ty] of targets) clickAt(tx, ty);

  events = evs;
  trail = evs.filter((e) => e.type === "mousemove").slice(-800).map((e) => ({ x: e.x, y: e.y, isBot: false }));
  document.getElementById("meta-events").textContent = String(events.length);
  drawChallenge();

  const wait = Math.max(30, Math.min(t - performance.now() + 40, 14000));
  await new Promise((r) => setTimeout(r, wait));
  const r = await fetch("http://127.0.0.1:8010/api/challenge/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: challenge.session_id,
      events,
      participant: "human",
      verify_token: challenge.verify_token,
      client_started_ms: clientInteractionStartMs,
      client_submitted_ms: performance.now(),
      source: "dashboard",
    }),
  });
  const out = await r.json();
  if (r.ok) showResults(out);
  return {
    ok: r.ok,
    type: data.challenge_type,
    n: evs.length,
    human: out.human_confidence,
    eval: out.evaluation,
    task: out.task_result,
    detail: out.detail || null,
  };
})()
