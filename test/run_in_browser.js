// Injected browser runner for button-sequence bot.
(async () => {
  const API_BASE = window.location.origin;
  const LOGICAL_WIDTH = 960;
  const LOGICAL_HEIGHT = 640;

  function ease(t) { return t * t * (3 - 2 * t); }
  function bezier(p0, p1, p2, t) {
    const u = 1 - t;
    return [
      u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
      u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
    ];
  }
  function rand(a, b) { return a + Math.random() * (b - a); }

  const cfg = {
    max_speed_px_ms: 0.7316875895872461,
    dwell_ms: [114.62991296279344, 255.6147550655669],
    pause_ms: [279.6033811448404, 531.9694939812301],
    move_interval_ms: [18.471673210841097, 30.745156930123535],
    jitter_px: 4.468538585944212,
    overshoot_px: 22.266995999175176,
    pre_click_delay_ms: [57.820268261401765, 197.16327163697687],
    wander_strength: 4.602561447658741,
  };

  function addMove(events, x, y, t) {
    events.push({ x: Math.round(x + rand(-0.9, 0.9)), y: Math.round(y + rand(-0.9, 0.9)), t: +t.toFixed(2), type: "mousemove" });
  }
  function wander(events, pos, t0, ms, strength) {
    const steps = Math.max(2, Math.floor(ms / rand(18, 30)));
    for (let i = 1; i <= steps; i++) {
      addMove(events, pos[0] + rand(-strength, strength), pos[1] + rand(-strength, strength), t0 + ms * (i / steps));
    }
    return t0 + ms;
  }
  function moveTo(events, start, end, t0) {
    const dx = end[0] - start[0], dy = end[1] - start[1];
    const dist = Math.hypot(dx, dy);
    if (dist < 1) return [t0, end];
    const nx = -dy / dist, ny = dx / dist;
    const offset = rand(-cfg.overshoot_px, cfg.overshoot_px);
    const mid = [(start[0] + end[0]) / 2 + nx * offset, (start[1] + end[1]) / 2 + ny * offset];
    const duration = Math.max(dist / cfg.max_speed_px_ms, rand(350, 1100) * (dist / 450));
    const steps = Math.max(6, Math.floor(duration / rand(cfg.move_interval_ms[0], cfg.move_interval_ms[1])));
    for (let i = 1; i <= steps; i++) {
      const raw = i / steps;
      const [x, y] = bezier(start, mid, end, ease(raw));
      addMove(events, x, y, t0 + duration * raw);
    }
    return [t0 + duration, end];
  }
  function clickAt(events, x, y, t0) {
    const ox = x + rand(-cfg.jitter_px, cfg.jitter_px);
    const oy = y + rand(-cfg.jitter_px, cfg.jitter_px);
    events.push({ x: Math.round(ox), y: Math.round(oy), t: +t0.toFixed(2), type: "mousedown" });
    const dwell = rand(cfg.dwell_ms[0], cfg.dwell_ms[1]);
    const micro = 2 + Math.floor(Math.random() * 5);
    for (let i = 1; i <= micro; i++) addMove(events, ox + rand(-1.5, 1.5), oy + rand(-1.5, 1.5), t0 + dwell * (i / micro));
    const tUp = t0 + dwell;
    events.push({ x: Math.round(ox), y: Math.round(oy), t: +tUp.toFixed(2), type: "mouseup" });
    return tUp;
  }

  function generateEvents(challenge) {
    const spec = challenge.spec;
    const buttons = Object.fromEntries(spec.buttons.map(b => [b.id, b]));
    const area = challenge.area || { x: 80, y: 120, width: 840, height: 480 };
    const events = [];
    let t = rand(150, 350);
    let pos = [area.x + rand(60, 180), area.y + rand(60, 180)];
    addMove(events, pos[0], pos[1], t);
    t = wander(events, pos, t, rand(120, 280), cfg.wander_strength);
    for (const btnId of spec.sequence) {
      const target = [buttons[btnId].x, buttons[btnId].y];
      [t, pos] = moveTo(events, pos, target, t);
      t += rand(cfg.pre_click_delay_ms[0], cfg.pre_click_delay_ms[1]);
      t = wander(events, pos, t, rand(40, 120), cfg.wander_strength * 0.4);
      t = clickAt(events, target[0], target[1], t);
      pos = [target[0] + rand(-2, 2), target[1] + rand(-2, 2)];
      t += rand(cfg.pause_ms[0], cfg.pause_ms[1]);
      t = wander(events, pos, t, rand(80, 220), cfg.wander_strength * 0.6);
    }
    return events;
  }

  const challengeRes = await fetch(`${API_BASE}/api/challenge/new`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ challenge_type: "button_sequence" }),
  });
  const challenge = await challengeRes.json();
  const events = generateEvents(challenge);

  const verifyRes = await fetch(`${API_BASE}/api/challenge/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: challenge.session_id, events, participant: "bot_custom_humanlike" }),
  });
  const result = await verifyRes.json();

  // Update visible dashboard if helper functions exist.
  if (typeof startNewChallenge === "function") {
    location.reload();
  }

  return {
    human_confidence: result.human_confidence,
    model: result.evaluation?.model_type,
    bot_probability: result.evaluation?.bot_probability,
    passed: result.task_result?.passed,
    flags: result.evaluation?.flags || [],
    event_count: events.length,
  };
})();
