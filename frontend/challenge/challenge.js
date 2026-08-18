const API_BASE = window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8010";
const LOGICAL_WIDTH = 960;
const MIN_MOVE_DT_MS = 1;
const DRAG_STROKE_TYPES = new Set([
    "shape_trace", "maze_navigate", "zigzag_trace", "spiral_trace",
    "connect_dots", "hazard_run", "path_gates",
]);

let canvas, ctx;
let challenge = null;
let events = [];
let isCapturing = false;
let sequenceProgress = 0;
let pinProgress = 0;
let trail = [];
let clientInteractionStartMs = 0;
let strokeArmed = false;

document.addEventListener("DOMContentLoaded", () => {
    canvas = document.getElementById("challenge-canvas");
    ctx = canvas.getContext("2d");
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    document.getElementById("btn-new-challenge").addEventListener("click", () => startNewChallenge());
    document.getElementById("btn-next").addEventListener("click", () => {
        hideOverlay();
        startNewChallenge();
    });

    canvas.style.touchAction = "none";
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove, { passive: true });
    canvas.addEventListener("pointerrawupdate", onPointerRawUpdate, { passive: true });
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("dblclick", () => submitAttempt());

    checkHealth();
    startNewChallenge();
});

function resizeCanvas() {
    const wrap = canvas.parentElement;
    const rect = wrap.getBoundingClientRect();
    canvas.width = LOGICAL_WIDTH;
    canvas.height = LOGICAL_HEIGHT;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    drawChallenge();
}

function canvasPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = LOGICAL_WIDTH / rect.width;
    const scaleY = LOGICAL_HEIGHT / rect.height;
    return {
        x: (evt.clientX - rect.left) * scaleX,
        y: (evt.clientY - rect.top) * scaleY,
    };
}

function recordEvent(x, y, t, type, pointerMeta) {
    const evt = { x: Math.round(x), y: Math.round(y), t: Math.round(t * 100) / 100, type };
    if (pointerMeta) {
        if (pointerMeta.pid != null) evt.pid = pointerMeta.pid;
        if (pointerMeta.pt) evt.pt = pointerMeta.pt;
        if (pointerMeta.src) evt.src = pointerMeta.src;
    }
    if (type === "mousemove") {
        for (let i = events.length - 1; i >= 0; i--) {
            if (events[i].type !== "mousemove") continue;
            if (evt.t - events[i].t < MIN_MOVE_DT_MS) return;
            break;
        }
    }
    events.push(evt);
    document.getElementById("meta-events").textContent = String(events.length);
    if (type === "mousemove") {
        trail.push({ x, y });
        if (trail.length > 600) trail.shift();
    }
}

async function sealedVerifyBody() {
    if (!challenge || !challenge.telemetry_key) {
        throw new Error("Missing local capture key. Start a new challenge.");
    }
    const sealed = await sealTelemetryEvents(challenge.telemetry_key, events);
    persistSealedLocally(challenge.session_id, sealed);
    return {
        session_id: challenge.session_id,
        participant: "human",
        verify_token: challenge.verify_token,
        client_started_ms: clientInteractionStartMs,
        client_submitted_ms: performance.now(),
        source: "dashboard",
        sealed_telemetry: sealed,
    };
}

function onPointerDown(evt) {
    if (!challenge || !isCapturing) return;
    canvas.setPointerCapture(evt.pointerId);
    const p = canvasPoint(evt);
    const meta = { pid: evt.pointerId, pt: evt.pointerType || "mouse", src: "canvas" };
    if (challenge.challenge_type === "path_gates") {
        const start = challenge.spec.start;
        const tol = challenge.spec.start_tolerance_px || 32;
        strokeArmed = Math.hypot(p.x - start[0], p.y - start[1]) <= tol;
    } else {
        strokeArmed = DRAG_STROKE_TYPES.has(challenge.challenge_type);
    }
    recordEvent(p.x, p.y, performance.now(), "mousedown", meta);
    drawChallenge();
}

function recordPointerSamples(evt) {
    const points = typeof evt.getCoalescedEvents === "function" ? evt.getCoalescedEvents() : [evt];
    points.forEach(e => {
        const p = canvasPoint(e);
        recordEvent(p.x, p.y, e.timeStamp || performance.now(), "mousemove", { pid: e.pointerId, pt: e.pointerType || "mouse", src: "canvas" });
    });
}

function onPointerRawUpdate(evt) {
    if (!challenge || !isCapturing) return;
    recordPointerSamples(evt);
}

function onPointerMove(evt) {
    if (!challenge || !isCapturing) return;
    if (!("onpointerrawupdate" in window)) recordPointerSamples(evt);
    drawChallenge();
    if (!DRAG_STROKE_TYPES.has(challenge.challenge_type)) maybeAutoComplete();
}

function onPointerUp(evt) {
    if (!challenge || !isCapturing) return;
    const p = canvasPoint(evt);
    recordEvent(p.x, p.y, evt.timeStamp || performance.now(), "mouseup", { pid: evt.pointerId, pt: evt.pointerType || "mouse", src: "canvas" });
    if (DRAG_STROKE_TYPES.has(challenge.challenge_type)) {
        drawChallenge();
        if (challenge.challenge_type === "path_gates" && !strokeArmed) return;
        if (events.length >= 4) submitAttempt();
        return;
    }
    handleClick(p.x, p.y);
    drawChallenge();
    maybeAutoComplete();
}

function handleClick(x, y) {
    if (!challenge) return;
    const type = challenge.challenge_type;
    const spec = challenge.spec;

    if (type === "button_sequence") {
        const tol = spec.click_tolerance_px;
        for (const btn of spec.buttons) {
            if (Math.hypot(x - btn.x, y - btn.y) <= tol + btn.radius) {
                if (btn.id === spec.sequence[sequenceProgress]) {
                    sequenceProgress += 1;
                    btn.clicked = true;
                }
                break;
            }
        }
    }

    if (type === "pin_entry") {
        const tol = spec.click_tolerance_px;
        for (const key of spec.keys) {
            if (Math.hypot(x - key.x, y - key.y) <= tol) {
                const expected = spec.code[pinProgress];
                if (key.digit === expected) {
                    pinProgress += 1;
                    key.pressed = true;
                }
                break;
            }
        }
    }
}

function maybeAutoComplete() {
    if (!challenge) return;
    const type = challenge.challenge_type;
    const spec = challenge.spec;

    if (type === "button_sequence" && sequenceProgress >= spec.sequence.length) {
        submitAttempt();
    }
    if (type === "pin_entry" && pinProgress >= spec.code.length) {
        submitAttempt();
    }
}

function checkHealth() {
    const pill = document.getElementById("backend-status");
    fetch(`${API_BASE}/api/health`)
        .then(r => r.json())
        .then(data => {
            const ok = data.status === "healthy";
            pill.textContent = ok ? `Online · detection ${data.detection_status}` : "Degraded";
            pill.className = `status-pill ${ok ? "ok" : "warn"}`;
        })
        .catch(() => {
            pill.textContent = "Challenge server offline";
            pill.className = "status-pill warn";
        });
}

function startNewChallenge() {
    isCapturing = false;
    events = [];
    trail = [];
    sequenceProgress = 0;
    pinProgress = 0;
    strokeArmed = false;
    hideOverlay();

    fetch(`${API_BASE}/api/challenge/new`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
    })
        .then(r => r.json())
        .then(data => {
            challenge = data;
            isCapturing = true;
            clientInteractionStartMs = performance.now();
            document.getElementById("challenge-title").textContent = data.title;
            document.getElementById("challenge-instructions").textContent = data.instructions;
            document.getElementById("meta-type").textContent = data.challenge_type;
            document.getElementById("meta-seed").textContent = String(data.seed);
            document.getElementById("meta-events").textContent = "0";
            document.getElementById("human-confidence").textContent = "—%";
            document.getElementById("decision-badge").textContent = "IN PROGRESS";
            document.getElementById("decision-badge").className = "decision-badge badge-neutral";
            document.getElementById("model-line").textContent = "Complete the task to evaluate";
            document.getElementById("task-result").textContent = "Attempt the challenge.";
            document.getElementById("flags-list").innerHTML = "<li class='flag idle'>No evaluation yet.</li>";
            if (data.spec.buttons) data.spec.buttons.forEach(b => { b.clicked = false; });
            if (data.spec.keys) data.spec.keys.forEach(k => { k.pressed = false; });
            drawChallenge();
        })
        .catch(err => {
            document.getElementById("challenge-title").textContent = "Failed to load challenge";
            document.getElementById("challenge-instructions").textContent = String(err);
        });
}

function drawChallenge() {
    if (!challenge) return;
    ctx.clearRect(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT);
    drawGrid();

    const type = challenge.challenge_type;
    if (type === "shape_trace") drawShapeTrace(challenge.spec);
    if (type === "button_sequence") drawButtonSequence(challenge.spec);
    if (type === "maze_navigate") drawMaze(challenge.spec);
    if (type === "pin_entry") drawPinEntry(challenge.spec);
    if (type === "path_gates") drawPathGates(challenge.spec);

    drawTrail();
}

function drawGrid() {
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let x = 0; x < LOGICAL_WIDTH; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, LOGICAL_HEIGHT); ctx.stroke();
    }
    for (let y = 0; y < LOGICAL_HEIGHT; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(LOGICAL_WIDTH, y); ctx.stroke();
    }
}

function drawTrail() {
    if (trail.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(trail[0].x, trail[0].y);
    for (let i = 1; i < trail.length; i++) ctx.lineTo(trail[i].x, trail[i].y);
    ctx.strokeStyle = "rgba(34, 211, 238, 0.75)";
    ctx.lineWidth = 2.5;
    ctx.stroke();
}

function drawShapeTrace(spec) {
    const pts = spec.guide_points;
    ctx.strokeStyle = "rgba(168, 85, 247, 0.55)";
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.setLineDash([]);
}

function drawButtonSequence(spec) {
    spec.buttons.forEach(btn => {
        ctx.beginPath();
        ctx.arc(btn.x, btn.y, btn.radius, 0, Math.PI * 2);
        ctx.fillStyle = btn.clicked ? "rgba(16, 185, 129, 0.35)" : "rgba(34, 211, 238, 0.2)";
        ctx.fill();
        ctx.strokeStyle = btn.clicked ? "#10b981" : "#22d3ee";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#f8fafc";
        ctx.font = "bold 16px Outfit";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(btn.id), btn.x, btn.y);
    });
}

function drawMaze(spec) {
    const { maze, origin, cell_width: cw, cell_height: ch } = spec;
    const ox = origin[0], oy = origin[1];
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(ox, oy, maze.cols * cw, maze.rows * ch);

    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 3;
    for (let r = 0; r <= maze.rows; r++) {
        for (let c = 0; c < maze.cols; c++) {
            if (maze.h_walls[r][c]) {
                ctx.beginPath();
                ctx.moveTo(ox + c * cw, oy + r * ch);
                ctx.lineTo(ox + (c + 1) * cw, oy + r * ch);
                ctx.stroke();
            }
        }
    }
    for (let r = 0; r < maze.rows; r++) {
        for (let c = 0; c <= maze.cols; c++) {
            if (maze.v_walls[r][c]) {
                ctx.beginPath();
                ctx.moveTo(ox + c * cw, oy + r * ch);
                ctx.lineTo(ox + c * cw, oy + (r + 1) * ch);
                ctx.stroke();
            }
        }
    }

    const [sx, sy] = spec.start;
    const [ex, ey] = spec.end;
    ctx.fillStyle = "#10b981";
    ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath(); ctx.arc(ex, ey, 10, 0, Math.PI * 2); ctx.fill();
}

function drawPinEntry(spec) {
    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 28px JetBrains Mono";
    ctx.fillText(`Code: ${spec.display_code}`, challenge.area.x + 40, challenge.area.y + 50);

    spec.keys.forEach(key => {
        const w = key.w, h = key.h;
        const left = key.x - w / 2, top = key.y - h / 2;
        ctx.fillStyle = key.pressed ? "rgba(16, 185, 129, 0.35)" : "rgba(30, 41, 65, 0.95)";
        ctx.strokeStyle = key.pressed ? "#10b981" : "#475569";
        ctx.lineWidth = 2;
        ctx.fillRect(left, top, w, h);
        ctx.strokeRect(left, top, w, h);
        ctx.fillStyle = "#f8fafc";
        ctx.font = "bold 18px JetBrains Mono";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(key.digit), key.x, key.y);
    });
}

function drawPathGates(spec) {
    const area = challenge.area;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(spec.start[0], spec.start[1]);
    spec.gates.forEach(g => ctx.lineTo(g.x, g.gap_y));
    ctx.lineTo(spec.end[0], spec.end[1]);
    ctx.stroke();

    spec.gates.forEach(gate => {
        const x = gate.x, gy = gate.gap_y, half = gate.gap_half, t = gate.thickness;
        ctx.fillStyle = "#94a3b8";
        ctx.fillRect(x - t / 2, challenge.area.y, t, gy - half - challenge.area.y);
        ctx.fillRect(x - t / 2, gy + half, t, challenge.area.y + challenge.area.height - (gy + half));
    });

    ctx.fillStyle = "#10b981";
    ctx.beginPath(); ctx.arc(spec.start[0], spec.start[1], 12, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath(); ctx.arc(spec.end[0], spec.end[1], 12, 0, Math.PI * 2); ctx.fill();
}

let submitting = false;

function submitAttempt() {
    if (!challenge || submitting || !isCapturing) return;
    submitting = true;
    isCapturing = false;

    sealedVerifyBody().then((body) => {
        return fetch(`${API_BASE}/api/challenge/verify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
    })
        .then(r => r.json())
        .then(showResults)
        .catch(err => {
            document.getElementById("task-result").textContent = `Submit failed: ${err}`;
        })
        .finally(() => { submitting = false; });
}

function showResults(data) {
    const evalData = data.evaluation || {};
    const task = data.task_result || {};
    const botProb = evalData.bot_probability;
    const humanPct = typeof botProb === "number" ? Math.round((1 - botProb) * 100) : (
        data.human_confidence != null ? Math.round(data.human_confidence * 100) : null
    );

    document.getElementById("human-confidence").textContent = humanPct != null ? `${humanPct}%` : "—%";
    document.getElementById("task-result").textContent = task.reason || "Done";
    document.getElementById("model-line").textContent = `Model: ${evalData.model_type || "unknown"}`;

    const badge = document.getElementById("decision-badge");
    if (evalData.is_bot) {
        badge.textContent = "BLOCKED: BOT";
        badge.className = "decision-badge badge-blocked";
    } else if (evalData.is_bot === false) {
        badge.textContent = "PASSED: HUMAN";
        badge.className = "decision-badge badge-passed";
    } else {
        badge.textContent = "UNKNOWN";
        badge.className = "decision-badge badge-neutral";
    }

    const flagsList = document.getElementById("flags-list");
    flagsList.innerHTML = "";
    const flags = evalData.flags || [];
    if (flags.length === 0) {
        flagsList.innerHTML = "<li class='flag idle'>No risk flags.</li>";
    } else {
        flags.forEach(f => {
            const li = document.createElement("li");
            li.textContent = f;
            flagsList.appendChild(li);
        });
    }

    const grid = document.getElementById("result-grid");
    grid.innerHTML = "";
    addResultCell(grid, "Task", task.passed ? "PASS" : "FAIL");
    addResultCell(grid, "Model human", humanPct != null ? `${humanPct}%` : "—");
    addResultCell(grid, "Bot prob.", evalData.bot_probability != null ? evalData.bot_probability.toFixed(3) : "—");
    addResultCell(grid, "Heuristic", evalData.heuristic_score != null ? evalData.heuristic_score.toFixed(3) : "—");
    addResultCell(grid, "Fused risk", evalData.risk_score != null ? evalData.risk_score.toFixed(3) : "—");
    addResultCell(grid, "Type", data.challenge_type);

    document.getElementById("overlay-title").textContent = task.passed ? "Challenge complete" : "Challenge incomplete";
    document.getElementById("overlay-message").textContent = task.reason || "";
    showOverlay();
}

function addResultCell(grid, label, value) {
    const div = document.createElement("div");
    div.innerHTML = `${label}<strong>${value}</strong>`;
    grid.appendChild(div);
}

function showOverlay() {
    document.getElementById("challenge-overlay").classList.remove("hidden");
}

function hideOverlay() {
    document.getElementById("challenge-overlay").classList.add("hidden");
}
