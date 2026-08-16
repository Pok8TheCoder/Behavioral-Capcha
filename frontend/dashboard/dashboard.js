const API_BASE = window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8010";
const LOGICAL_WIDTH = 960;
const LOGICAL_HEIGHT = 640;

const MODEL_LABELS = {
    random_forest_humanlike: "Random Forest (Human-like)",
    isolation_forest: "Isolation Forest",
    random_forest: "Random Forest (Legacy)",
};

let canvas, ctx, botCursor;
let challenge = null;
let events = [];
let trail = [];
let isCapturing = false;
let isBotRunning = false;
let sequenceProgress = 0;
let pinProgress = 0;
let submitting = false;

document.addEventListener("DOMContentLoaded", () => {
    canvas = document.getElementById("challenge-canvas");
    ctx = canvas.getContext("2d");
    botCursor = document.getElementById("bot-cursor");

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    document.getElementById("btn-new-challenge").addEventListener("click", () => startNewChallenge());
    document.getElementById("btn-next").addEventListener("click", () => { hideOverlay(); startNewChallenge(); });
    document.getElementById("btn-submit").addEventListener("click", () => submitAttempt("human"));
    document.getElementById("active-model").addEventListener("change", activateModel);

    document.querySelectorAll("[data-bot]").forEach(btn => {
        btn.addEventListener("click", () => runBot(btn.dataset.bot));
    });

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointerleave", onPointerUp);
    canvas.addEventListener("dblclick", () => submitAttempt("human"));

    loadModelSelector();
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
    return {
        x: (evt.clientX - rect.left) * (LOGICAL_WIDTH / rect.width),
        y: (evt.clientY - rect.top) * (LOGICAL_HEIGHT / rect.height),
    };
}

function recordEvent(x, y, t, type, isBot = false) {
    events.push({ x: Math.round(x), y: Math.round(y), t: Math.round(t * 100) / 100, type });
    document.getElementById("meta-events").textContent = String(events.length);
    if (type === "mousemove") {
        trail.push({ x, y, isBot });
        if (trail.length > 800) trail.shift();
    }
}

function onPointerDown(evt) {
    if (!challenge || !isCapturing || isBotRunning) return;
    canvas.setPointerCapture(evt.pointerId);
    const p = canvasPoint(evt);
    recordEvent(p.x, p.y, performance.now(), "mousedown");
    drawChallenge();
}

function onPointerMove(evt) {
    if (!challenge || !isCapturing || isBotRunning) return;
    const points = typeof evt.getCoalescedEvents === "function" ? evt.getCoalescedEvents() : [evt];
    points.forEach(e => {
        const p = canvasPoint(e);
        recordEvent(p.x, p.y, e.timeStamp, "mousemove");
    });
    drawChallenge();
    maybeAutoComplete();
}

function onPointerUp(evt) {
    if (!challenge || !isCapturing || isBotRunning) return;
    const p = canvasPoint(evt);
    recordEvent(p.x, p.y, evt.timeStamp, "mouseup");
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
                if (key.digit === spec.code[pinProgress]) {
                    pinProgress += 1;
                    key.pressed = true;
                }
                break;
            }
        }
    }
}

function maybeAutoComplete() {
    if (!challenge || isBotRunning) return;
    const type = challenge.challenge_type;
    const spec = challenge.spec;

    if (type === "button_sequence" && sequenceProgress >= spec.sequence.length) submitAttempt("human");
    if (type === "pin_entry" && pinProgress >= spec.code.length) submitAttempt("human");

    const last = events[events.length - 1];
    if (!last) return;

    if (type === "maze_navigate") {
        const end = spec.end;
        if (Math.hypot(last.x - end[0], last.y - end[1]) <= spec.finish_tolerance_px) submitAttempt("human");
    }
    if (type === "path_gates") {
        const end = spec.end;
        if (Math.hypot(last.x - end[0], last.y - end[1]) <= spec.finish_tolerance_px) submitAttempt("human");
    }
}

function checkHealth() {
    const pill = document.getElementById("backend-status");
    fetch(`${API_BASE}/api/health`)
        .then(r => r.json())
        .then(data => {
            pill.textContent = `Online · ${data.active_model?.replaceAll("_", " ") || "ready"}`;
            pill.className = "status-pill ok";
        })
        .catch(() => {
            pill.textContent = "Server offline";
            pill.className = "status-pill";
        });
}

function loadModelSelector() {
    updateModelSelector({
        active_model: "random_forest_humanlike",
        available_models: {
            random_forest_humanlike: { available: true },
            isolation_forest: { available: true },
            random_forest: { available: true },
        },
    });

    fetch(`${API_BASE}/api/models`)
        .then(r => r.json())
        .then(updateModelSelector)
        .catch(() => {});
}

function updateModelSelector(data) {
    const selector = document.getElementById("active-model");
    const options = [];
    Object.entries(data.available_models || {}).forEach(([name, details]) => {
        if (!details.available) return;
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = MODEL_LABELS[name] || name.replaceAll("_", " ");
        opt.selected = name === data.active_model;
        options.push(opt);
    });
    if (!options.length) return;
    selector.innerHTML = "";
    options.forEach(o => selector.appendChild(o));
}

function activateModel() {
    const selector = document.getElementById("active-model");
    selector.disabled = true;
    fetch(`${API_BASE}/api/models/activate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: selector.value }),
    })
        .then(r => r.json())
        .then(data => {
            updateModelSelector(data);
            document.getElementById("model-line").textContent = `Model: ${data.active_model.replaceAll("_", " ")}`;
        })
        .finally(() => { selector.disabled = false; });
}

function startNewChallenge() {
    isCapturing = false;
    isBotRunning = false;
    events = [];
    trail = [];
    sequenceProgress = 0;
    pinProgress = 0;
    hideOverlay();
    setButtonsDisabled(false);

    fetch(`${API_BASE}/api/challenge/new`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
    })
        .then(r => r.json())
        .then(data => {
            challenge = data;
            isCapturing = true;
            document.getElementById("challenge-title").textContent = data.title;
            document.getElementById("challenge-instructions").textContent = data.instructions;
            document.getElementById("meta-type").textContent = data.challenge_type;
            document.getElementById("meta-seed").textContent = String(data.seed);
            document.getElementById("meta-events").textContent = "0";
            document.getElementById("meta-participant").textContent = "—";
            resetAssessmentUI("IN PROGRESS", "Attempt the challenge or run a bot.");
            if (data.spec.buttons) data.spec.buttons.forEach(b => { b.clicked = false; });
            if (data.spec.keys) data.spec.keys.forEach(k => { k.pressed = false; });
            drawChallenge();
        })
        .catch(err => {
            document.getElementById("challenge-title").textContent = "Failed to load challenge";
            document.getElementById("challenge-instructions").textContent = String(err);
        });
}

function resetAssessmentUI(badge, taskText) {
    document.getElementById("human-confidence").textContent = "—%";
    document.getElementById("decision-badge").textContent = badge;
    document.getElementById("decision-badge").className = "decision-badge badge-neutral";
    document.getElementById("task-result").textContent = taskText;
    document.getElementById("flags-list").innerHTML = "<li class='flag idle'>No evaluation yet.</li>";
}

function setButtonsDisabled(disabled) {
    document.querySelectorAll("[data-bot], #btn-submit").forEach(b => { b.disabled = disabled; });
}

function submitAttempt(participant) {
    if (!challenge || submitting) return;
    submitting = true;
    isCapturing = false;
    setButtonsDisabled(true);

    fetch(`${API_BASE}/api/challenge/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: challenge.session_id,
            events,
            participant,
        }),
    })
        .then(r => r.json())
        .then(showResults)
        .catch(err => {
            document.getElementById("task-result").textContent = `Submit failed: ${err}`;
        })
        .finally(() => {
            submitting = false;
            setButtonsDisabled(false);
        });
}

function runBot(botName) {
    if (!challenge || isBotRunning || submitting) return;
    isBotRunning = true;
    isCapturing = false;
    setButtonsDisabled(true);
    events = [];
    trail = [];

    fetch(`${API_BASE}/api/challenge/run-bot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: challenge.session_id, bot: botName }),
    })
        .then(r => r.json())
        .then(data => {
            document.getElementById("meta-participant").textContent = data.participant;
            animateBotEvents(data.events, () => {
                isBotRunning = false;
                showResults(data);
                setButtonsDisabled(false);
            });
        })
        .catch(err => {
            isBotRunning = false;
            document.getElementById("task-result").textContent = `Bot run failed: ${err}`;
            setButtonsDisabled(false);
        });
}

function animateBotEvents(botEvents, onComplete) {
    botCursor.style.display = "block";
    let idx = 0;
    const moves = botEvents.filter(e => e.type === "mousemove");

    const tick = () => {
        if (idx >= moves.length) {
            botCursor.style.display = "none";
            events = botEvents.map(e => ({ ...e }));
            document.getElementById("meta-events").textContent = String(events.length);
            if (onComplete) onComplete();
            return;
        }
        const batch = moves.slice(idx, idx + 3);
        batch.forEach(e => {
            trail.push({ x: e.x, y: e.y, isBot: true });
            if (trail.length > 800) trail.shift();
        });
        const last = batch[batch.length - 1];
        const wrap = canvas.parentElement;
        botCursor.style.left = `${(last.x / LOGICAL_WIDTH) * wrap.clientWidth}px`;
        botCursor.style.top = `${(last.y / LOGICAL_HEIGHT) * wrap.clientHeight}px`;
        idx += 3;
        drawChallenge();
        setTimeout(tick, 16);
    };

    tick();
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
    for (let x = 0; x < LOGICAL_WIDTH; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, LOGICAL_HEIGHT); ctx.stroke();
    }
    for (let y = 0; y < LOGICAL_HEIGHT; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(LOGICAL_WIDTH, y); ctx.stroke();
    }
}

function drawTrail() {
    if (trail.length < 2) return;
    for (let i = 1; i < trail.length; i++) {
        const p0 = trail[i - 1], p1 = trail[i];
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.strokeStyle = p0.isBot ? "rgba(244, 63, 94, 0.8)" : "rgba(34, 211, 238, 0.75)";
        ctx.lineWidth = p0.isBot ? 3 : 2.5;
        ctx.stroke();
    }
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
    ctx.fillStyle = "#10b981";
    ctx.beginPath(); ctx.arc(spec.start[0], spec.start[1], 10, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath(); ctx.arc(spec.end[0], spec.end[1], 10, 0, Math.PI * 2); ctx.fill();
}

function drawPinEntry(spec) {
    ctx.fillStyle = "#f8fafc";
    ctx.font = "bold 28px JetBrains Mono";
    ctx.fillText(`Code: ${spec.display_code}`, challenge.area.x + 40, challenge.area.y + 50);
    spec.keys.forEach(key => {
        const left = key.x - key.w / 2, top = key.y - key.h / 2;
        ctx.fillStyle = key.pressed ? "rgba(16, 185, 129, 0.35)" : "rgba(30, 41, 65, 0.95)";
        ctx.strokeStyle = key.pressed ? "#10b981" : "#475569";
        ctx.lineWidth = 2;
        ctx.fillRect(left, top, key.w, key.h);
        ctx.strokeRect(left, top, key.w, key.h);
        ctx.fillStyle = "#f8fafc";
        ctx.font = "bold 18px JetBrains Mono";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(key.digit), key.x, key.y);
    });
}

function drawPathGates(spec) {
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
    ctx.beginPath(); ctx.arc(spec.start[0], spec.start[1], 8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath(); ctx.arc(spec.end[0], spec.end[1], 8, 0, Math.PI * 2); ctx.fill();
}

function showResults(data) {
    const evalData = data.evaluation || {};
    const task = data.task_result || {};
    const humanPct = data.human_confidence != null ? Math.round(data.human_confidence * 100) : null;

    document.getElementById("human-confidence").textContent = humanPct != null ? `${humanPct}%` : "—%";
    document.getElementById("task-result").textContent = task.reason || "Done";
    document.getElementById("model-line").textContent = `Model: ${evalData.model_type || "unknown"}`;
    document.getElementById("meta-participant").textContent = data.participant || "—";

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
    if (!flags.length) {
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
    addResultCell(grid, "Risk", evalData.risk_score != null ? evalData.risk_score.toFixed(3) : "—");
    addResultCell(grid, "Bot prob.", evalData.bot_probability != null ? evalData.bot_probability.toFixed(3) : "—");
    addResultCell(grid, "Participant", data.participant || "—");

    document.getElementById("overlay-title").textContent = task.passed ? "Complete" : "Incomplete";
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
