// Behavioral CAPTCHA Challenge Suite Engine (v1.0)

const API_BASE = window.location.origin.includes("http") ? window.location.origin : "http://127.0.0.1:8050";

let currentTab = "sequential";
let activeModel = "random_forest_humanlike";

let telemetryEvents = [];
let seqCurrentStep = 1;
let challengeStartTime = null;

let mazeCanvas, mazeCtx;
let mazeState = "idle"; // idle, playing, completed, failed
let mazePathPoints = [];

let isBotSimulating = false;
let simBotCursor;

document.addEventListener("DOMContentLoaded", () => {
    initModelSelector();
    initTabs();
    initSequentialChallenge();
    initMazeChallenge();
    initFormChallenge();
    initBotAutoplay();
    startTelemetryStreamLoop();
});

// 1. Model Selector Engine
function initModelSelector() {
    const select = document.getElementById("model-select");

    // Fetch live model registry
    fetch(`${API_BASE}/api/models`)
        .then(res => res.json())
        .then(data => {
            activeModel = data.active_model;
            select.value = activeModel;
        })
        .catch(err => console.warn("Failed fetching model registry:", err));

    select.addEventListener("change", (e) => {
        const chosen = e.target.value;
        fetch(`${API_BASE}/api/models/activate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_name: chosen })
        })
        .then(res => res.json())
        .then(data => {
            activeModel = data.active_model;
            resetTelemetry();
        });
    });
}

// 2. Challenge Tabs Navigation
function initTabs() {
    const tabs = document.querySelectorAll(".tab-btn");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            currentTab = tab.dataset.tab;
            document.getElementById(`stage-${currentTab}`).classList.add("active");

            updateTabMetadata(currentTab);
            resetChallengeState();
        });
    });
}

function updateTabMetadata(tab) {
    const titleEl = document.getElementById("challenge-title");
    const descEl = document.getElementById("challenge-desc");

    if (tab === "sequential") {
        titleEl.textContent = "Sequential Button Precision Test";
        descEl.textContent = "Click the numbered targets in sequence from 1 to 4.";
    } else if (tab === "maze") {
        titleEl.textContent = "Precision Mouse Maze Navigation";
        descEl.textContent = "Guide your cursor through the winding corridor from START to FINISH without touching walls.";
        resizeMazeCanvas();
    } else if (tab === "form") {
        titleEl.textContent = "Form Data Entry Verification";
        descEl.textContent = "Fill out the security form fields and click Submit Verification.";
    }
}

function resetTelemetry() {
    telemetryEvents = [];
    challengeStartTime = null;

    document.getElementById("hud-pct").textContent = "--%";
    document.getElementById("stat-straightness").textContent = "--";
    document.getElementById("stat-velocity").textContent = "-- px/ms";
    document.getElementById("stat-acc-std").textContent = "--";
    document.getElementById("stat-jitter").textContent = "--";
    document.getElementById("stat-dwell").textContent = "-- ms";

    updateHudRing(0.5, "WAITING FOR INPUT", "status-neutral");
    document.getElementById("flags-ul").innerHTML = '<li class="flag-placeholder">Complete challenge tasks to generate live biometric analysis...</li>';
}

function resetChallengeState() {
    resetTelemetry();

    // Reset Sequential Buttons
    seqCurrentStep = 1;
    for (let i = 1; i <= 4; i++) {
        const el = document.getElementById(`target-${i}`);
        el.className = `seq-target-badge ${i === 1 ? '' : 'disabled'}`;
    }
    document.getElementById("seq-hint").textContent = "Click Button [1] to begin";

    // Reset Maze
    mazeState = "idle";
    mazePathPoints = [];
    if (mazeCtx) drawMaze();

    // Reset Form
    document.getElementById("captcha-form").reset();
}

// 3. Challenge 1: Sequential Buttons Logic
function initSequentialChallenge() {
    for (let i = 1; i <= 4; i++) {
        const btn = document.getElementById(`target-${i}`);
        btn.addEventListener("click", (e) => {
            if (isBotSimulating) return;
            handleSeqClick(i, e.clientX, e.clientY);
        });
    }
}

function handleSeqClick(num, x, y) {
    if (num !== seqCurrentStep) return;

    if (seqCurrentStep === 1) {
        challengeStartTime = performance.now();
    }

    // Record telemetry click
    telemetryEvents.push({ x, y, t: performance.now(), type: "mousedown", button: "left" });
    telemetryEvents.push({ x, y, t: performance.now() + 80, type: "mouseup", button: "left" });

    const currentEl = document.getElementById(`target-${num}`);
    currentEl.className = "seq-target-badge completed";

    if (num < 4) {
        seqCurrentStep++;
        const nextEl = document.getElementById(`target-${seqCurrentStep}`);
        nextEl.className = "seq-target-badge";
        document.getElementById("seq-hint").textContent = `Great! Now click Button [${seqCurrentStep}]`;
    } else {
        // Challenge Completed
        document.getElementById("seq-hint").textContent = "🎉 Challenge Completed!";
        const durationSec = ((performance.now() - challengeStartTime) / 1000.0).toFixed(2);
        triggerChallengeEvaluation(durationSec);
    }
}

// 4. Challenge 2: Precision Maze Logic
function initMazeChallenge() {
    mazeCanvas = document.getElementById("maze-canvas");
    mazeCtx = mazeCanvas.getContext("2d");
    simBotCursor = document.getElementById("sim-bot-cursor");

    window.addEventListener("resize", resizeMazeCanvas);

    mazeCanvas.addEventListener("pointermove", (e) => {
        if (isBotSimulating || currentTab !== "maze") return;

        const rect = mazeCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        recordPointerEvent(e.clientX, e.clientY, e.timeStamp);

        if (mazeState === "idle") {
            // Check if user entered Start Zone
            if (isInsideStartZone(x, y)) {
                mazeState = "playing";
                challengeStartTime = performance.now();
                mazePathPoints = [{ x, y }];
                document.getElementById("maze-hint").textContent = "Guide cursor through the maze path to FINISH 🏁";
            }
        } else if (mazeState === "playing") {
            mazePathPoints.push({ x, y });

            // Wall Collision Check
            if (checkMazeCollision(x, y)) {
                mazeState = "failed";
                document.getElementById("maze-hint").textContent = "❌ Wall touched! Move to START to retry.";
                setTimeout(() => { mazeState = "idle"; }, 1500);
            } else if (isInsideFinishZone(x, y)) {
                mazeState = "completed";
                document.getElementById("maze-hint").textContent = "🎉 Maze Solved!";
                const durationSec = ((performance.now() - challengeStartTime) / 1000.0).toFixed(2);
                triggerChallengeEvaluation(durationSec);
            }
        }

        drawMaze();
    });
}

function resizeMazeCanvas() {
    if (!mazeCanvas) return;
    const rect = mazeCanvas.parentElement.getBoundingClientRect();
    mazeCanvas.width = rect.width * window.devicePixelRatio;
    mazeCanvas.height = rect.height * window.devicePixelRatio;
    mazeCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    drawMaze();
}

const mazeLayout = {
    start: { x: 80, y: 80, r: 32 },
    finish: { x: 750, y: 380, r: 35 },
    corridors: [
        { x1: 80, y1: 80, x2: 450, y2: 80, width: 64 },
        { x1: 450, y1: 80, x2: 450, y2: 380, width: 64 },
        { x1: 450, y1: 380, x2: 750, y2: 380, width: 64 }
    ]
};

function isInsideStartZone(x, y) {
    return Math.hypot(x - mazeLayout.start.x, y - mazeLayout.start.y) <= mazeLayout.start.r;
}

function isInsideFinishZone(x, y) {
    return Math.hypot(x - mazeLayout.finish.x, y - mazeLayout.finish.y) <= mazeLayout.finish.r;
}

function checkMazeCollision(x, y) {
    // Return true if point is outside all corridors and start/finish zones
    if (isInsideStartZone(x, y) || isInsideFinishZone(x, y)) return false;

    for (let c of mazeLayout.corridors) {
        const halfW = c.width / 2;
        const minX = Math.min(c.x1, c.x2) - halfW;
        const maxX = Math.max(c.x1, c.x2) + halfW;
        const minY = Math.min(c.y1, c.y2) - halfW;
        const maxY = Math.max(c.y1, c.y2) + halfW;

        if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
            return false;
        }
    }

    return true;
}

function drawMaze() {
    if (!mazeCtx) return;
    const rect = mazeCanvas.parentElement.getBoundingClientRect();
    mazeCtx.clearRect(0, 0, rect.width, rect.height);

    // Draw Corridors
    mazeCtx.lineCap = "round";
    mazeCtx.lineJoin = "round";

    for (let c of mazeLayout.corridors) {
        mazeCtx.beginPath();
        mazeCtx.moveTo(c.x1, c.y1);
        mazeCtx.lineTo(c.x2, c.y2);
        mazeCtx.strokeStyle = "rgba(6, 182, 212, 0.15)";
        mazeCtx.lineWidth = c.width;
        mazeCtx.stroke();

        // Border Glow
        mazeCtx.strokeStyle = "rgba(6, 182, 212, 0.4)";
        mazeCtx.lineWidth = c.width + 4;
        mazeCtx.stroke();
    }

    // Draw Start Zone
    mazeCtx.beginPath();
    mazeCtx.arc(mazeLayout.start.x, mazeLayout.start.y, mazeLayout.start.r, 0, 2*Math.PI);
    mazeCtx.fillStyle = "rgba(16, 185, 129, 0.3)";
    mazeCtx.fill();
    mazeCtx.strokeStyle = "var(--accent-emerald)";
    mazeCtx.lineWidth = 3;
    mazeCtx.stroke();

    mazeCtx.fillStyle = "var(--text-primary)";
    mazeCtx.font = "bold 12px Outfit";
    mazeCtx.textAlign = "center";
    mazeCtx.fillText("START", mazeLayout.start.x, mazeLayout.start.y + 4);

    // Draw Finish Zone
    mazeCtx.beginPath();
    mazeCtx.arc(mazeLayout.finish.x, mazeLayout.finish.y, mazeLayout.finish.r, 0, 2*Math.PI);
    mazeCtx.fillStyle = "rgba(168, 85, 247, 0.3)";
    mazeCtx.fill();
    mazeCtx.strokeStyle = "var(--accent-purple)";
    mazeCtx.lineWidth = 3;
    mazeCtx.stroke();

    mazeCtx.fillStyle = "var(--text-primary)";
    mazeCtx.fillText("FINISH 🏁", mazeLayout.finish.x, mazeLayout.finish.y + 4);

    // Draw User Path Trail
    if (mazePathPoints.length > 1) {
        mazeCtx.beginPath();
        mazeCtx.moveTo(mazePathPoints[0].x, mazePathPoints[0].y);
        for (let p of mazePathPoints) {
            mazeCtx.lineTo(p.x, p.y);
        }
        mazeCtx.strokeStyle = mazeState === "failed" ? "var(--accent-rose)" : "var(--accent-cyan)";
        mazeCtx.lineWidth = 4;
        mazeCtx.stroke();
    }
}

// 5. Challenge 3: Form Data Entry Logic
function initFormChallenge() {
    const form = document.getElementById("captcha-form");
    const inputs = form.querySelectorAll("input, select");

    inputs.forEach(inp => {
        inp.addEventListener("focus", () => {
            if (!challengeStartTime) challengeStartTime = performance.now();
        });

        inp.addEventListener("keydown", (e) => {
            recordKeyEvent("keydown", e.key, e.timeStamp);
        });

        inp.addEventListener("keyup", (e) => {
            recordKeyEvent("keyup", e.key, e.timeStamp);
        });
    });

    document.getElementById("btn-reset-form").addEventListener("click", () => {
        form.reset();
        resetTelemetry();
    });

    document.getElementById("btn-submit-form").addEventListener("click", () => {
        if (isBotSimulating) return;

        const durationSec = challengeStartTime ? ((performance.now() - challengeStartTime) / 1000.0).toFixed(2) : "2.10";
        triggerChallengeEvaluation(durationSec);
    });
}

function recordPointerEvent(x, y, t) {
    telemetryEvents.push({ x: Math.round(x), y: Math.round(y), t: Math.round(t), type: "mousemove" });
}

function recordKeyEvent(type, key, t) {
    telemetryEvents.push({ x: 0, y: 0, t: Math.round(t), type, key });
}

// 6. Telemetry & AI Evaluation Stream Loop
function startTelemetryStreamLoop() {
    setInterval(() => {
        if (telemetryEvents.length < 5) return;

        const nowT = telemetryEvents[telemetryEvents.length - 1].t;
        const windowEvents = telemetryEvents.filter(e => (nowT - e.t) <= 3500);

        if (windowEvents.length < 5) return;

        fetch(`${API_BASE}/api/verify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                events: windowEvents,
                challenge_id: `challenge-${currentTab}`
            })
        })
        .then(res => res.json())
        .then(data => {
            applyEvaluationResult(data);
        })
        .catch(err => console.warn("Evaluation fetch error:", err));

    }, 350);
}

function applyEvaluationResult(data) {
    const evalData = data.evaluation;
    const feats = data.features;

    const botProb = evalData.bot_probability || evalData.risk_score;
    const humanProb = 1.0 - botProb;
    const humanPct = Math.round(humanProb * 100);

    document.getElementById("hud-pct").textContent = `${humanPct}%`;

    let statusText = "VERIFIED HUMAN";
    let statusClass = "status-passed";

    if (evalData.is_bot) {
        statusText = "BLOCKED BOT";
        statusClass = "status-blocked";
    }

    updateHudRing(humanProb, statusText, statusClass);

    document.getElementById("stat-straightness").textContent = feats.straightness_index.toFixed(3);
    document.getElementById("stat-velocity").textContent = `${feats.velocity_mean.toFixed(2)} px/ms`;
    document.getElementById("stat-acc-std").textContent = feats.acceleration_std.toFixed(4);
    document.getElementById("stat-jitter").textContent = feats.jitter_ratio.toFixed(3);
    document.getElementById("stat-dwell").textContent = feats.avg_click_dwell_ms >= 0 ? `${feats.avg_click_dwell_ms.toFixed(1)} ms` : "--";

    const flagsUl = document.getElementById("flags-ul");
    flagsUl.innerHTML = "";
    if (evalData.flags && evalData.flags.length > 0) {
        evalData.flags.forEach(flag => {
            const li = document.createElement("li");
            li.className = "flag-blocked";
            li.textContent = flag;
            flagsUl.appendChild(li);
        });
    } else {
        const li = document.createElement("li");
        li.className = "flag-passed";
        li.textContent = "✓ Verified organic human biometric interaction profile.";
        flagsUl.appendChild(li);
    }
}

function updateHudRing(humanRatio, text, pillClass) {
    const ring = document.getElementById("hud-ring-fill");
    const circumference = 364.42; // 2 * pi * 58
    const offset = circumference - (humanRatio * circumference);
    ring.style.strokeDashoffset = offset;

    if (humanRatio >= 0.5) {
        ring.style.stroke = "var(--accent-emerald)";
    } else {
        ring.style.stroke = "var(--accent-rose)";
    }

    const pill = document.getElementById("status-pill");
    pill.className = `status-pill ${pillClass}`;
    pill.textContent = text;
}

// 7. Challenge Completion Evaluation & Modal
function triggerChallengeEvaluation(durationSec) {
    if (telemetryEvents.length < 5) return;

    fetch(`${API_BASE}/api/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            events: telemetryEvents,
            challenge_id: `completion-${currentTab}`
        })
    })
    .then(res => res.json())
    .then(data => {
        showCompletionModal(data, durationSec);
    });
}

function showCompletionModal(data, durationSec) {
    const evalData = data.evaluation;
    const feats = data.features;

    const botProb = evalData.bot_probability || evalData.risk_score;
    const humanProb = 1.0 - botProb;
    const humanPct = Math.round(humanProb * 100);

    const modal = document.getElementById("completion-modal");
    const verdictBanner = document.getElementById("modal-verdict-banner");
    const verdictText = document.getElementById("modal-verdict-text");
    const iconEl = document.getElementById("modal-icon");

    if (!evalData.is_bot) {
        iconEl.textContent = "🏆";
        verdictBanner.style.borderColor = "var(--accent-emerald)";
        verdictBanner.style.background = "rgba(16, 185, 129, 0.15)";
        verdictText.style.color = "var(--accent-emerald)";
        verdictText.textContent = `VERIFIED HUMAN (${humanPct}%)`;
    } else {
        iconEl.textContent = "🤖";
        verdictBanner.style.borderColor = "var(--accent-rose)";
        verdictBanner.style.background = "rgba(244, 63, 94, 0.15)";
        verdictText.style.color = "var(--accent-rose)";
        verdictText.textContent = `BLOCKED BOT (${humanPct}% Human)`;
    }

    document.getElementById("modal-time").textContent = `${durationSec}s`;
    document.getElementById("modal-model").textContent = activeModel.includes("rf") ? "Supervised RF" : "Isolation Forest";
    document.getElementById("modal-vel").textContent = `${feats.velocity_mean.toFixed(2)} px/ms`;
    document.getElementById("modal-jitter").textContent = feats.jitter_ratio.toFixed(3);

    modal.classList.add("active");

    document.getElementById("btn-modal-close").onclick = () => {
        modal.classList.remove("active");
        resetChallengeState();
    };
}

// 8. Bot Auto-Play Engine
function initBotAutoplay() {
    document.getElementById("btn-bot-linear").addEventListener("click", () => runBotAutoplay("linear"));
    document.getElementById("btn-bot-bezier").addEventListener("click", () => runBotAutoplay("bezier"));
    document.getElementById("btn-bot-humanoid").addEventListener("click", () => runBotAutoplay("humanoid"));
}

function runBotAutoplay(botType) {
    if (isBotSimulating) return;
    isBotSimulating = true;
    simBotCursor.style.display = "block";
    resetTelemetry();

    let waypoints = [];

    if (currentTab === "sequential") {
        const t1 = getElementCenter("target-1");
        const t2 = getElementCenter("target-2");
        const t3 = getElementCenter("target-3");
        const t4 = getElementCenter("target-4");

        waypoints = [t1, t2, t3, t4];
    } else if (currentTab === "maze") {
        const stage = document.getElementById("interactive-stage").getBoundingClientRect();
        waypoints = [
            { x: stage.left + 80, y: stage.top + 80 },
            { x: stage.left + 450, y: stage.top + 80 },
            { x: stage.left + 450, y: stage.top + 380 },
            { x: stage.left + 750, y: stage.top + 380 }
        ];
    } else {
        // Form
        const f1 = getElementCenter("form-name");
        const f2 = getElementCenter("form-email");
        const f3 = getElementCenter("form-code");
        const sub = getElementCenter("btn-submit-form");

        waypoints = [f1, f2, f3, sub];
    }

    // Generate multi-waypoint bot trajectory
    generateBotTrajectory(botType, waypoints, (allEvents) => {
        telemetryEvents = allEvents;
        animateBotCursor(allEvents, () => {
            isBotSimulating = false;
            triggerChallengeEvaluation("1.85");
        });
    });
}

function getElementCenter(id) {
    const el = document.getElementById(id);
    const rect = el.getBoundingClientRect();
    return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
}

function generateBotTrajectory(botType, waypoints, onComplete) {
    let allEvents = [];
    let tCur = performance.now();

    for (let i = 0; i < waypoints.length - 1; i++) {
        const p1 = waypoints[i];
        const p2 = waypoints[i + 1];
        const n = 35;
        const dt = 14;

        if (botType === "linear") {
            for (let k = 0; k < n; k++) {
                const u = k / (n - 1);
                allEvents.push({
                    x: Math.round(p1.x + u * (p2.x - p1.x)),
                    y: Math.round(p1.y + u * (p2.y - p1.y)),
                    t: Math.round(tCur + k * dt),
                    type: "mousemove"
                });
            }
        } else if (botType === "bezier") {
            const ctrlX1 = p1.x + (p2.x - p1.x) * 0.25 + 60;
            const ctrlY1 = p1.y + (p2.y - p1.y) * 0.25 - 50;
            const ctrlX2 = p1.x + (p2.x - p1.x) * 0.75 - 40;
            const ctrlY2 = p1.y + (p2.y - p1.y) * 0.75 + 50;
            for (let k = 0; k < n; k++) {
                const u = k / (n - 1);
                const cx = (1-u)**3 * p1.x + 3*(1-u)**2 * u * ctrlX1 + 3*(1-u) * u**2 * ctrlX2 + u**3 * p2.x;
                const cy = (1-u)**3 * p1.y + 3*(1-u)**2 * u * ctrlY1 + 3*(1-u) * u**2 * ctrlY2 + u**3 * p2.y;
                allEvents.push({ x: Math.round(cx), y: Math.round(cy), t: Math.round(tCur + k * dt), type: "mousemove" });
            }
        } else {
            // Humanoid
            const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
            const perpX = -(p2.y - p1.y) / (dist + 1e-6);
            const perpY = (p2.x - p1.x) / (dist + 1e-6);
            const amp = 0.15 * dist;
            for (let k = 0; k < n; k++) {
                const u = k / (n - 1);
                const poly = 10 * (u**3) - 15 * (u**4) + 6 * (u**5);
                const basex = p1.x + (p2.x - p1.x) * poly;
                const basey = p1.y + (p2.y - p1.y) * poly;
                const curve = Math.sin(Math.pi * u) * amp;
                const tremorX = 0.8 * Math.sin(2 * Math.pi * 8.5 * (k * 0.015));
                const tremorY = 0.8 * Math.cos(2 * Math.pi * 8.5 * (k * 0.015));
                allEvents.push({
                    x: Math.round(basex + perpX * curve + tremorX),
                    y: Math.round(basey + perpY * curve + tremorY),
                    t: Math.round(tCur + k * dt),
                    type: "mousemove"
                });
            }
        }

        // Add Click at waypoint target
        const lastT = allEvents[allEvents.length - 1].t;
        allEvents.push({ x: Math.round(p2.x), y: Math.round(p2.y), t: lastT + 5, type: "mousedown", button: "left" });
        allEvents.push({ x: Math.round(p2.x), y: Math.round(p2.y), t: lastT + (botType === "humanoid" ? 85 : 3), type: "mouseup", button: "left" });

        tCur = lastT + 120;
    }

    onComplete(allEvents);
}

function animateBotCursor(events, onComplete) {
    let idx = 0;
    const interval = setInterval(() => {
        if (idx >= events.length) {
            clearInterval(interval);
            simBotCursor.style.display = "none";
            if (onComplete) onComplete();
            return;
        }
        const e = events[idx];
        simBotCursor.style.left = `${e.x}px`;
        simBotCursor.style.top = `${e.y}px`;
        idx++;
    }, 18);
}
