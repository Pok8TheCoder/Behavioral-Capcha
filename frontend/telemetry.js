// Behavioral CAPTCHA Telemetry & Visualization Engine

const API_BASE = "http://127.0.0.1:8000";

let telemetryEvents = [];
let isCapturing = true;
let canvas, ctx;
let simCursor;

document.addEventListener("DOMContentLoaded", () => {
    initCanvas();
    initEventListeners();
    checkBackendHealth();
});

function checkBackendHealth() {
    const statusDot = document.querySelector(".status-dot");
    const statusText = document.querySelector(".status-text");

    fetch(`${API_BASE}/api/health`)
        .then(res => res.json())
        .then(data => {
            if (data.status === "healthy") {
                statusDot.classList.remove("pulsing");
                statusDot.classList.add("connected");
                statusText.textContent = "Backend Connected (FastAPI + ML Engine)";
            }
        })
        .catch(err => {
            statusText.textContent = "Backend Offline (Run uvicorn)";
            console.warn("Backend connection check failed:", err);
        });
}

function initCanvas() {
    canvas = document.getElementById("trajectory-canvas");
    ctx = canvas.getContext("2d");
    simCursor = document.getElementById("simulated-cursor");

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
}

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    renderCanvas();
}

function initEventListeners() {
    // Record mouse movement relative to canvas/viewport
    window.addEventListener("mousemove", (e) => {
        if (!isCapturing) return;
        
        const timestamp = performance.now();
        const eventData = {
            x: Math.round(e.clientX),
            y: Math.round(e.clientY),
            t: Math.round(timestamp * 100) / 100,
            type: "mousemove"
        };

        telemetryEvents.push(eventData);
        updateOverlayStats();
        renderCanvas();
    });

    window.addEventListener("mousedown", (e) => {
        if (!isCapturing) return;
        telemetryEvents.push({
            x: Math.round(e.clientX),
            y: Math.round(e.clientY),
            t: Math.round(performance.now() * 100) / 100,
            type: "mousedown"
        });
    });

    window.addEventListener("mouseup", (e) => {
        if (!isCapturing) return;
        telemetryEvents.push({
            x: Math.round(e.clientX),
            y: Math.round(e.clientY),
            t: Math.round(performance.now() * 100) / 100,
            type: "mouseup"
        });
    });

    // CAPTCHA Checkbox verification trigger
    const checkbox = document.getElementById("captcha-checkbox");
    checkbox.addEventListener("click", () => {
        verifyUserBehavior();
    });

    // Reset button
    document.getElementById("btn-reset-telemetry").addEventListener("click", resetTelemetry);

    // Bot simulator buttons
    document.getElementById("btn-bot-linear").addEventListener("click", () => runBotSimulation("linear"));
    document.getElementById("btn-bot-bezier").addEventListener("click", () => runBotSimulation("bezier"));
    document.getElementById("btn-bot-jitter").addEventListener("click", () => runBotSimulation("jitter"));
    document.getElementById("btn-bot-teleport").addEventListener("click", () => runBotSimulation("teleport"));
}

function updateOverlayStats() {
    const pointsEl = document.getElementById("overlay-points");
    const durationEl = document.getElementById("overlay-duration");

    pointsEl.textContent = telemetryEvents.length;
    if (telemetryEvents.length > 1) {
        const dur = Math.round(telemetryEvents[telemetryEvents.length - 1].t - telemetryEvents[0].t);
        durationEl.textContent = `${dur} ms`;
    }
}

function resetTelemetry() {
    telemetryEvents = [];
    isCapturing = true;

    const checkbox = document.getElementById("captcha-checkbox");
    checkbox.className = "captcha-checkbox";

    document.getElementById("risk-score-display").textContent = "--";
    document.getElementById("risk-score-display").style.color = "var(--accent-cyan)";
    
    const badge = document.getElementById("decision-badge");
    badge.className = "decision-badge badge-neutral";
    badge.textContent = "READY";

    document.getElementById("val-straightness").textContent = "--";
    document.getElementById("val-velocity").textContent = "--";
    document.getElementById("val-vel-std").textContent = "--";
    document.getElementById("val-jitter").textContent = "--";
    document.getElementById("val-click").textContent = "--";
    document.getElementById("val-idle").textContent = "--";

    document.getElementById("flags-list").innerHTML = '<li class="no-flags">No telemetry analyzed yet. Perform a check above.</li>';

    renderCanvas();
}

function renderCanvas() {
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();

    ctx.clearRect(0, 0, rect.width, rect.height);

    // Draw background grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 1;
    const step = 40;
    for (let x = 0; x < rect.width; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, rect.height);
        ctx.stroke();
    }
    for (let y = 0; y < rect.height; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(rect.width, y);
        ctx.stroke();
    }

    if (telemetryEvents.length < 2) return;

    // Filter points inside canvas space for drawing relative paths
    const points = telemetryEvents.map(e => {
        return {
            x: e.x - rect.left,
            y: e.y - rect.top,
            t: e.t,
            type: e.type,
            isBot: e.isBot || false
        };
    });

    // Draw trajectory segments with velocity colors
    for (let i = 1; i < points.length; i++) {
        const p1 = points[i - 1];
        const p2 = points[i];

        if (p1.type !== "mousemove" && p1.type !== "move") continue;

        const dt = Math.max(0.1, p2.t - p1.t);
        const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
        const vel = dist / dt;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);

        if (p2.isBot) {
            ctx.strokeStyle = "#f43f5e"; // Rose for Bot
            ctx.lineWidth = 3;
        } else {
            // Cyan for fast, Purple for slow
            const speedRatio = Math.min(1.0, vel / 2.0);
            ctx.strokeStyle = `rgb(${Math.round(6 + speedRatio * 162)}, ${Math.round(182 - speedRatio * 97)}, ${Math.round(212 + speedRatio * 43)})`;
            ctx.lineWidth = 2.5;
        }

        ctx.stroke();
    }

    // Draw points
    for (let i = 0; i < points.length; i += 3) {
        const p = points[i];
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = p.isBot ? "#f43f5e" : "#06b6d4";
        ctx.fill();
    }
}

function verifyUserBehavior() {
    const checkbox = document.getElementById("captcha-checkbox");
    checkbox.classList.add("loading");

    fetch(`${API_BASE}/api/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            events: telemetryEvents,
            challenge_id: "demo-user-check"
        })
    })
    .then(res => res.json())
    .then(data => {
        checkbox.classList.remove("loading");
        applyEvaluationResults(data);
    })
    .catch(err => {
        checkbox.classList.remove("loading");
        console.error("Verification failed:", err);
        alert("Failed connecting to backend verification endpoint.");
    });
}

function runBotSimulation(botType) {
    const canvasRect = canvas.getBoundingClientRect();
    const targetCheckbox = document.getElementById("captcha-checkbox").getBoundingClientRect();

    const startX = canvasRect.left + 30;
    const startY = canvasRect.top + 30;
    const endX = targetCheckbox.left + targetCheckbox.width / 2;
    const endY = targetCheckbox.top + targetCheckbox.height / 2;

    fetch(`${API_BASE}/api/simulate-bot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            bot_type: botType,
            start_x: startX,
            start_y: startY,
            end_x: endX,
            end_y: endY,
            steps: botType === "teleport" ? 2 : 40
        })
    })
    .then(res => res.json())
    .then(data => {
        const botEvents = data.events.map(e => ({ ...e, isBot: true }));

        // Animate simulated bot cursor visually
        animateBotCursor(botEvents, () => {
            // Evaluate bot payload
            fetch(`${API_BASE}/api/verify`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    events: botEvents,
                    challenge_id: `bot-sim-${botType}`
                })
            })
            .then(res => res.json())
            .then(evalData => {
                telemetryEvents = botEvents;
                renderCanvas();
                updateOverlayStats();
                applyEvaluationResults(evalData);
            });
        });
    })
    .catch(err => {
        console.error("Bot simulation request failed:", err);
    });
}

function animateBotCursor(events, onComplete) {
    simCursor.style.display = "block";
    let index = 0;

    const interval = setInterval(() => {
        if (index >= events.length) {
            clearInterval(interval);
            simCursor.style.display = "none";
            if (onComplete) onComplete();
            return;
        }
        const e = events[index];
        simCursor.style.left = `${e.x}px`;
        simCursor.style.top = `${e.y}px`;
        index++;
    }, 20);
}

function applyEvaluationResults(data) {
    const evalData = data.evaluation;
    const feats = data.features;

    const checkbox = document.getElementById("captcha-checkbox");
    const decisionBadge = document.getElementById("decision-badge");
    const riskScoreDisplay = document.getElementById("risk-score-display");

    checkbox.className = "captcha-checkbox";
    if (evalData.is_bot) {
        checkbox.classList.add("failed");
        decisionBadge.className = "decision-badge badge-blocked";
        decisionBadge.textContent = "BLOCKED (BOT)";
        riskScoreDisplay.style.color = "var(--accent-rose)";
    } else {
        checkbox.classList.add("passed");
        decisionBadge.className = "decision-badge badge-passed";
        decisionBadge.textContent = "PASSED (HUMAN)";
        riskScoreDisplay.style.color = "var(--accent-emerald)";
    }

    riskScoreDisplay.textContent = evalData.risk_score.toFixed(2);

    // Update metrics UI
    document.getElementById("val-straightness").textContent = feats.straightness_index.toFixed(3);
    document.getElementById("val-velocity").textContent = feats.velocity_mean.toFixed(2);
    document.getElementById("val-vel-std").textContent = feats.velocity_std.toFixed(3);
    document.getElementById("val-jitter").textContent = feats.jitter_ratio.toFixed(3);
    document.getElementById("val-click").textContent = `${feats.avg_click_dwell_ms.toFixed(1)} ms`;
    document.getElementById("val-idle").textContent = (feats.idle_ratio * 100).toFixed(1) + "%";

    // Update flags
    const flagsList = document.getElementById("flags-list");
    flagsList.innerHTML = "";
    if (evalData.flags && evalData.flags.length > 0) {
        evalData.flags.forEach(flag => {
            const li = document.createElement("li");
            li.textContent = flag;
            flagsList.appendChild(li);
        });
    } else {
        flagsList.innerHTML = '<li class="no-flags" style="color: var(--accent-emerald);">✓ Natural human behavior signature verified. Zero flags.</li>';
    }
}
