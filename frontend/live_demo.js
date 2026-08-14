// Dynamic API Base URL matching origin port
let API_BASE = window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8010";
const MODEL_API_CANDIDATES = [...new Set([API_BASE, "http://127.0.0.1:8010", "http://127.0.0.1:8000"])];

let canvas, ctx;
let telemetryEvents = [];
let trailPoints = [];
let botCursor;
let isBotSimulating = false;

const MODEL_LABELS = {
    random_forest_humanlike: "Random Forest (Human-like)",
    isolation_forest: "Isolation Forest",
    random_forest: "Random Forest (Legacy)",
};

function formatModelLabel(modelName) {
    return MODEL_LABELS[modelName] || modelName.replaceAll("_", " ");
}

document.addEventListener("DOMContentLoaded", () => {
    initCanvas();
    initEventListeners();
    loadModelSelector();
    startTelemetryStreamLoop();
});

function initCanvas() {
    canvas = document.getElementById("full-canvas");
    ctx = canvas.getContext("2d");
    botCursor = document.getElementById("bot-cursor");

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
}

function resizeCanvas() {
    canvas.width = window.innerWidth * window.devicePixelRatio;
    canvas.height = window.innerHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

function initEventListeners() {
    window.addEventListener("pointermove", (e) => {
        if (isBotSimulating) return;

        let eventsToProcess = [];
        if (typeof e.getCoalescedEvents === "function") {
            const coalesced = e.getCoalescedEvents();
            if (coalesced && coalesced.length > 0) {
                eventsToProcess = coalesced;
            } else {
                eventsToProcess = [e];
            }
        } else {
            eventsToProcess = [e];
        }

        const now = performance.now();

        eventsToProcess.forEach(evt => {
            const x = Math.round(evt.clientX);
            const y = Math.round(evt.clientY);
            const t = Math.round(evt.timeStamp * 100) / 100;

            telemetryEvents.push({ x, y, t, type: "mousemove" });
            trailPoints.push({ x, y, alpha: 1.0, isBot: false });
            if (trailPoints.length > 400) trailPoints.shift();
        });
    });

    window.addEventListener("pointerdown", (e) => {
        if (isBotSimulating) return;
        telemetryEvents.push({
            x: Math.round(e.clientX),
            y: Math.round(e.clientY),
            t: Math.round(e.timeStamp * 100) / 100,
            type: "mousedown"
        });
    });

    window.addEventListener("pointerup", (e) => {
        if (isBotSimulating) return;
        telemetryEvents.push({
            x: Math.round(e.clientX),
            y: Math.round(e.clientY),
            t: Math.round(e.timeStamp * 100) / 100,
            type: "mouseup"
        });
    });

    document.getElementById("btn-reset").addEventListener("click", resetCanvas);

    document.getElementById("btn-bot-linear").addEventListener("click", () => triggerBotSimulation("linear"));
    document.getElementById("btn-bot-bezier").addEventListener("click", () => triggerBotSimulation("bezier"));
    document.getElementById("btn-bot-humanoid").addEventListener("click", () => triggerBotSimulation("humanoid"));
    document.getElementById("btn-bot-teleport").addEventListener("click", () => triggerBotSimulation("teleport"));
    document.getElementById("active-model").addEventListener("change", activateSelectedModel);
}

function loadModelSelector() {
    updateModelSelector({
        active_model: "random_forest_humanlike",
        available_models: {
            random_forest_humanlike: { available: true },
            isolation_forest: { available: true },
            random_forest: { available: true }
        }
    });

    const tryCandidate = (index) => {
        if (index >= MODEL_API_CANDIDATES.length) {
            console.warn("Model API is unavailable on all configured local servers.");
            return;
        }

        const candidate = MODEL_API_CANDIDATES[index];
        fetch(`${candidate}/api/models`)
            .then(res => {
                if (!res.ok) throw new Error(`Model API returned ${res.status}`);
                return res.json();
            })
            .then(data => {
                API_BASE = candidate;
                updateModelSelector(data);
            })
            .catch(() => tryCandidate(index + 1));
    };

    tryCandidate(0);
}

function updateModelSelector(data) {
    const selector = document.getElementById("active-model");
    const options = [];

    Object.entries(data.available_models || {}).forEach(([modelName, details]) => {
        if (!details.available) return;
        const option = document.createElement("option");
        option.value = modelName;
        option.textContent = formatModelLabel(modelName);
        option.selected = modelName === data.active_model;
        options.push(option);
    });

    if (options.length === 0) {
        selector.innerHTML = "";
        const fallback = document.createElement("option");
        fallback.value = "";
        fallback.disabled = true;
        fallback.selected = true;
        fallback.textContent = "No models available";
        selector.appendChild(fallback);
        return;
    }

    selector.innerHTML = "";
    options.forEach(option => selector.appendChild(option));
}

function activateSelectedModel(event) {
    const selector = event.target;
    selector.disabled = true;

    fetch(`${API_BASE}/api/models/activate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: selector.value })
    })
    .then(res => {
        if (!res.ok) throw new Error("Model activation failed");
        return res.json();
    })
    .then(data => {
        updateModelSelector(data);
        resetCanvas();
    })
    .catch(err => {
        console.warn("Model activation error:", err);
        loadModelSelector();
    })
    .finally(() => {
        selector.disabled = false;
    });
}

function resetCanvas() {
    telemetryEvents = [];
    trailPoints = [];
    isBotSimulating = false;

    updateConfidenceRing(0.5, "SWIRL MOUSE", "neutral");
    document.getElementById("confidence-pct").textContent = "--%";
    document.getElementById("val-straightness").textContent = "--";
    document.getElementById("val-velocity").textContent = "--";
    document.getElementById("val-acc-std").textContent = "--";
    document.getElementById("val-jitter").textContent = "--";

    document.getElementById("flags-list").innerHTML = '<li class="flag-item idle">Move cursor across canvas to stream telemetry...</li>';
}

function startTelemetryStreamLoop() {
    setInterval(() => {
        if (telemetryEvents.length < 8) return;

        // Extract active movement window (last 3.5 seconds)
        const nowT = telemetryEvents[telemetryEvents.length - 1].t;
        const windowEvents = telemetryEvents.filter(e => (nowT - e.t) <= 3500);

        if (windowEvents.length < 5) return;

        fetch(`${API_BASE}/api/verify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                events: windowEvents,
                challenge_id: "live-canvas-stream"
            })
        })
        .then(res => res.json())
        .then(data => {
            applyEvaluationResult(data);
        })
        .catch(err => {
            console.warn("Backend evaluation fetch error:", err);
        });

    }, 350);

    function animate() {
        renderCanvasTrail();
        requestAnimationFrame(animate);
    }
    animate();
}

function applyEvaluationResult(data) {
    const evalData = data.evaluation;
    const feats = data.features;

    const botProb = evalData.bot_probability || evalData.risk_score;
    const humanProb = 1.0 - botProb;
    const humanPct = Math.round(humanProb * 100);

    const confidencePctEl = document.getElementById("confidence-pct");
    confidencePctEl.textContent = `${humanPct}%`;

    let statusText = "PASSED: HUMAN";
    let statusClass = "badge-passed";

    if (evalData.is_bot) {
        statusText = "BLOCKED: BOT";
        statusClass = "badge-blocked";
    }

    updateConfidenceRing(humanProb, statusText, statusClass);

    document.getElementById("val-straightness").textContent = feats.straightness_index.toFixed(3);
    document.getElementById("val-velocity").textContent = feats.velocity_mean.toFixed(2);
    document.getElementById("val-acc-std").textContent = feats.acceleration_std.toFixed(4);
    document.getElementById("val-jitter").textContent = feats.jitter_ratio.toFixed(3);

    const flagsList = document.getElementById("flags-list");
    flagsList.innerHTML = "";
    if (evalData.flags && evalData.flags.length > 0) {
        evalData.flags.forEach(flag => {
            const li = document.createElement("li");
            li.className = "flag-item blocked";
            li.textContent = flag;
            flagsList.appendChild(li);
        });
    } else {
        const li = document.createElement("li");
        li.className = "flag-item passed";
        li.textContent = "✓ Verified organic human interaction profile.";
        flagsList.appendChild(li);
    }
}

function updateConfidenceRing(humanRatio, text, badgeClass) {
    const ring = document.getElementById("confidence-ring");
    const circumference = 339.29; // 2 * pi * 54
    const offset = circumference - (humanRatio * circumference);
    ring.style.strokeDashoffset = offset;

    if (humanRatio >= 0.5) {
        ring.style.stroke = "var(--accent-emerald)";
    } else {
        ring.style.stroke = "var(--accent-rose)";
    }

    const badge = document.getElementById("status-badge");
    badge.className = `decision-badge ${badgeClass}`;
    badge.textContent = text;
}

function renderCanvasTrail() {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 1;
    for (let x = 0; x < window.innerWidth; x += 50) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, window.innerHeight); ctx.stroke();
    }
    for (let y = 0; y < window.innerHeight; y += 50) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(window.innerWidth, y); ctx.stroke();
    }

    for (let i = 1; i < trailPoints.length; i++) {
        const p1 = trailPoints[i - 1];
        const p2 = trailPoints[i];

        p1.alpha *= 0.96;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);

        if (p1.isBot) {
            ctx.strokeStyle = `rgba(244, 63, 94, ${p1.alpha})`;
            ctx.lineWidth = 4;
        } else {
            ctx.strokeStyle = `rgba(6, 182, 212, ${p1.alpha})`;
            ctx.lineWidth = 3;
        }
        ctx.stroke();
    }
}

function triggerBotSimulation(botType) {
    isBotSimulating = true;
    botCursor.style.display = "block";

    const startX = 100;
    const startY = 100;
    const endX = window.innerWidth - 300;
    const endY = window.innerHeight - 300;

    fetch(`${API_BASE}/api/simulate-bot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            bot_type: botType,
            start_x: startX,
            start_y: startY,
            end_x: endX,
            end_y: endY,
            steps: botType === "teleport" ? 2 : 45
        })
    })
    .then(res => res.json())
    .then(data => {
        const botEvents = data.events.map(e => ({ ...e, isBot: true }));
        animateBotCursor(botEvents, () => {
            telemetryEvents = botEvents;
            botEvents.forEach(be => {
                trailPoints.push({ x: be.x, y: be.y, alpha: 1.0, isBot: true });
            });

            fetch(`${API_BASE}/api/verify`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    events: botEvents,
                    challenge_id: `live-bot-${botType}`
                })
            })
            .then(res => res.json())
            .then(evalData => {
                applyEvaluationResult(evalData);
                isBotSimulating = false;
            });
        });
    });
}

function animateBotCursor(events, onComplete) {
    let idx = 0;
    const interval = setInterval(() => {
        if (idx >= events.length) {
            clearInterval(interval);
            botCursor.style.display = "none";
            if (onComplete) onComplete();
            return;
        }
        const e = events[idx];
        botCursor.style.left = `${e.x}px`;
        botCursor.style.top = `${e.y}px`;
        idx++;
    }, 20);
}
