const API_BASE = window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8010";
const LOGICAL_WIDTH = 960;
const LOGICAL_HEIGHT = 640;

const DRAG_STROKE_TYPES = new Set([
    "shape_trace",
    "maze_navigate",
    "zigzag_trace",
    "spiral_trace",
    "connect_dots",
    "hazard_run",
    "path_gates",
]);

const LEGACY_MODEL_LABELS = {
    random_forest_challenge: "Challenge RF (→ latest Qmodel)",
    random_forest_humanlike: "Random Forest (Human-like)",
    isolation_forest: "Isolation Forest",
    random_forest: "Random Forest (Legacy)",
};

function modelOptionLabel(name, details) {
    if (details.label) return details.label;
    if (details.family === "qmodel") return name.replace(/^qmodel_/i, "Qmodel ").replace("_", " ");
    if (details.family === "ymodel") return name.replace(/^ymodel_/i, "Ymodel ").replace("_", " ");
    if (details.family === "zmodel") return name.replace(/^zmodel_/i, "Zmodel ").replace("_", " ");
    if (details.family === "xmodel") return name.replace(/^xmodel_/i, "Xmodel ").replace("_", " ");
    return LEGACY_MODEL_LABELS[name] || name.replaceAll("_", " ");
}

function lookupModelDetails(models, modelId) {
    if (!models || !modelId) return {};
    if (models[modelId]) return models[modelId];
    const lowered = String(modelId).toLowerCase();
    const match = Object.keys(models).find(key => key.toLowerCase() === lowered);
    return match ? models[match] : {};
}

function formatActiveModelLabel(data) {
    const details = lookupModelDetails(data.available_models, data.active_model);
    return data.active_model_label || modelOptionLabel(data.active_model, details);
}

let canvas, ctx, botCursor;
let challenge = null;
let events = [];
let trail = [];
let isCapturing = false;
let isBotRunning = false;
let sequenceProgress = 0;
let pinProgress = 0;
let targetHits = 0;
let isDrawing = false;
let submitting = false;
let lastCompletedEvents = [];
let canvasRedrawQueued = false;
const NATIVE_CAPTURE_HZ = 1000;
const RECORD_60_HZ = 60;
let clientInteractionStartMs = 0;
let lastRawMove = null;
const rawMoveTimes = [];

function record60HzEnabled() {
    const box = document.getElementById("record-60hz");
    return Boolean(box && box.checked);
}

function enhancedSensorModeEnabled() {
    const box = document.getElementById("enhanced-sensor-mode");
    return Boolean(box && box.checked);
}


function captureTargetHz() {
    return record60HzEnabled() ? RECORD_60_HZ : NATIVE_CAPTURE_HZ;
}

function captureMinMoveDtMs() {
    return 1000 / captureTargetHz();
}

function downsampleEventsToHz(source, hz) {
    if (!source || !source.length || !hz || hz <= 0) return source.slice();
    const ordered = source.slice().sort((a, b) => Number(a.t) - Number(b.t));
    if (ordered.length < 3) return ordered;
    const period = 1000 / hz;
    const out = [];
    let lastKeepT = null;
    const lastIdx = ordered.length - 1;
    for (let i = 0; i < ordered.length; i++) {
        const event = ordered[i];
        const t = Number(event.t || 0);
        const key = String(event.type || "mousemove").toLowerCase().replace(/_/g, "");
        const keepClick = key === "mousedown" || key === "mouseup" || key === "mouseclickdown" || key === "mouseclickup" || key === "1" || key === "2";
        if (i === 0 || i === lastIdx || keepClick) {
            out.push(event);
            lastKeepT = t;
            continue;
        }
        if (lastKeepT == null || (t - lastKeepT) >= period * 0.92) {
            out.push(event);
            lastKeepT = t;
        }
    }
    return out;
}

function eventsForCaptureMode(source) {
    return record60HzEnabled() ? downsampleEventsToHz(source, RECORD_60_HZ) : source.slice();
}

document.addEventListener("DOMContentLoaded", () => {
    canvas = document.getElementById("challenge-canvas");
    ctx = canvas.getContext("2d");
    botCursor = document.getElementById("bot-cursor");

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    document.getElementById("btn-new-challenge").addEventListener("click", () => startNewChallenge());
    document.getElementById("btn-next").addEventListener("click", () => { hideOverlay(); startNewChallenge(); });
    document.getElementById("btn-save-human").addEventListener("click", saveHumanCapture);
    document.getElementById("btn-submit").addEventListener("click", () => submitAttempt("human"));
    document.getElementById("active-model").addEventListener("change", activateModel);
    document.getElementById("btn-run-bot").addEventListener("click", () => {
        const botName = document.getElementById("dashboard-bot").value;
        if (botName) runBot(botName);
    });
    document.getElementById("import-model-file").addEventListener("change", importJoblibModel);
    document.getElementById("dashboard-bot").addEventListener("change", () => {
        localStorage.setItem("dashboardBot", document.getElementById("dashboard-bot").value);
    });
    const hzBox = document.getElementById("record-60hz");
    if (hzBox) {
        hzBox.checked = localStorage.getItem("dashboardRecord60Hz") === "1";
        hzBox.addEventListener("change", () => {
            localStorage.setItem("dashboardRecord60Hz", hzBox.checked ? "1" : "0");
            updateHzReadout();
        });
    }
    const esmBox = document.getElementById("enhanced-sensor-mode");
    if (esmBox) {
        esmBox.checked = localStorage.getItem("dashboardEnhancedSensor") === "1";
        esmBox.addEventListener("change", () => {
            localStorage.setItem("dashboardEnhancedSensor", esmBox.checked ? "1" : "0");
        });
    }
    canvas.style.touchAction = "none";
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove, { passive: true });
    canvas.addEventListener("pointerrawupdate", onPointerRawUpdate, { passive: true });
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);

    loadModelSelector();
    loadBotSelector();
    loadChallengeCatalog();
    checkHealth();
    startNewChallenge();
});

function usesDragStroke() {
    return challenge && DRAG_STROKE_TYPES.has(challenge.challenge_type);
}

function beginDragStroke(p, t, meta) {
    isDrawing = true;
    events = [];
    trail = [];
    lastRawMove = null;
    rawMoveTimes.length = 0;
    recordEvent(p.x, p.y, t, "mousedown", meta);
    drawChallenge();
}

function nearPathGatesStart(p) {
    if (!challenge || challenge.challenge_type !== "path_gates") return true;
    const start = challenge.spec.start;
    const tol = challenge.spec.start_tolerance_px || 32;
    return Math.hypot(p.x - start[0], p.y - start[1]) <= tol;
}

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

function round2(v) {
    return Math.round(v * 100) / 100;
}

function noteRawMoveHz(t) {
    rawMoveTimes.push(t);
    const cutoff = t - 280;
    while (rawMoveTimes.length > 2 && rawMoveTimes[0] < cutoff) {
        rawMoveTimes.shift();
    }
}

function currentCursorHz() {
    if (rawMoveTimes.length < 4) return 0;
    const span = rawMoveTimes[rawMoveTimes.length - 1] - rawMoveTimes[0];
    if (span < 24) return 0;
    return (rawMoveTimes.length - 1) * 1000 / span;
}

function updateHzReadout() {
    const hzEl = document.getElementById("meta-cursor-hz");
    const streamEl = document.getElementById("meta-stream-hz");
    if (!hzEl || !streamEl) return;
    const hz = currentCursorHz();
    hzEl.textContent = hz > 0 ? `${Math.round(hz)}` : "—";
    if (hz > 0) {
        streamEl.textContent = `${Math.min(captureTargetHz(), Math.round(hz))}`;
    } else {
        streamEl.textContent = "—";
    }
}

function emitRecorded(evt, interpolated) {
    if (!interpolated && evt.type === "mousemove") {
        for (let i = events.length - 1; i >= 0; i--) {
            if (events[i].type !== "mousemove") continue;
            if (evt.t - events[i].t < captureMinMoveDtMs()) return;
            break;
        }
        if (events.length) {
            const prev = events[events.length - 1];
            if (prev.type === "mousemove" && prev.x === evt.x && prev.y === evt.y && prev.t === evt.t) {
                return;
            }
        }
    }
    events.push(evt);
    if (!interpolated && evt.type === "mousemove" && (events.length & 1) === 0) {
        trail.push({ x: evt.x, y: evt.y, isBot: false });
        if (trail.length > 800) trail.shift();
    }
    scheduleCanvasRedraw();
}

function recordEvent(x, y, t, type, pointerMeta = null) {
    const evt = {
        x: round2(x),
        y: round2(y),
        t: round2(t),
        type,
    };
    if (pointerMeta) {
        if (pointerMeta.pid != null) evt.pid = pointerMeta.pid;
        if (pointerMeta.pt) evt.pt = pointerMeta.pt;
        if (pointerMeta.src) evt.src = pointerMeta.src;
    }
    if (type === "mousemove") {
        noteRawMoveHz(evt.t);
        lastRawMove = evt;
    } else {
        lastRawMove = evt;
    }
    emitRecorded(evt, false);
}

function pointerMetaFrom(evt) {
    return {
        pid: evt.pointerId,
        pt: evt.pointerType || "mouse",
        src: "canvas",
    };
}

function scheduleCanvasRedraw() {
    if (canvasRedrawQueued) return;
    canvasRedrawQueued = true;
    requestAnimationFrame(() => {
        canvasRedrawQueued = false;
        const meta = document.getElementById("meta-events");
        if (meta) meta.textContent = String(events.length);
        updateHzReadout();
        drawChallenge();
    });
}

function resetLocalCapture() {
    events = [];
}

async function sealedVerifyBody(participant) {
    if (!challenge || !challenge.telemetry_key) {
        throw new Error("Missing local capture key. Start a new challenge.");
    }
    const sealed = await sealTelemetryEvents(challenge.telemetry_key, events);
    persistSealedLocally(challenge.session_id, sealed);
    return {
        session_id: challenge.session_id,
        participant,
        verify_token: challenge.verify_token,
        client_started_ms: clientInteractionStartMs,
        client_submitted_ms: performance.now(),
        source: "dashboard",
        sealed_telemetry: sealed,
        record_hz: record60HzEnabled() ? RECORD_60_HZ : null,
        enhanced_sensor_mode: enhancedSensorModeEnabled(),
    };
}

function eventTime(evt) {
    if (evt && typeof evt.timeStamp === "number" && evt.timeStamp > 0) return evt.timeStamp;
    return performance.now();
}

function onPointerDown(evt) {
    if (!challenge || !isCapturing || isBotRunning) return;
    const p = canvasPoint(evt);
    const meta = pointerMetaFrom(evt);
    if (usesDragStroke()) {
        if (!nearPathGatesStart(p)) return;
        canvas.setPointerCapture(evt.pointerId);
        beginDragStroke(p, eventTime(evt), meta);
        return;
    }
    canvas.setPointerCapture(evt.pointerId);
    recordEvent(p.x, p.y, eventTime(evt), "mousedown", meta);
}

function recordPointerSamples(evt) {
    if (usesDragStroke() && !isDrawing) return;
    const points = typeof evt.getCoalescedEvents === "function" ? evt.getCoalescedEvents() : [evt];
    for (let i = 0; i < points.length; i++) {
        const e = points[i];
        const p = canvasPoint(e);
        recordEvent(p.x, p.y, eventTime(e), "mousemove", pointerMetaFrom(e));
    }
}

function onPointerRawUpdate(evt) {
    if (!challenge || !isCapturing || isBotRunning) return;
    if (usesDragStroke() && !isDrawing) return;
    recordPointerSamples(evt);
}

function onPointerMove(evt) {
    if (!challenge || !isCapturing || isBotRunning) return;
    if (usesDragStroke() && !isDrawing) return;
    if (!("onpointerrawupdate" in window)) {
        recordPointerSamples(evt);
    }
    if (!usesDragStroke()) {
        maybeAutoComplete();
    }
}

function onPointerUp(evt) {
    if (!challenge || !isCapturing || isBotRunning) return;
    try {
        canvas.releasePointerCapture(evt.pointerId);
    } catch (err) {
        /* already released */
    }
    if (usesDragStroke()) {
        if (!isDrawing) return;
        const p = canvasPoint(evt);
        const meta = pointerMetaFrom(evt);
        recordEvent(p.x, p.y, eventTime(evt), "mouseup", meta);
        isDrawing = false;
        drawChallenge();
        if (events.length >= 4) submitAttempt("human");
        return;
    }
    const p = canvasPoint(evt);
    recordEvent(p.x, p.y, eventTime(evt), "mouseup", pointerMetaFrom(evt));
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

    if (type === "target_hunt") {
        const tol = spec.click_tolerance_px;
        for (const target of spec.targets) {
            if (target.clicked) continue;
            if (Math.hypot(x - target.x, y - target.y) <= tol + target.radius) {
                target.clicked = true;
                targetHits += 1;
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
    if (type === "target_hunt" && targetHits >= spec.targets.length) submitAttempt("human");
}

function checkHealth() {
    const pill = document.getElementById("backend-status");
    fetch(`${API_BASE}/api/health`)
        .then(r => r.json())
        .then(data => {
            pill.textContent = `Online · ${formatActiveModelLabel(data) || data.active_model || "ready"}`;
            pill.className = "status-pill ok";
        })
        .catch(() => {
            pill.textContent = "Server offline";
            pill.className = "status-pill";
        });
}

function loadModelSelector() {
    updateModelSelector({
        active_model: "random_forest_challenge",
        available_models: {
            random_forest_challenge: { available: true },
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
    const models = data.available_models || {};
    const qmodelIds = Object.entries(models)
        .filter(([, d]) => d.family === "qmodel" && d.available)
        .sort((a, b) => (a[1].version || "").localeCompare(b[1].version || "", undefined, { numeric: true }));
    const ymodelIds = Object.entries(models)
        .filter(([, d]) => d.family === "ymodel" && d.available)
        .sort((a, b) => (a[1].version || "").localeCompare(b[1].version || "", undefined, { numeric: true }));
    const zmodelIds = Object.entries(models)
        .filter(([, d]) => d.family === "zmodel" && d.available)
        .sort((a, b) => (a[1].version || "").localeCompare(b[1].version || "", undefined, { numeric: true }));
    const xmodelIds = Object.entries(models)
        .filter(([, d]) => d.family === "xmodel" && d.available)
        .sort((a, b) => (a[1].version || "").localeCompare(b[1].version || "", undefined, { numeric: true }));
    const legacyIds = Object.entries(models)
        .filter(([, d]) => d.family === "legacy" && d.available);

    selector.innerHTML = "";

    if (xmodelIds.length) {
        const group = document.createElement("optgroup");
        group.label = "Xmodel versions";
        xmodelIds.forEach(([name, details]) => {
            const opt = document.createElement("option");
            opt.value = name;
            const status = details.status ? ` · ${details.status}` : "";
            opt.textContent = `${modelOptionLabel(name, details)}${status}`;
            opt.selected = name.toLowerCase() === String(data.active_model || "").toLowerCase();
            group.appendChild(opt);
        });
        selector.appendChild(group);
    }

    if (qmodelIds.length) {
        const group = document.createElement("optgroup");
        group.label = "Qmodel versions";
        qmodelIds.forEach(([name, details]) => {
            const opt = document.createElement("option");
            opt.value = name;
            const status = details.status ? ` · ${details.status}` : "";
            opt.textContent = `${modelOptionLabel(name, details)}${status}`;
            opt.selected = name.toLowerCase() === String(data.active_model || "").toLowerCase();
            group.appendChild(opt);
        });
        selector.appendChild(group);
    }

    if (ymodelIds.length) {
        const group = document.createElement("optgroup");
        group.label = "Ymodel versions";
        ymodelIds.forEach(([name, details]) => {
            const opt = document.createElement("option");
            opt.value = name;
            const status = details.status ? ` · ${details.status}` : "";
            opt.textContent = `${modelOptionLabel(name, details)}${status}`;
            opt.selected = name.toLowerCase() === String(data.active_model || "").toLowerCase();
            group.appendChild(opt);
        });
        selector.appendChild(group);
    }

    if (zmodelIds.length) {
        const group = document.createElement("optgroup");
        group.label = "Zmodel versions";
        zmodelIds.forEach(([name, details]) => {
            const opt = document.createElement("option");
            opt.value = name;
            const status = details.status ? ` · ${details.status}` : "";
            opt.textContent = `${modelOptionLabel(name, details)}${status}`;
            opt.selected = name.toLowerCase() === String(data.active_model || "").toLowerCase();
            group.appendChild(opt);
        });
        selector.appendChild(group);
    }

    if (legacyIds.length) {
        const group = document.createElement("optgroup");
        group.label = "Legacy models";
        legacyIds.forEach(([name, details]) => {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = modelOptionLabel(name, details);
            opt.selected = name.toLowerCase() === String(data.active_model || "").toLowerCase();
            group.appendChild(opt);
        });
        selector.appendChild(group);
    }

    const importedIds = Object.entries(models)
        .filter(([, d]) => d.family === "imported" && d.available);
    if (importedIds.length) {
        const group = document.createElement("optgroup");
        group.label = "Imported models";
        importedIds.forEach(([name, details]) => {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = modelOptionLabel(name, details);
            opt.selected = name.toLowerCase() === String(data.active_model || "").toLowerCase();
            group.appendChild(opt);
        });
        selector.appendChild(group);
    }

    if (!selector.options.length) return;
    document.getElementById("model-line").textContent = `Model: ${formatActiveModelLabel(data)}`;
}

function activateModel() {
    const selector = document.getElementById("active-model");
    selector.disabled = true;
    fetch(`${API_BASE}/api/models/activate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: selector.value }),
    })
        .then(async r => {
            const data = await r.json();
            if (!r.ok) {
                const message = data.detail?.message || data.detail || "Could not activate model.";
                throw new Error(typeof message === "string" ? message : "Could not activate model.");
            }
            updateModelSelector(data);
            document.getElementById("model-line").textContent = `Model: ${formatActiveModelLabel(data)}`;
        })
        .catch(err => {
            console.error(err);
            alert(err.message || "Could not activate model.");
        })
        .finally(() => { selector.disabled = false; });
}

function importJoblibModel(evt) {
    const input = evt.target;
    const file = input.files && input.files[0];
    input.value = "";
    if (!file) return;
    const body = new FormData();
    body.append("file", file, file.name);
    const importBtn = document.querySelector(".import-model-btn");
    if (importBtn) importBtn.setAttribute("aria-busy", "true");
    fetch(`${API_BASE}/api/models/import`, { method: "POST", body })
        .then(async r => {
            const data = await r.json();
            if (!r.ok) {
                const message = data.detail?.message || data.detail || "Import failed.";
                throw new Error(typeof message === "string" ? message : "Import failed.");
            }
            updateModelSelector(data);
            document.getElementById("model-line").textContent = `Model: ${formatActiveModelLabel(data)}`;
        })
        .catch(err => {
            console.error(err);
            alert(err.message || "Could not import joblib.");
        })
        .finally(() => {
            if (importBtn) importBtn.removeAttribute("aria-busy");
        });
}

function loadBotSelector() {
    fetch(`${API_BASE}/api/challenge/bots`)
        .then(r => r.json())
        .then(data => {
            const selector = document.getElementById("dashboard-bot");
            const bots = data.bots || [];
            const saved = localStorage.getItem("dashboardBot");
            selector.innerHTML = "";
            const groups = [];
            bots.forEach(bot => {
                if (!groups.includes(bot.group)) groups.push(bot.group);
            });
            groups.forEach(groupName => {
                const group = document.createElement("optgroup");
                group.label = groupName;
                bots.filter(b => b.group === groupName).forEach(bot => {
                    const opt = document.createElement("option");
                    opt.value = bot.id;
                    opt.textContent = bot.label;
                    opt.selected = bot.id === saved;
                    group.appendChild(opt);
                });
                selector.appendChild(group);
            });
            if (saved && [...selector.options].some(o => o.value === saved)) {
                selector.value = saved;
            }
        })
        .catch(() => {});
}

function loadChallengeCatalog() {
    fetch(`${API_BASE}/api/challenge/types`)
        .then(r => r.json())
        .then(data => {
            const picker = document.getElementById("challenge-type");
            const catalog = document.getElementById("challenge-catalog");
            const entries = data.catalog || {};
            catalog.innerHTML = "";
            Object.entries(entries).forEach(([id, info]) => {
                const opt = document.createElement("option");
                opt.value = id;
                opt.textContent = info.title || id.replaceAll("_", " ");
                picker.appendChild(opt);

                const li = document.createElement("li");
                li.dataset.type = id;
                li.innerHTML = `<strong>${info.title || id}</strong> · ${info.interaction || ""}<br><span>${info.summary || ""}</span>`;
                catalog.appendChild(li);
            });
        })
        .catch(() => {
            document.getElementById("challenge-catalog").innerHTML =
                "<li class='flag idle'>Could not load catalog.</li>";
        });
}

function highlightCatalogEntry(type) {
    document.querySelectorAll("#challenge-catalog li").forEach(li => {
        li.classList.toggle("active", li.dataset.type === type);
    });
}

function startNewChallenge() {
    isCapturing = false;
    isBotRunning = false;
    isDrawing = false;
    events = [];
    lastCompletedEvents = [];
    trail = [];
    lastRawMove = null;
    rawMoveTimes.length = 0;
    sequenceProgress = 0;
    pinProgress = 0;
    targetHits = 0;
    hideOverlay();
    resetSaveHumanButton();
    setButtonsDisabled(false);

    const picker = document.getElementById("challenge-type");
    const chosen = picker && picker.value !== "random" ? picker.value : null;

    fetch(`${API_BASE}/api/challenge/new`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ challenge_type: chosen }),
    })
        .then(r => r.json())
        .then(data => {
            challenge = data;
            isCapturing = true;
            clientInteractionStartMs = performance.now();
            resetLocalCapture();
            document.getElementById("challenge-title").textContent = data.title;
            document.getElementById("challenge-instructions").textContent = data.instructions;
            document.getElementById("meta-type").textContent = data.challenge_type;
            document.getElementById("meta-seed").textContent = String(data.seed);
            document.getElementById("meta-events").textContent = "0";
            document.getElementById("meta-participant").textContent = "—";
            highlightCatalogEntry(data.challenge_type);
            resetAssessmentUI("IN PROGRESS", data.instructions);
            const submitBtn = document.getElementById("btn-submit");
            submitBtn.style.display = usesDragStroke() ? "none" : "inline-block";
            if (data.spec.buttons) data.spec.buttons.forEach(b => { b.clicked = false; });
            if (data.spec.keys) data.spec.keys.forEach(k => { k.pressed = false; });
            if (data.spec.targets) data.spec.targets.forEach(t => { t.clicked = false; });
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
    ["btn-submit", "btn-run-bot", "dashboard-bot"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = disabled;
    });
}

function submitAttempt(participant) {
    if (!challenge || submitting) return;
    submitting = true;
    isCapturing = false;
    setButtonsDisabled(true);

    sealedVerifyBody(participant).then((body) => {
        return fetch(`${API_BASE}/api/challenge/verify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
    })
        .then(async r => {
            const data = await r.json();
            if (!r.ok) {
                const detail = data.detail;
                const msg = typeof detail === "object" ? detail.message || JSON.stringify(detail) : detail || r.statusText;
                throw new Error(msg);
            }
            return data;
        })
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
    lastRawMove = null;
    rawMoveTimes.length = 0;

    fetch(`${API_BASE}/api/challenge/run-bot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: challenge.session_id,
            bot: botName,
            record_hz: record60HzEnabled() ? RECORD_60_HZ : null,
            enhanced_sensor_mode: enhancedSensorModeEnabled(),
        }),
    })
        .then(r => r.json())
        .then(data => {
            document.getElementById("meta-participant").textContent = data.participant;
            animateBotEvents(eventsForCaptureMode(data.events || []), () => {
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
    if (type === "shape_trace" || type === "zigzag_trace" || type === "spiral_trace") drawGuidePath(challenge.spec);
    if (type === "connect_dots") drawConnectDots(challenge.spec);
    if (type === "target_hunt") drawTargetHunt(challenge.spec);
    if (type === "hazard_run") drawHazardRun(challenge.spec);
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

function drawGuidePath(spec) {
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

function drawConnectDots(spec) {
    const dots = spec.dots || [];
    if (dots.length > 1) {
        ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 8]);
        ctx.beginPath();
        ctx.moveTo(dots[0].x, dots[0].y);
        for (let i = 1; i < dots.length; i++) ctx.lineTo(dots[i].x, dots[i].y);
        ctx.stroke();
        ctx.setLineDash([]);
    }
    dots.forEach(dot => {
        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(34, 211, 238, 0.2)";
        ctx.fill();
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#f8fafc";
        ctx.font = "bold 16px Outfit";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(dot.id), dot.x, dot.y);
    });
}

function drawTargetHunt(spec) {
    spec.targets.forEach(target => {
        ctx.beginPath();
        ctx.arc(target.x, target.y, target.radius, 0, Math.PI * 2);
        ctx.fillStyle = target.clicked ? "rgba(16, 185, 129, 0.35)" : `${target.color}33`;
        ctx.fill();
        ctx.strokeStyle = target.clicked ? "#10b981" : target.color;
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

function drawHazardRun(spec) {
    spec.hazards.forEach(h => {
        ctx.beginPath();
        ctx.arc(h.x, h.y, h.radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(244, 63, 94, 0.25)";
        ctx.fill();
        ctx.strokeStyle = "rgba(244, 63, 94, 0.7)";
        ctx.lineWidth = 2;
        ctx.stroke();
    });
    ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    const wps = spec.waypoints || [];
    if (wps.length) {
        ctx.moveTo(wps[0][0], wps[0][1]);
        for (let i = 1; i < wps.length; i++) ctx.lineTo(wps[i][0], wps[i][1]);
        ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.fillStyle = "#10b981";
    ctx.beginPath(); ctx.arc(spec.start[0], spec.start[1], 10, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath(); ctx.arc(spec.end[0], spec.end[1], 10, 0, Math.PI * 2); ctx.fill();
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
    ctx.beginPath(); ctx.arc(spec.start[0], spec.start[1], 12, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath(); ctx.arc(spec.end[0], spec.end[1], 12, 0, Math.PI * 2); ctx.fill();
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
    document.getElementById("model-line").textContent = `Model: ${evalData.qmodel_label || evalData.model_type || "unknown"}`;
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
    addResultCell(grid, "Model human", humanPct != null ? `${humanPct}%` : "—");
    addResultCell(grid, "Bot prob.", evalData.bot_probability != null ? evalData.bot_probability.toFixed(3) : "—");
    addResultCell(grid, "Heuristic", evalData.heuristic_score != null ? evalData.heuristic_score.toFixed(3) : "—");
    addResultCell(grid, "Fused risk", evalData.risk_score != null ? evalData.risk_score.toFixed(3) : "—");
    addResultCell(grid, "Participant", data.participant || "—");

    document.getElementById("overlay-title").textContent = task.passed ? "Complete" : "Incomplete";
    document.getElementById("overlay-message").textContent = task.reason || "";

    const saveBtn = document.getElementById("btn-save-human");
    const saveStatus = document.getElementById("overlay-save-status");
    const isHumanAttempt = !String(data.participant || "human").toLowerCase().startsWith("bot");
    lastCompletedEvents = events.slice();
    saveStatus.textContent = "";
    if (isHumanAttempt && lastCompletedEvents.length >= 4) {
        saveBtn.classList.remove("hidden");
        saveBtn.disabled = false;
        saveBtn.textContent = "Keep as human data & next";
    } else {
        saveBtn.classList.add("hidden");
        if (!isHumanAttempt) {
            saveStatus.textContent = "Bot runs are not saved as human data.";
        }
    }
    showOverlay();
}

function addResultCell(grid, label, value) {
    const div = document.createElement("div");
    div.innerHTML = `${label}<strong>${value}</strong>`;
    grid.appendChild(div);
}

function showOverlay() {
    document.getElementById("challenge-overlay").classList.remove("hidden");
    if (canvas) canvas.style.pointerEvents = "none";
}

function hideOverlay() {
    document.getElementById("challenge-overlay").classList.add("hidden");
    if (canvas) canvas.style.pointerEvents = "";
}

function resetSaveHumanButton() {
    const btn = document.getElementById("btn-save-human");
    const status = document.getElementById("overlay-save-status");
    btn.disabled = false;
    btn.classList.add("hidden");
    btn.textContent = "Keep as human data & next";
    status.textContent = "";
}

function saveHumanCapture() {
    if (!challenge) return;
    const snapshot = eventsForCaptureMode(lastCompletedEvents.length ? lastCompletedEvents : events);
    const sessionId = challenge.session_id;
    const verifyToken = challenge.verify_token;
    const challengeType = challenge.challenge_type;
    const seed = challenge.seed;
    const btn = document.getElementById("btn-save-human");
    const status = document.getElementById("overlay-save-status");
    btn.disabled = true;
    status.textContent = "Saving capture…";
    fetch(`${API_BASE}/api/challenge/save-human`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: sessionId,
            verify_token: verifyToken,
            events: snapshot,
            challenge_type: challengeType,
            seed,
            spec: challenge.spec || {},
            record_hz: record60HzEnabled() ? RECORD_60_HZ : null,
        }),
    })
        .then(async r => {
            const data = await r.json().catch(() => ({}));
            if (!r.ok) {
                const detail = data.detail;
                throw new Error(typeof detail === "object" ? detail.message || JSON.stringify(detail) : detail || r.statusText);
            }
            return data;
        })
        .then(() => {
            hideOverlay();
            startNewChallenge();
        })
        .catch(err => {
            btn.disabled = false;
            status.textContent = `Could not save: ${err.message || err}`;
        });
}
