const canvas = document.getElementById("arena");
const ctx = canvas.getContext("2d");
const W = 960;
const H = 640;

const els = {
    model: document.getElementById("model-select"),
    watch: document.getElementById("watch-bot"),
    scrollAll: document.getElementById("scroll-all"),
    workerCount: document.getElementById("worker-count"),
    workerHint: document.getElementById("worker-hint"),
    start: document.getElementById("btn-start"),
    pause: document.getElementById("btn-pause"),
    resume: document.getElementById("btn-resume"),
    stop: document.getElementById("btn-stop"),
    trainNow: document.getElementById("btn-train-now"),
    probe: document.getElementById("btn-probe"),
    windowTarget: document.getElementById("window-target"),
    probeScore: document.getElementById("probe-score"),
    probeList: document.getElementById("probe-list"),
    phase: document.getElementById("phase-pill"),
    modelPill: document.getElementById("model-pill"),
    challenge: document.getElementById("challenge-label"),
    bot: document.getElementById("bot-label"),
    workers: document.getElementById("workers-pill"),
    hc: document.getElementById("hc-value"),
    hcSub: document.getElementById("hc-sub"),
    windowCounts: document.getElementById("window-counts"),
    windowBar: document.getElementById("window-bar"),
    iter: document.getElementById("stat-iter"),
    attempts: document.getElementById("stat-attempts"),
    beats: document.getElementById("stat-beats"),
    marks: document.getElementById("stat-marks"),
    families: document.getElementById("family-grid"),
    log: document.getElementById("log"),
    stopReason: document.getElementById("stop-reason"),
    datasetTab: document.getElementById("dataset-tab"),
    datasetDrawer: document.getElementById("dataset-drawer"),
    datasetClose: document.getElementById("dataset-close"),
    datasetList: document.getElementById("dataset-list"),
    testDataset: document.getElementById("btn-test-dataset"),
    trainFails: document.getElementById("btn-train-fails"),
    datasetScore: document.getElementById("dataset-score"),
    datasetResults: document.getElementById("dataset-results"),
};

let lastTrail = [];
let animTrail = [];
let animTimer = null;
let shownAttempt = null;
let lastState = null;
let probeView = null;
let probeHits = [];
let probeTimer = null;
let selectedDatasetId = "";
let cachedDatasets = [];
let lastLeakCount = 0;
let trainFailsBusy = false;
const PROBE_HOLD_MS = 18000;

function watchingBot() {
    return els.watch && els.watch.value ? els.watch.value : "";
}

function scrollAllEnabled() {
    return Boolean(els.scrollAll && els.scrollAll.checked);
}

function shouldShowBot(botId) {
    if (scrollAllEnabled()) return true;
    const watch = watchingBot();
    return !watch || watch === botId;
}

function syncWatchControls() {
    if (els.watch) els.watch.disabled = scrollAllEnabled();
}

function api(path, method = "GET", body) {
    return fetch(path, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
    }).then(async (r) => {
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.detail || r.statusText);
        return data;
    });
}

function drawGrid() {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "rgba(34,211,238,0.06)";
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
}

function drawTrail(points, cursor) {
    drawGrid();
    if (!points.length) return;
    ctx.beginPath();
    ctx.strokeStyle = "rgba(34,211,238,0.85)";
    ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(34,211,238,0.6)";
    ctx.shadowBlur = 12;
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    ctx.stroke();
    ctx.shadowBlur = 0;
    const tip = cursor || points[points.length - 1];
    ctx.fillStyle = "#fb7185";
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, 7, 0, Math.PI * 2);
    ctx.fill();
}

function playTrail(points) {
    lastTrail = points || [];
    animTrail = [];
    if (animTimer) clearInterval(animTimer);
    if (!lastTrail.length) {
        drawGrid();
        return;
    }
    let i = 0;
    animTimer = setInterval(() => {
        if (i >= lastTrail.length) {
            clearInterval(animTimer);
            drawTrail(lastTrail);
            return;
        }
        animTrail.push(lastTrail[i]);
        drawTrail(animTrail);
        i += Math.max(1, Math.ceil(lastTrail.length / 40));
    }, 16);
}

function renderFamilies(families, activeBot) {
    els.families.innerHTML = "";
    (families || []).forEach((fam) => {
        const div = document.createElement("div");
        const pct = Math.round((fam.best_hc || 0) * 100);
        div.className = "fam";
        if (fam.id === activeBot) div.classList.add("active");
        if (!scrollAllEnabled() && fam.id === watchingBot()) div.classList.add("active");
        if (fam.phase === "expanding") div.classList.add("expanding");
        if ((fam.best_hc || 0) >= 0.65) div.classList.add("win");
        else if ((fam.best_hc || 0) >= 0.45) div.classList.add("hot");
        const extra = fam.phase === "expanding" ? " · expanding" : "";
        const w = fam.workers ? ` · ${fam.workers}w` : "";
        div.innerHTML = `<b>${fam.label}</b>${pct}% · ${fam.wins || 0} wins · ${fam.attempts || 0}${w}${extra}`;
        div.title = "Click to watch this family";
        div.onclick = () => {
            els.scrollAll.checked = false;
            els.watch.value = fam.id;
            syncWatchControls();
            renderFamilies(families, activeBot);
        };
        els.families.appendChild(div);
    });
}

function showAttempt(attempt, trail) {
    if (probeView) return;
    if (!attempt || !shouldShowBot(attempt.bot)) return;
    shownAttempt = attempt;
    els.challenge.textContent = attempt.challenge_type || "";
    els.bot.textContent = attempt.bot || "";
    const pct = Math.round((attempt.human_confidence || 0) * 100);
    els.hc.textContent = `${pct}%`;
    els.hc.className = "gauge-value " + (pct >= 65 ? "hot" : "cold");
    els.hcSub.textContent = attempt.win ? "EVADED" : attempt.mark ? "marked >=60%" : "blocked";
    if (trail) playTrail(trail);
}

function applyState(state) {
    if (!state) return;
    lastState = state;
    const running = Boolean(state.running);
    els.phase.textContent = state.phase || "idle";
    els.phase.className = "pill" + (state.phase === "stopped" ? " miss" : state.running ? "" : " dim");
    const shownModel = (!running && state.selected_model_id) ? state.selected_model_id : (state.model_id || "no model");
    els.modelPill.textContent = shownModel;
    if (els.workers) {
        const n = state.workers || 0;
        const per = state.workers_per_family || 1;
        els.workers.textContent = n ? `${n} workers · ~${per}/family` : "0 workers";
        els.workers.className = "pill" + (n > 1 ? "" : " dim");
    }
    els.iter.textContent = state.iteration || 1;
    els.attempts.textContent = state.attempts || 0;
    els.beats.textContent = state.beats || 0;
    els.marks.textContent = state.marks || 0;
    const target = state.windows_target || 5000;
    const cycle = state.windows_cycle || 0;
    els.windowCounts.textContent = `${cycle} / ${target}`;
    els.windowBar.style.width = `${Math.min(100, (cycle / target) * 100)}%`;
    if (els.windowTarget && document.activeElement !== els.windowTarget) {
        els.windowTarget.value = String(target);
    }
    const last = state.last_attempt;
    if (last && shouldShowBot(last.bot)) showAttempt(last);
    renderFamilies(state.families, last && last.bot);
    els.stopReason.textContent = state.stop_reason || "";
    if (state.log_tail && state.log_tail.length) {
        els.log.textContent = state.log_tail.slice(-40).join("\n");
        els.log.scrollTop = els.log.scrollHeight;
    }
    els.start.disabled = running;
    els.pause.disabled = !running || state.paused;
    els.resume.disabled = !running || !state.paused;
    els.stop.disabled = !running;
    els.model.disabled = false;
    const retraining = state.phase === "retraining" || trainFailsBusy;
    if (els.trainNow) els.trainNow.disabled = retraining;
    if (els.probe) els.probe.disabled = retraining;
    if (els.testDataset) els.testDataset.disabled = retraining;
    if (els.trainFails) {
        const leaks = (state.dataset_probe && Array.isArray(state.dataset_probe.leaks))
            ? state.dataset_probe.leaks.length
            : 0;
        if (leaks && !trainFailsBusy) lastLeakCount = Number(state.dataset_probe.passed) || leaks;
        els.trainFails.disabled = retraining || lastLeakCount < 1;
    }
    syncWatchControls();
}

async function boot() {
    drawGrid();
    const data = await api("/api/arena/bootstrap");
    const versions = (data.xmodels && data.xmodels.versions) || [];
    const latest = data.xmodels && data.xmodels.latest;
    fillModels(versions, latest, latest);
    els.watch.innerHTML = "";
    (data.bots || []).forEach((bot) => {
        const opt = document.createElement("option");
        opt.value = bot.id;
        opt.textContent = `${bot.label} (${bot.group})`;
        els.watch.appendChild(opt);
    });
    if (data.bots && data.bots.length) els.watch.value = data.bots[0].id;
    if (els.workerCount) {
        const max = data.workers_max || 128;
        els.workerCount.max = String(max);
        const current = (data.state && data.state.workers) || data.workers_default || 32;
        els.workerCount.value = String(current);
    }
    if (els.workerHint) {
        const rec = data.workers_default || 32;
        const safe = data.workers_safe || 64;
        const max = data.workers_max || 128;
        els.workerHint.textContent = `${rec} recommended · ${safe} safe on a 3060 12GB · ${max} hard cap`;
    }
    syncWatchControls();
    renderProbe(null);
    applyState(data.state);
    let datasets = data.datasets || [];
    if (!datasets.length) {
        try {
            const extra = await api("/api/arena/datasets");
            datasets = extra.datasets || [];
        } catch (_err) {
            datasets = [];
        }
    }
    fillDatasets(datasets);
    setDrawerOpen(true);
    if (!data.state || !data.state.running) {
        await pushSelectedModel(true);
    }
}

async function refreshModels(preferred) {
    const data = await api("/api/arena/bootstrap");
    const versions = (data.xmodels && data.xmodels.versions) || [];
    const latest = data.xmodels && data.xmodels.latest;
    fillModels(versions, latest, preferred || latest);
}

function fillModels(versions, latest, preferred) {
    const rank = (v) => (v.id === latest || v.status === "latest") ? 0 : v.status === "available" ? 1 : 2;
    const sorted = [...(versions || [])].sort((a, b) => rank(a) - rank(b) || String(a.id).localeCompare(String(b.id)));
    els.model.innerHTML = "";
    sorted.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = `${v.label} · ${v.status}`;
        els.model.appendChild(opt);
    });
    const pick = preferred || latest || (sorted[0] && sorted[0].id);
    if (pick) els.model.value = pick;
}

function setDrawerOpen(open) {
    if (els.datasetDrawer) els.datasetDrawer.classList.toggle("open", open);
    document.body.classList.toggle("drawer-open", open);
    if (els.datasetTab) els.datasetTab.setAttribute("aria-expanded", open ? "true" : "false");
}

function fillDatasets(datasets) {
    if (!els.datasetList) return;
    if (datasets && datasets.length) cachedDatasets = datasets;
    const rows = cachedDatasets;
    els.datasetList.innerHTML = "";
    if (!rows.length) {
        els.datasetList.innerHTML = "<p class=\"probe-score\">No loop-winner datasets found.</p>";
        return;
    }
    if (!selectedDatasetId) {
        const featured = rows.find((d) => d.featured) || rows[0];
        selectedDatasetId = featured ? featured.id : "";
    }
    rows.forEach((ds) => {
        const div = document.createElement("button");
        div.type = "button";
        div.className = "dataset-card" + (ds.featured ? " featured" : "") + (ds.id === selectedDatasetId ? " active" : "");
        div.innerHTML = `<b>${ds.label}</b><span>${ds.detail || ""}</span>`;
        div.onclick = () => {
            selectedDatasetId = ds.id;
            fillDatasets();
        };
        els.datasetList.appendChild(div);
    });
}

function renderDatasetResults(probe) {
    if (!els.datasetScore || !els.datasetResults) return;
    if (!probe) {
        lastLeakCount = 0;
        if (els.trainFails && !trainFailsBusy) els.trainFails.disabled = true;
        els.datasetScore.textContent = "Select a dataset, then test it on the selected model.";
        els.datasetResults.innerHTML = "";
        return;
    }
    lastLeakCount = Number(probe.passed) || 0;
    if (els.trainFails && !trainFailsBusy) els.trainFails.disabled = lastLeakCount < 1;
    els.datasetScore.textContent =
        `${probe.passed || 0} still beat · ${probe.failed || 0} caught  (${probe.tested || 0} windows · ${probe.model_id || ""})`;
    els.datasetResults.innerHTML = "";
    const fails = (probe.episodes || []).filter((row) => row.error || !row.passed);
    const show = fails.length ? fails.slice(0, 40) : (probe.episodes || []).slice(0, 12);
    show.forEach((row) => {
        const div = document.createElement("div");
        div.className = "probe-row " + (row.error || !row.passed ? "fail" : "pass");
        div.innerHTML =
            `<b>${row.bot_id || "?"} · ${row.challenge_type || ""}</b>` +
            `${row.windows_passed || 0} beat / ${row.windows_failed || 0} caught` +
            `${row.error ? " · " + row.error : ""}`;
        els.datasetResults.appendChild(div);
    });
    if (fails.length > 40) {
        const more = document.createElement("p");
        more.className = "probe-score";
        more.textContent = `Showing 40 of ${fails.length} leaking / error episodes`;
        els.datasetResults.appendChild(more);
    }
}

async function pushSelectedModel(silent) {
    const modelId = els.model && els.model.value;
    if (!modelId) return;
    try {
        const out = await api("/api/arena/select-model", "POST", { model_id: modelId });
        applyState(out.state);
        if (!silent && els.stopReason) els.stopReason.textContent = `Using ${modelId}`;
    } catch (err) {
        if (els.stopReason) els.stopReason.textContent = err.message;
    }
}

function clearProbeView(restore) {
    probeView = null;
    probeHits = [];
    if (probeTimer) {
        clearTimeout(probeTimer);
        probeTimer = null;
    }
    if (restore) {
        if (lastTrail.length) drawTrail(lastTrail);
        else drawGrid();
    }
}

function drawDatasetHeatmap(probe) {
    drawGrid();
    ctx.save();
    ctx.fillStyle = "rgba(5, 7, 14, 0.86)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#f8fafc";
    ctx.font = "700 22px Outfit, sans-serif";
    ctx.fillText(probe.dataset_label || "Full dataset", 24, 34);
    ctx.font = "500 14px Outfit, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(
        `${probe.passed || 0} still beat · ${probe.failed || 0} caught · ${probe.tested || 0} windows vs ${probe.model_id || ""}`,
        24,
        56
    );
    const cells = [];
    (probe.episodes || []).forEach((ep) => {
        (ep.windows || []).forEach((win) => {
            cells.push({ window: win, episode: ep });
        });
        if (!(ep.windows || []).length) {
            cells.push({
                window: { passed: false, human_confidence: 0, index: 0 },
                episode: ep,
            });
        }
    });
    probeHits = [];
    const padX = 24;
    const padY = 78;
    const gap = 3;
    const count = Math.max(1, cells.length);
    const cols = Math.max(1, Math.ceil(Math.sqrt(count * (W / H))));
    const rows = Math.max(1, Math.ceil(count / cols));
    const cellW = Math.max(8, Math.floor((W - padX * 2 - gap * (cols - 1)) / cols));
    const cellH = Math.max(8, Math.floor((H - padY - 24 - gap * (rows - 1)) / rows));
    cells.forEach((cell, i) => {
        const cx = padX + (i % cols) * (cellW + gap);
        const cy = padY + Math.floor(i / cols) * (cellH + gap);
        ctx.fillStyle = cell.window.passed ? "rgba(52,211,153,0.85)" : "rgba(251,113,133,0.85)";
        ctx.fillRect(cx, cy, cellW, cellH);
        probeHits.push({ x: cx, y: cy, w: cellW, h: cellH, window: cell.window, episode: cell.episode });
    });
    ctx.restore();
}

function drawProbeCanvas(probe) {
    const episodes = probe.episodes || [];
    const windowCount = Number(probe.tested || 0);
    if (episodes.length > 3 || windowCount > 24) {
        drawDatasetHeatmap(probe);
        return;
    }
    drawGrid();
    ctx.save();
    ctx.fillStyle = "rgba(5, 7, 14, 0.82)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#f8fafc";
    ctx.font = "700 22px Outfit, sans-serif";
    ctx.fillText(`Last 2 datasets vs ${probe.model_id || ""}`, 24, 34);
    ctx.font = "500 14px Outfit, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.fillText(
        `${probe.passed || 0} windows still beat · ${probe.failed || 0} caught · click a window · live view in ${Math.round(PROBE_HOLD_MS / 1000)}s`,
        24,
        56
    );
    probeHits = [];
    const colW = (W - 36) / Math.max(1, episodes.length);
    episodes.forEach((ep, ei) => {
        const x0 = 18 + ei * colW;
        const y0 = 78;
        ctx.fillStyle = ep.error ? "rgba(251,113,133,0.16)" : "rgba(255,255,255,0.04)";
        ctx.fillRect(x0, y0, colW - 12, H - y0 - 18);
        ctx.strokeStyle = ep.passed ? "rgba(52,211,153,0.45)" : "rgba(251,113,133,0.4)";
        ctx.strokeRect(x0, y0, colW - 12, H - y0 - 18);
        ctx.fillStyle = "#e2e8f0";
        ctx.font = "700 14px Outfit, sans-serif";
        const title = `${ep.bot_id || "?"} · ${ep.challenge_type || ""}`;
        ctx.fillText(title.slice(0, 42), x0 + 10, y0 + 22);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "500 11px JetBrains Mono, monospace";
        ctx.fillText(
            ep.error
                ? String(ep.error).slice(0, 48)
                : `${ep.windows_passed || 0} beat / ${ep.windows_failed || 0} caught · was ${ep.original_hc == null ? "—" : Math.round(ep.original_hc * 100) + "%"}`,
            x0 + 10,
            y0 + 40
        );
        const windows = ep.windows || [];
        const cellW = 54;
        const cellH = 36;
        const innerW = colW - 32;
        const cols = Math.max(1, Math.floor(innerW / (cellW + 6)));
        windows.forEach((win, wi) => {
            const cx = x0 + 10 + (wi % cols) * (cellW + 6);
            const cy = y0 + 54 + Math.floor(wi / cols) * (cellH + 6);
            ctx.fillStyle = win.passed ? "rgba(52,211,153,0.22)" : "rgba(251,113,133,0.22)";
            ctx.fillRect(cx, cy, cellW, cellH);
            ctx.strokeStyle = win.passed ? "rgba(52,211,153,0.7)" : "rgba(251,113,133,0.7)";
            ctx.strokeRect(cx, cy, cellW, cellH);
            ctx.fillStyle = win.passed ? "#34d399" : "#fb7185";
            ctx.font = "700 12px JetBrains Mono, monospace";
            ctx.fillText(`W${wi + 1}`, cx + 6, cy + 14);
            ctx.fillText(`${Math.round((win.human_confidence || 0) * 100)}%`, cx + 6, cy + 28);
            probeHits.push({ x: cx, y: cy, w: cellW, h: cellH, window: win, episode: ep });
        });
    });
    ctx.restore();
}

function showProbeOnCanvas(probe) {
    probeView = probe;
    drawProbeCanvas(probe);
    els.hc.textContent = `${probe.passed || 0}/${probe.tested || 0}`;
    els.hc.className = "gauge-value " + ((probe.failed || 0) === 0 ? "hot" : "cold");
    els.hcSub.textContent = "windows still beating";
    els.challenge.textContent = "probe";
    els.bot.textContent = probe.model_id || "";
    if (probeTimer) clearTimeout(probeTimer);
    const hold = (probe.tested || 0) > 24 ? 30000 : PROBE_HOLD_MS;
    probeTimer = setTimeout(() => clearProbeView(true), hold);
}

function renderProbe(probe) {
    if (!els.probeScore || !els.probeList) return;
    if (!probe || !(probe.tested || (probe.episodes && probe.episodes.length))) {
        els.probeScore.textContent = "Not run yet";
        els.probeList.innerHTML = "";
        return;
    }
    els.probeScore.textContent =
        `${probe.passed || 0} windows still beat · ${probe.failed || 0} caught  (${probe.tested || 0} windows · ${probe.model_id || ""})`;
    els.probeList.innerHTML = "";
    (probe.episodes || []).forEach((row) => {
        const div = document.createElement("div");
        const ok = Boolean(row.passed);
        div.className = "probe-row " + (row.error ? "fail" : ok ? "pass" : "fail");
        const orig = row.original_hc == null ? "—" : `${Math.round(row.original_hc * 100)}%`;
        const status = row.error ? "error" : ok ? "HAS BEATING WINDOWS" : "ALL CAUGHT";
        const chips = (row.windows || []).map((win, idx) => {
            const cls = win.passed ? "pass" : "fail";
            const hc = Math.round((win.human_confidence || 0) * 100);
            return `<span class="probe-win ${cls}">W${idx + 1} ${hc}%</span>`;
        }).join("");
        div.innerHTML =
            `<b>${status} · ${row.bot_id || "?"} · ${row.challenge_type || ""}</b>` +
            `${row.filename || ""}<br>${row.windows_passed || 0} beat / ${row.windows_failed || 0} caught (was ${orig})` +
            `${row.error ? " · " + row.error : ""}` +
            `<div class="probe-windows">${chips}</div>`;
        els.probeList.appendChild(div);
    });
}

function readWindowTarget() {
    const n = Number(els.windowTarget && els.windowTarget.value);
    return Number.isFinite(n) ? n : 5000;
}

function readWorkerCount() {
    const n = Number(els.workerCount && els.workerCount.value);
    return Number.isFinite(n) ? n : 32;
}

let workerTimer = null;
function pushWorkerCount() {
    const n = readWorkerCount();
    api("/api/arena/workers", "POST", { workers: n }).catch((err) => {
        if (els.stopReason) els.stopReason.textContent = err.message;
    });
}

els.start.onclick = async () => {
    const cycle = (lastState && lastState.windows_cycle) || 0;
    if (cycle > 0) {
        const ok = window.confirm(
            `Start loop resets the current harvest (${cycle} winning windows, ${lastState.attempts || 0} attempts). Use Resume if you just reloaded. Continue anyway?`
        );
        if (!ok) return;
    }
    els.start.disabled = true;
    try {
        const out = await api("/api/arena/start", "POST", {
            model_id: els.model.value,
            workers: readWorkerCount(),
        });
        applyState(out.state);
    } catch (err) {
        els.stopReason.textContent = err.message;
        els.start.disabled = false;
    }
};
els.pause.onclick = () => api("/api/arena/pause", "POST");
els.resume.onclick = () => api("/api/arena/resume", "POST");
els.stop.onclick = () => api("/api/arena/stop", "POST");
if (els.trainNow) {
    els.trainNow.onclick = async () => {
        els.trainNow.disabled = true;
        els.stopReason.textContent = "Training on current winning windows…";
        try {
            const out = await api("/api/arena/train-now", "POST", { model_id: els.model.value });
            applyState(out.state);
            await refreshModels(out.model_id);
            renderProbe(out.probe);
            if (out.probe) showProbeOnCanvas(out.probe);
            els.stopReason.textContent = `Trained ${out.model_id} from ${out.windows} windows`;
        } catch (err) {
            els.stopReason.textContent = err.message;
        } finally {
            els.trainNow.disabled = false;
        }
    };
}
if (els.probe) {
    els.probe.onclick = async () => {
        els.probe.disabled = true;
        els.stopReason.textContent = `Testing last 2 winner datasets on ${els.model.value}…`;
        try {
            const out = await api("/api/arena/probe-recent", "POST", { model_id: els.model.value, count: 2 });
            applyState(out.state);
            renderProbe(out.probe);
            showProbeOnCanvas(out.probe);
        } catch (err) {
            els.stopReason.textContent = err.message;
        } finally {
            els.probe.disabled = false;
        }
    };
}
if (els.model) {
    els.model.onchange = () => pushSelectedModel(false);
}
if (els.datasetTab) {
    els.datasetTab.onclick = () => {
        const open = !(els.datasetDrawer && els.datasetDrawer.classList.contains("open"));
        setDrawerOpen(open);
    };
}
if (els.datasetClose) {
    els.datasetClose.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        setDrawerOpen(false);
    });
}
if (els.testDataset) {
    els.testDataset.onclick = async () => {
        if (!selectedDatasetId) {
            els.stopReason.textContent = "Select a dataset first.";
            return;
        }
        els.testDataset.disabled = true;
        els.stopReason.textContent = `Testing ${selectedDatasetId} on ${els.model.value}…`;
        if (els.datasetScore) els.datasetScore.textContent = "Scoring every window…";
        try {
            const out = await api("/api/arena/probe-dataset", "POST", {
                dataset_id: selectedDatasetId,
                model_id: els.model.value,
            });
            applyState(out.state);
            renderDatasetResults(out.probe);
            showProbeOnCanvas(out.probe);
            els.stopReason.textContent =
                `${out.probe.passed || 0} still beat / ${out.probe.failed || 0} caught`;
        } catch (err) {
            els.stopReason.textContent = err.message;
            if (els.datasetScore) els.datasetScore.textContent = err.message;
        } finally {
            els.testDataset.disabled = false;
        }
    };
}
if (els.trainFails) {
    els.trainFails.title = "Retrain the selected model on windows the last dataset test still missed (HC ≥ 65%). Does not reset the harvest.";
    els.trainFails.onclick = async () => {
        if (lastLeakCount < 1) {
            els.stopReason.textContent = "Test the entire dataset first so failed windows are known.";
            return;
        }
        trainFailsBusy = true;
        els.trainFails.disabled = true;
        if (els.testDataset) els.testDataset.disabled = true;
        els.stopReason.textContent = `Training ${els.model.value} on ${lastLeakCount} failed windows…`;
        if (els.datasetScore) els.datasetScore.textContent = "Retraining on windows the model still missed…";
        try {
            const out = await api("/api/arena/train-fails", "POST", { model_id: els.model.value });
            lastLeakCount = 0;
            applyState(out.state);
            await refreshModels(out.model_id);
            if (els.datasetScore) {
                els.datasetScore.textContent =
                    `Trained ${out.model_id} on ${out.windows || 0} failed windows. Test the dataset again to see what still leaks.`;
            }
            els.stopReason.textContent =
                `Trained ${out.model_id} on ${out.windows || 0} failed windows (${out.episodes || 0} episodes)`;
        } catch (err) {
            els.stopReason.textContent = err.message;
            if (els.datasetScore) els.datasetScore.textContent = err.message;
        } finally {
            trainFailsBusy = false;
            if (els.trainFails) els.trainFails.disabled = lastLeakCount < 1;
            if (els.testDataset) els.testDataset.disabled = false;
        }
    };
}
let targetTimer = null;
if (els.windowTarget) {
    const pushTarget = () => {
        api("/api/arena/windows-target", "POST", { windows_target: readWindowTarget() }).catch((err) => {
            if (els.stopReason) els.stopReason.textContent = err.message;
        });
    };
    els.windowTarget.onchange = pushTarget;
    els.windowTarget.oninput = () => {
        clearTimeout(targetTimer);
        targetTimer = setTimeout(pushTarget, 600);
    };
}
els.scrollAll.onchange = syncWatchControls;
els.watch.onchange = syncWatchControls;
if (els.workerCount) {
    els.workerCount.onchange = () => {
        clearTimeout(workerTimer);
        pushWorkerCount();
    };
    els.workerCount.oninput = () => {
        clearTimeout(workerTimer);
        workerTimer = setTimeout(pushWorkerCount, 600);
    };
}

canvas.addEventListener("click", (ev) => {
    if (!probeView) return;
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) * (W / rect.width);
    const y = (ev.clientY - rect.top) * (H / rect.height);
    const hit = probeHits.find((h) => x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h);
    if (!hit) return;
    const pct = Math.round((hit.window.human_confidence || 0) * 100);
    els.hc.textContent = `${pct}%`;
    els.hc.className = "gauge-value " + (hit.window.passed ? "hot" : "cold");
    els.hcSub.textContent = hit.window.passed ? "window still beats" : "window caught";
    els.challenge.textContent = hit.episode.challenge_type || "";
    els.bot.textContent = `${hit.episode.bot_id || ""} W${(hit.window.index || 0) + 1}`;
    if (hit.window.trail && hit.window.trail.length) {
        lastTrail = hit.window.trail;
        drawProbeCanvas(probeView);
        ctx.save();
        ctx.globalAlpha = 0.95;
        ctx.beginPath();
        ctx.strokeStyle = hit.window.passed ? "rgba(52,211,153,0.95)" : "rgba(251,113,133,0.95)";
        ctx.lineWidth = 2;
        ctx.moveTo(hit.window.trail[0].x, hit.window.trail[0].y);
        for (let i = 1; i < hit.window.trail.length; i++) {
            ctx.lineTo(hit.window.trail[i].x, hit.window.trail[i].y);
        }
        ctx.stroke();
        ctx.restore();
    }
});

const src = new EventSource("/api/arena/stream");
src.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.state) applyState(msg.state);
    if (msg.type === "attempt" && msg.attempt) {
        if (probeView) clearProbeView(false);
        showAttempt(msg.attempt, msg.attempt.trail || []);
    }
    if (msg.type === "log" && msg.message) {
        els.log.textContent = (els.log.textContent + "\n" + msg.message).trim();
        els.log.scrollTop = els.log.scrollHeight;
    }
    if (msg.type === "heartbeat" || msg.type === "hello" || msg.type === "state") {
        applyState(msg.state);
    }
    if (msg.type === "stopped") applyState(msg.state);
};

boot().catch((err) => { els.stopReason.textContent = err.message; });
