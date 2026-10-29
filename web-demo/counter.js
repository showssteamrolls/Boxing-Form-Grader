(function (root) {
  const FIGHTERS = ["A", "B"];

  const n0 = (v) => String(Math.round(v));
  const n1 = (v) => v.toFixed(1);
  const pct = (v) => Math.round(v * 100) + "%";
  const signed1 = (v) => (v > 0 ? "+" : "") + v.toFixed(1);

  const ROWS = [
    { key: "out", label: "Output", unit: "punches/min", fmt: n0, note: "last 10s" },
    { key: "land", label: "Land rate", unit: "", fmt: pct, fixed: 1, note: "last 30s" },
    { key: "lead_act", label: "Lead-hand activity", unit: "torso/s", fmt: n1, note: "last 5s" },
    { key: "lead_angle", label: "Lead-arm angle", unit: "deg", fmt: n0, fixed: 180, note: "180 = fully extended" },
    { key: "guard", label: "Guard up", unit: "", fmt: pct, fixed: 1, note: "last 10s" },
    { key: "speed", label: "Hand speed", unit: "torso/s", fmt: n1, note: "last 8 punches" },
    { key: "press", label: "Pressure", unit: "torso/s", fmt: signed1, signed: true, note: "last 3s" },
  ];

  const BONES = [[1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8], [7, 8]];
  const COLORS = { A: "#00e5ff", B: "#ff2fb0" };

  const core = {
    FIGHTERS,
    ROWS,

    upperBound(arr, t) {
      let lo = 0;
      let hi = arr.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] <= t) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    },

    buildIndex(spar) {
      const index = {};
      for (const id of FIGHTERS) {
        const mine = spar.punches.filter((p) => p.fighter === id);
        const times = (keep) => mine.filter(keep).map((p) => p.t).sort((a, b) => a - b);
        index[id] = {
          thrown: times(() => true),
          landed: times((p) => p.landed),
          jab: times((p) => p.type === "jab"),
          power: times((p) => p.type === "power"),
          lead: times((p) => p.lead),
        };
      }
      return index;
    },

    countsAt(index, t) {
      const out = {};
      for (const id of FIGHTERS) {
        const thrown = core.upperBound(index[id].thrown, t);
        const landed = core.upperBound(index[id].landed, t);
        const lead = core.upperBound(index[id].lead, t);
        out[id] = {
          thrown,
          landed,
          lead,
          jab: core.upperBound(index[id].jab, t),
          power: core.upperBound(index[id].power, t),
          rate: thrown > 0 ? landed / thrown : null,
          leadShare: thrown > 0 ? lead / thrown : null,
        };
      }
      return out;
    },

    seriesIndex(series, t) {
      const i = Math.floor((t - series.t0) / series.dt + 1e-9);
      return Math.min(series.n - 1, Math.max(0, i));
    },

    valueAt(series, id, key, t) {
      const v = series[id][key][core.seriesIndex(series, t)];
      return v === null || v === undefined ? null : v;
    },

    domain(series, row) {
      if (row.fixed) return { lo: 0, hi: row.fixed };
      const vals = [];
      for (const id of FIGHTERS) for (const v of series[id][row.key]) if (v !== null && isFinite(v)) vals.push(v);
      if (!vals.length) return { lo: 0, hi: 1 };
      vals.sort((a, b) => a - b);
      const hi = Math.max(vals[Math.floor(0.97 * (vals.length - 1))], 1e-6);
      const lo = row.signed ? Math.min(0, vals[Math.floor(0.03 * (vals.length - 1))]) : 0;
      return { lo, hi };
    },

    leadShare(a, b) {
      if (a === null || b === null) return null;
      const total = Math.max(0, a) + Math.max(0, b);
      if (total < 0.05) return null;
      return Math.max(0, a) / total;
    },

    formatClock(t) {
      const whole = Math.max(0, Math.floor(t));
      const m = Math.floor(whole / 60);
      const s = whole % 60;
      return m + ":" + String(s).padStart(2, "0");
    },

    poseAt(pose, id, t) {
      const k = Math.round((t - pose.t0) * pose.hz);
      if (k < 0 || k >= pose[id].length) return null;
      return pose[id][k];
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = core;
  root.PunchCounterCore = core;

  if (typeof document === "undefined") return;

  const section = document.getElementById("counter");
  if (!section) return;

  const $ = (id) => document.getElementById(id);
  const wip = $("gradeWip");
  const videoHost = $("pcVideo");
  const overlay = $("pcOverlay");
  const overlayBtn = $("pcOverlayBtn");
  const timeline = $("pcTimeline");
  const clock = $("pcClock");
  const picker = $("pcPicker");
  const rowsHost = $("pcRows");
  const chart = $("pcChart");
  const chartTitle = $("pcChartTitle");
  const leadBar = $("pcLeadBar");
  const leadText = $("pcLeadText");
  const cards = { A: $("pcCardA"), B: $("pcCardB") };

  const state = {
    spar: null, index: null, pose: null, player: null, raf: 0, lastT: -1, token: 0,
    playhead: null, rowEls: {}, chartRow: ROWS[2], showOverlay: true, shapes: [],
  };

  function buildCard(card, id) {
    card.className = "pc-card " + id.toLowerCase();
    card.innerHTML =
      '<div class="pc-name"><span data-k="label"></span><span class="pc-stance" data-k="stance"></span></div>' +
      '<div class="pc-hero">' +
      '<div><span class="pc-big" data-k="thrown">0</span><span class="pc-cap">thrown</span></div>' +
      '<div><span class="pc-big" data-k="landed">0</span><span class="pc-cap">landed</span></div>' +
      '<div><span class="pc-big" data-k="rate">--</span><span class="pc-cap">land rate</span></div>' +
      "</div>" +
      '<div class="pc-stats">' +
      '<div><b data-k="jab">0</b><span>jabs</span></div>' +
      '<div><b data-k="power">0</b><span>power</span></div>' +
      '<div><b data-k="leadShare">--</b><span>from lead hand</span></div>' +
      "</div>";
  }

  function field(card, key) {
    return card.querySelector('[data-k="' + key + '"]');
  }

  function setValue(card, key, value, bump) {
    const el = field(card, key);
    const text = String(value);
    if (el.textContent === text) return;
    const prev = Number(el.textContent);
    el.textContent = text;
    if (bump && Number.isFinite(prev) && value - prev > 0 && value - prev <= 2) {
      el.classList.remove("bump");
      void el.offsetWidth;
      el.classList.add("bump");
    }
  }

  function buildRows() {
    rowsHost.innerHTML = "";
    state.rowEls = {};
    for (const row of ROWS) {
      const el = document.createElement("button");
      el.type = "button";
      el.className = "pc-row";
      el.dataset.key = row.key;
      el.innerHTML =
        '<span class="pc-val a">--</span>' +
        '<span class="pc-mid-cell">' +
        '<span class="pc-rowlabel">' + row.label + "<em>" + (row.unit ? row.unit + " · " : "") + row.note + "</em></span>" +
        '<span class="pc-bars"><span class="pc-bar a"><i></i></span><span class="pc-bar b"><i></i></span></span>' +
        "</span>" +
        '<span class="pc-val b">--</span>';
      el.addEventListener("click", () => {
        state.chartRow = row;
        markRows();
        renderChart(state.lastT);
      });
      rowsHost.appendChild(el);
      state.rowEls[row.key] = {
        el,
        a: el.querySelector(".pc-val.a"),
        b: el.querySelector(".pc-val.b"),
        barA: el.querySelector(".pc-bar.a i"),
        barB: el.querySelector(".pc-bar.b i"),
      };
    }
    markRows();
  }

  function markRows() {
    for (const row of ROWS) state.rowEls[row.key].el.classList.toggle("active", row === state.chartRow);
  }

  function setText(el, text) {
    if (el.textContent !== text) el.textContent = text;
  }

  function renderRows(t) {
    const series = state.spar.series;
    for (const row of ROWS) {
      const els = state.rowEls[row.key];
      const dom = state.domains[row.key];
      const va = core.valueAt(series, "A", row.key, t);
      const vb = core.valueAt(series, "B", row.key, t);
      setText(els.a, va === null ? "--" : row.fmt(va));
      setText(els.b, vb === null ? "--" : row.fmt(vb));
      const width = (v) =>
        v === null ? 0 : Math.max(0, Math.min(1, (Math.max(0, v) - Math.max(0, dom.lo)) / (dom.hi - Math.max(0, dom.lo) || 1)));
      els.barA.style.width = width(va) * 100 + "%";
      els.barB.style.width = width(vb) * 100 + "%";
      els.a.classList.toggle("lead", va !== null && vb !== null && va > vb);
      els.b.classList.toggle("lead", va !== null && vb !== null && vb > va);
    }
    const share = core.leadShare(
      core.valueAt(series, "A", "lead_act", t),
      core.valueAt(series, "B", "lead_act", t)
    );
    const labels = state.spar.fighters;
    if (share === null) {
      leadBar.style.setProperty("--share", "50%");
      leadBar.classList.add("idle");
      setText(leadText, "Both lead hands quiet");
    } else {
      leadBar.classList.remove("idle");
      leadBar.style.setProperty("--share", share * 100 + "%");
      setText(leadText, labels[0].label + " " + Math.round(share * 100) + "%  ·  " + Math.round((1 - share) * 100) + "% " + labels[1].label);
    }
  }

  function sizeCanvas(cvs) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(cvs.clientWidth * dpr));
    const h = Math.max(1, Math.round(cvs.clientHeight * dpr));
    if (cvs.width !== w || cvs.height !== h) {
      cvs.width = w;
      cvs.height = h;
    }
    return { w, h, dpr };
  }

  function renderChart(t) {
    if (!state.spar || !state.spar.series) return;
    const { w, h, dpr } = sizeCanvas(chart);
    const ctx = chart.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    const series = state.spar.series;
    const row = state.chartRow;
    const dom = state.domains[row.key];
    const padL = 8 * dpr, padR = 8 * dpr, padT = 10 * dpr, padB = 10 * dpr;
    const x = (i) => padL + (i / Math.max(1, series.n - 1)) * (w - padL - padR);
    const lo = dom.lo, hi = dom.hi;
    const y = (v) => padT + (1 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo || 1)) * (h - padT - padB);
    setText(chartTitle, row.label + (row.unit ? " (" + row.unit + ")" : "") + " · " + row.note);

    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = dpr;
    for (let g = 0; g <= 3; g++) {
      const gy = padT + (g / 3) * (h - padT - padB);
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(w - padR, gy);
      ctx.stroke();
    }
    if (row.signed && lo < 0) {
      ctx.strokeStyle = "rgba(255,255,255,0.2)";
      ctx.beginPath();
      ctx.moveTo(padL, y(0));
      ctx.lineTo(w - padR, y(0));
      ctx.stroke();
    }

    const now = core.seriesIndex(series, t);
    for (const id of FIGHTERS) {
      const data = series[id][row.key];
      for (const pass of [0, 1]) {
        ctx.beginPath();
        ctx.strokeStyle = COLORS[id];
        ctx.globalAlpha = pass === 0 ? 0.22 : 1;
        ctx.lineWidth = (pass === 0 ? 1.5 : 2.4) * dpr;
        ctx.lineJoin = "round";
        let open = false;
        const last = pass === 0 ? data.length - 1 : now;
        for (let i = 0; i <= last; i++) {
          const v = data[i];
          if (v === null) {
            open = false;
            continue;
          }
          if (!open) {
            ctx.moveTo(x(i), y(v));
            open = true;
          } else ctx.lineTo(x(i), y(v));
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      const cur = data[now];
      if (cur !== null && cur !== undefined) {
        ctx.fillStyle = COLORS[id];
        ctx.beginPath();
        ctx.arc(x(now), y(cur), 4 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = dpr;
    ctx.beginPath();
    ctx.moveTo(x(now), padT);
    ctx.lineTo(x(now), h - padB);
    ctx.stroke();
  }

  function renderOverlay(t) {
    const { w, h, dpr } = sizeCanvas(overlay);
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    if (!state.showOverlay || !state.pose) return;
    const pose = state.pose;
    for (const id of FIGHTERS) {
      const row = core.poseAt(pose, id, t);
      if (!row) continue;
      const pt = (j) => (row[j * 2] < 0 ? null : [(row[j * 2] / 1000) * w, (row[j * 2 + 1] / 1000) * h]);
      ctx.strokeStyle = COLORS[id];
      ctx.fillStyle = COLORS[id];
      ctx.lineWidth = 3 * dpr;
      ctx.lineCap = "round";
      ctx.shadowColor = "rgba(0,0,0,0.7)";
      ctx.shadowBlur = 3 * dpr;
      for (const [a, b] of BONES) {
        const pa = pt(a), pb = pt(b);
        if (!pa || !pb) continue;
        ctx.beginPath();
        ctx.moveTo(pa[0], pa[1]);
        ctx.lineTo(pb[0], pb[1]);
        ctx.stroke();
      }
      const nose = pt(0), ls = pt(1), rs = pt(2);
      if (nose && ls && rs) {
        ctx.beginPath();
        ctx.arc(nose[0], nose[1], Math.max(5 * dpr, Math.hypot(ls[0] - rs[0], ls[1] - rs[1]) * 0.35), 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.shadowBlur = 0;
      for (const j of [5, 6]) {
        const p = pt(j);
        if (!p) continue;
        ctx.beginPath();
        ctx.arc(p[0], p[1], 6 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    for (const p of state.spar.punches) {
      const age = t - p.t;
      if (age < 0 || age > 0.45) continue;
      const row = core.poseAt(pose, p.fighter, p.t);
      if (!row) continue;
      const j = p.hand === "left" ? 5 : 6;
      if (row[j * 2] < 0) continue;
      const cx = (row[j * 2] / 1000) * w, cy = (row[j * 2 + 1] / 1000) * h;
      const k = age / 0.45;
      ctx.strokeStyle = p.landed ? "#ffffff" : COLORS[p.fighter];
      ctx.globalAlpha = 1 - k;
      ctx.lineWidth = (p.landed ? 4 : 2.5) * dpr;
      ctx.beginPath();
      ctx.arc(cx, cy, (10 + 26 * k) * dpr, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  function render(t) {
    const counts = core.countsAt(state.index, t);
    for (const id of FIGHTERS) {
      const c = counts[id];
      setValue(cards[id], "thrown", c.thrown, true);
      setValue(cards[id], "landed", c.landed, true);
      setValue(cards[id], "rate", c.rate === null ? "--" : pct(c.rate));
      setValue(cards[id], "jab", c.jab);
      setValue(cards[id], "power", c.power);
      setValue(cards[id], "leadShare", c.leadShare === null ? "--" : pct(c.leadShare));
    }
    const spar = state.spar;
    setText(clock, core.formatClock(t - spar.start));
    const frac = (t - spar.start) / Math.max(1e-6, spar.end - spar.start);
    if (state.playhead) state.playhead.style.left = Math.min(1, Math.max(0, frac)) * 100 + "%";
    if (spar.series) {
      renderRows(t);
      renderChart(t);
      const range = spar.series.dist[core.seriesIndex(spar.series, t)];
      setText($("pcRange"), range === null || range === undefined ? "--" : range.toFixed(1));
    }
    renderOverlay(t);
  }

  function buildTimeline() {
    timeline.innerHTML = "";
    const spar = state.spar;
    const span = Math.max(1e-6, spar.end - spar.start);
    const lanes = {};
    for (const id of FIGHTERS) {
      const lane = document.createElement("div");
      lane.className = "pc-lane " + id.toLowerCase();
      timeline.appendChild(lane);
      lanes[id] = lane;
    }
    for (const p of spar.punches) {
      const tick = document.createElement("div");
      tick.className = "pc-tick" + (p.type === "power" ? " power" : "") + (p.landed ? " hit" : "");
      tick.style.left = ((p.t - spar.start) / span) * 100 + "%";
      lanes[p.fighter].appendChild(tick);
    }
    const head = document.createElement("div");
    head.className = "pc-playhead";
    timeline.appendChild(head);
    state.playhead = head;
  }

  function fillCards() {
    for (const f of state.spar.fighters) {
      field(cards[f.id], "label").textContent = f.label;
      field(cards[f.id], "stance").textContent = f.stance;
    }
    $("pcTitle").textContent = state.spar.title || "";
  }

  function loadYouTubeApi() {
    return new Promise((resolve) => {
      if (window.YT && window.YT.Player) return resolve();
      const previous = window.onYouTubeIframeAPIReady;
      window.onYouTubeIframeAPIReady = () => {
        if (previous) previous();
        resolve();
      };
      const tag = document.createElement("script");
      tag.src = "https://www.youtube.com/iframe_api";
      document.head.appendChild(tag);
    });
  }

  function showVideoError(message) {
    const note = document.createElement("div");
    note.className = "pc-video-error";
    note.textContent = message;
    videoHost.appendChild(note);
  }

  async function createYouTubePlayer(spar) {
    await loadYouTubeApi();
    const holder = document.createElement("div");
    videoHost.insertBefore(holder, overlay);
    const vars = { start: Math.floor(spar.start), rel: 0, modestbranding: 1, playsinline: 1 };
    if (spar.end) vars.end = Math.ceil(spar.end);
    return new Promise((resolve) => {
      const player = new window.YT.Player(holder, {
        videoId: spar.youtube,
        playerVars: vars,
        events: {
          onReady: () =>
            resolve({
              el: holder,
              time: () => player.getCurrentTime(),
              seek: (s) => player.seekTo(s, true),
              destroy: () => player.destroy(),
            }),
          onError: () => {
            showVideoError("This video can't be embedded.");
            resolve(null);
          },
        },
      });
    });
  }

  function createVideoPlayer(spar) {
    const video = document.createElement("video");
    video.src = spar.video;
    video.controls = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.addEventListener(
      "loadedmetadata",
      () => {
        if (spar.start) video.currentTime = spar.start;
      },
      { once: true }
    );
    videoHost.insertBefore(video, overlay);
    return {
      el: video,
      time: () => video.currentTime,
      seek: (s) => {
        video.currentTime = s;
      },
      destroy: () => video.remove(),
    };
  }

  function tick() {
    if (state.player) {
      let t = state.lastT;
      try {
        t = state.player.time();
      } catch (err) {
        t = state.lastT;
      }
      if (Number.isFinite(t) && Math.abs(t - state.lastT) > 0.001) {
        state.lastT = t;
        render(t);
      }
    }
    state.raf = requestAnimationFrame(tick);
  }

  function markPicker(slug) {
    for (const btn of picker.querySelectorAll("button")) {
      btn.classList.toggle("active", btn.dataset.slug === slug);
    }
  }

  async function selectSpar(slug) {
    const token = ++state.token;
    cancelAnimationFrame(state.raf);
    if (state.player) {
      state.player.destroy();
      state.player = null;
    }
    for (const node of [...videoHost.children]) if (node !== overlay && node !== overlayBtn) node.remove();
    const res = await fetch("spars/" + slug + ".json");
    const spar = await res.json();
    if (token !== state.token) return;
    state.spar = spar;
    state.index = core.buildIndex(spar);
    state.pose = null;
    state.lastT = -1;
    state.domains = {};
    if (spar.series) for (const row of ROWS) state.domains[row.key] = core.domain(spar.series, row);
    $("pcPanels").hidden = !spar.series;
    fillCards();
    buildTimeline();
    markPicker(slug);
    render(spar.start);
    const player = spar.youtube ? await createYouTubePlayer(spar) : createVideoPlayer(spar);
    if (token !== state.token) {
      if (player) player.destroy();
      return;
    }
    state.player = player;
    state.raf = requestAnimationFrame(tick);
    overlayBtn.hidden = true;
    try {
      const poseRes = await fetch("spars/" + slug + ".pose.json");
      if (poseRes.ok) {
        const pose = await poseRes.json();
        if (token === state.token) {
          state.pose = pose;
          overlayBtn.hidden = false;
          state.lastT = -1;
        }
      }
    } catch (err) {
      state.pose = null;
    }
  }

  function seekFraction(el, e) {
    if (!state.player || !state.spar) return;
    const rect = el.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    state.player.seek(state.spar.start + frac * (state.spar.end - state.spar.start));
  }

  timeline.addEventListener("click", (e) => seekFraction(timeline, e));
  chart.addEventListener("click", (e) => seekFraction(chart, e));
  overlayBtn.addEventListener("click", () => {
    state.showOverlay = !state.showOverlay;
    overlayBtn.textContent = state.showOverlay ? "Skeleton: on" : "Skeleton: off";
    overlayBtn.classList.toggle("off", !state.showOverlay);
    state.lastT = -1;
  });
  window.addEventListener("resize", () => {
    if (state.spar && state.lastT >= 0) render(state.lastT);
  });

  async function init() {
    let manifest;
    try {
      const res = await fetch("spars/index.json", { cache: "no-cache" });
      if (!res.ok) return;
      manifest = await res.json();
    } catch (err) {
      return;
    }
    if (!manifest.spars || !manifest.spars.length) return;

    buildCard(cards.A, "A");
    buildCard(cards.B, "B");
    buildRows();
    section.hidden = false;
    if (wip) wip.hidden = false;

    if (manifest.spars.length > 1) {
      for (const s of manifest.spars) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = s.title;
        btn.dataset.slug = s.slug;
        btn.addEventListener("click", () => selectSpar(s.slug));
        picker.appendChild(btn);
      }
    }

    const wanted = new URLSearchParams(location.search).get("spar");
    const initial = manifest.spars.find((s) => s.slug === wanted) || manifest.spars[0];
    await selectSpar(initial.slug);
  }

  init();
})(typeof window !== "undefined" ? window : globalThis);
