(function (root) {
  const FIGHTERS = ["A", "B"];

  const core = {
    FIGHTERS,

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
        };
      }
      return index;
    },

    countsAt(index, t) {
      const out = {};
      for (const id of FIGHTERS) {
        const thrown = core.upperBound(index[id].thrown, t);
        const landed = core.upperBound(index[id].landed, t);
        out[id] = {
          thrown,
          landed,
          jab: core.upperBound(index[id].jab, t),
          power: core.upperBound(index[id].power, t),
          pct: thrown > 0 ? Math.round((landed / thrown) * 100) : null,
        };
      }
      return out;
    },

    formatClock(t) {
      const whole = Math.max(0, Math.floor(t));
      const m = Math.floor(whole / 60);
      const s = whole % 60;
      return m + ":" + String(s).padStart(2, "0");
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = core;
  root.PunchCounterCore = core;

  if (typeof document === "undefined") return;

  const section = document.getElementById("counter");
  if (!section) return;

  const wip = document.getElementById("gradeWip");
  const videoHost = document.getElementById("pcVideo");
  const timeline = document.getElementById("pcTimeline");
  const clock = document.getElementById("pcClock");
  const picker = document.getElementById("pcPicker");
  const cards = { A: document.getElementById("pcCardA"), B: document.getElementById("pcCardB") };

  const state = { spar: null, index: null, player: null, raf: 0, lastT: -1, token: 0, playhead: null };

  function buildCard(card) {
    card.innerHTML =
      '<div class="pc-name"><span data-k="label"></span><span class="pc-stance" data-k="stance"></span></div>' +
      '<div class="pc-thrown"><span class="pc-big" data-k="thrown">0</span><span class="pc-cap">thrown</span></div>' +
      '<div class="pc-stats">' +
      '<div><b data-k="landed">0</b><span>landed</span></div>' +
      '<div><b data-k="pct">--</b><span>accuracy</span></div>' +
      '<div><b data-k="jab">0</b><span>jabs</span></div>' +
      '<div><b data-k="power">0</b><span>power</span></div>' +
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

  function render(t) {
    const counts = core.countsAt(state.index, t);
    for (const id of FIGHTERS) {
      const c = counts[id];
      setValue(cards[id], "thrown", c.thrown, true);
      setValue(cards[id], "landed", c.landed);
      setValue(cards[id], "jab", c.jab);
      setValue(cards[id], "power", c.power);
      setValue(cards[id], "pct", c.pct === null ? "--" : c.pct + "%");
    }
    const spar = state.spar;
    clock.textContent = core.formatClock(t - spar.start);
    const frac = (t - spar.start) / Math.max(1e-6, spar.end - spar.start);
    if (state.playhead) state.playhead.style.left = Math.min(1, Math.max(0, frac)) * 100 + "%";
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
    videoHost.innerHTML = "";
    const note = document.createElement("div");
    note.className = "pc-video-error";
    note.textContent = message;
    videoHost.appendChild(note);
  }

  async function createYouTubePlayer(spar) {
    await loadYouTubeApi();
    const holder = document.createElement("div");
    videoHost.appendChild(holder);
    const vars = { start: Math.floor(spar.start), rel: 0, modestbranding: 1, playsinline: 1 };
    if (spar.end) vars.end = Math.ceil(spar.end);
    return new Promise((resolve) => {
      const player = new window.YT.Player(holder, {
        videoId: spar.youtube,
        playerVars: vars,
        events: {
          onReady: () =>
            resolve({
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
    videoHost.appendChild(video);
    return {
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
      if (Number.isFinite(t) && Math.abs(t - state.lastT) > 0.01) {
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
    videoHost.innerHTML = "";
    const res = await fetch("spars/" + slug + ".json");
    const spar = await res.json();
    if (token !== state.token) return;
    state.spar = spar;
    state.index = core.buildIndex(spar);
    state.lastT = -1;
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
  }

  timeline.addEventListener("click", (e) => {
    if (!state.player || !state.spar) return;
    const rect = timeline.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    state.player.seek(state.spar.start + frac * (state.spar.end - state.spar.start));
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

    buildCard(cards.A);
    buildCard(cards.B);
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
