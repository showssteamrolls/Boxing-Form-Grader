# Web Demo (deployable to Vercel)

A browser-only version of the grader — pose detection runs entirely
client-side via MediaPipe's WASM build, so a visitor's webcam never leaves
their machine. No backend, no server-side Python.

**This is a separate, hand-ported copy of the scoring logic in `../src/`**,
not a build of the Python code — browsers can't run the Python pipeline
directly. If you retune thresholds in `src/config.py`, mirror the same
numbers in `index.html`'s `CONFIG` object (near the top of the `<script>`
block) or the two demos will disagree.

## Deploy to Vercel

This is a single static file, so there's nothing to build:

```bash
cd web-demo
npx vercel deploy
```

Or via the Vercel dashboard: New Project → import this folder (or point it
at `web-demo/` as the root directory if deploying the whole repo) → deploy.
No framework preset, no build command needed — it's just `index.html`.

## Local preview

```bash
cd web-demo
npx serve .
```

(Opening `index.html` directly via `file://` will NOT work — browsers block
webcam access on the `file://` protocol. You need a local server.)

## Known differences from the Python version

- **Upload-based, not live webcam** — three tiles, each takes an uploaded
  video file (drag-and-drop or click to browse). No `getUserMedia` webcam
  path in this version. The Python `--video` flag has no browser equivalent
  either way, but the browser demo's earlier webcam-only version is gone —
  swap `Grader.loadFile` for a `getUserMedia` path if you want live camera back.
- **No YouTube link support** — browsers can't pull raw pixels out of an
  embedded YouTube player (cross-origin restriction by design), and pose
  estimation needs raw frames. Download clips first (e.g. with `yt-dlp`,
  same as the Python side) and upload the resulting file.
- **No visibility field guarantee** — MediaPipe's browser build doesn't
  always populate per-landmark `visibility`; the code defaults to `1.0`
  (trust it) when absent, same fallback as the Python side.
- **Punch detection thresholds are not yet tuned** — ported directly from
  `src/config.py`'s defaults. Tune against real footage on the Python side
  first, then copy final numbers into this file's `CONFIG` object.
