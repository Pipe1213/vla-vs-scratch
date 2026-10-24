#!/usr/bin/env python3
"""Local page for coding the sampled evaluation videos (Phase 4 failure analysis).

    python scripts/code_videos.py                 # then open http://localhost:8765 in the Windows browser
    python scripts/code_videos.py --port 8766     # if 8765 is taken
    python scripts/code_videos.py --host 0.0.0.0  # only if Windows cannot reach localhost

Reads   notes/coding_sample.csv   (fixed sample, from scripts/make_coding_sample.py)
        outputs/eval/...mp4       (videos, streamed to the page)
Writes  notes/failure_codes.csv   (one row per coded video, saved on every "save"; safe to stop any time)

Blind coding: the page shows only the blind id (V01..V90) and the simulator outcome. The run (method, demos,
seed, checkpoint) is sent to the page only after that video has been saved.
Standard library only. Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "notes" / "coding_sample.csv"
CODES_CSV = ROOT / "notes" / "failure_codes.csv"

# Codes and definitions from notes/failure_log.md (same names, same order as the number keys).
CODES = [
    ("G-MISS", "Missed grasp: gripper closes beside / above the object, nothing is picked up"),
    ("G-SLIP", "Dropped object: object is lifted, then falls before reaching the basket"),
    ("NO-RETRY", "Continues without object: after a miss/drop, keeps moving to the basket and never re-attempts"),
    ("RETRY-FAIL", "Retries, then gives up: several grasp attempts after a miss, none succeed"),
    ("COLLIDE", "Collision: arm or gripper hits a neighbouring object, knocking it or the target away"),
    ("WRONG-OBJ", "Wrong object: approaches or picks an object other than the instructed one"),
    ("PLACE", "Bad placement: object reaches the basket area but is released outside / falls off"),
    ("TIMEOUT", "Stalls / hesitates: hovers, jitters or moves slowly until the time limit"),
    ("OTHER", "Other: describe it in the comment"),
]
CODE_NAMES = [c for c, _ in CODES]
ATTEMPTS = ["1", "2", "3+"]
OUT_FIELDS = ["vid", "success", "codes", "attempts", "flag", "comment", "coded_at",
              "cell", "run", "method", "task", "size", "seed", "checkpoint", "episode", "video"]
META_FIELDS = ["cell", "run", "method", "task", "size", "seed", "checkpoint", "episode"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


class Store:
    def __init__(self):
        self.sample = {r["vid"]: r for r in read_csv(SAMPLE)}
        if not self.sample:
            raise SystemExit(f"{SAMPLE} is missing or empty: run scripts/make_coding_sample.py first")
        self.codes = {r["vid"]: r for r in read_csv(CODES_CSV)}

    def items(self) -> list[dict]:
        out = []
        for vid, s in self.sample.items():
            c = self.codes.get(vid)
            item = dict(vid=vid, success=int(s["success"]), exists=(ROOT / s["video"]).is_file(), coded=c is not None)
            if c:
                item.update(codes=[x for x in c["codes"].split("+") if x], attempts=c["attempts"],
                            flag=c["flag"] == "1", comment=c["comment"], meta={k: s[k] for k in META_FIELDS})
            out.append(item)
        return out

    def save(self, d: dict) -> dict:
        vid = d.get("vid")
        if vid not in self.sample:
            raise ValueError(f"unknown video {vid}")
        codes = [c for c in d.get("codes", []) if c in CODE_NAMES]
        if len(set(codes)) != len(codes):
            raise ValueError("duplicate codes")
        attempts = str(d.get("attempts", "1"))
        if attempts not in ATTEMPTS:
            raise ValueError(f"attempts must be one of {ATTEMPTS}")
        s = self.sample[vid]
        self.codes[vid] = dict(vid=vid, success=s["success"], codes="+".join(codes), attempts=attempts,
                               flag="1" if d.get("flag") else "0", comment=str(d.get("comment", "")).strip(),
                               coded_at=datetime.now().isoformat(timespec="seconds"),
                               **{k: s[k] for k in META_FIELDS}, video=s["video"])
        self._write()
        return {k: s[k] for k in META_FIELDS}

    def _write(self):
        order = {vid: i for i, vid in enumerate(self.sample)}
        rows = sorted(self.codes.values(), key=lambda r: order.get(r["vid"], 1e9))
        fd, tmp = tempfile.mkstemp(dir=CODES_CSV.parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, CODES_CSV)  # atomic: a crash never leaves a half-written file


STORE: Store | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: smoother video seeking
    def log_request(self, code="-", size="-"):  # quiet: errors are still printed by log_error
        pass

    def _send(self, status, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(status, json.dumps(obj).encode(), "application/json")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = PAGE.replace("__CODES__", json.dumps(CODES)).replace("__ATTEMPTS__", json.dumps(ATTEMPTS))
            return self._send(HTTPStatus.OK, page.encode(), "text/html; charset=utf-8")
        if self.path == "/api/items":
            return self._json(STORE.items())
        m = re.fullmatch(r"/video/(V\d+)\.mp4", self.path)
        if m and m.group(1) in STORE.sample:
            return self._video(ROOT / STORE.sample[m.group(1)]["video"])
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", 0))
            meta = STORE.save(json.loads(self.rfile.read(n)))
            self._json({"ok": True, "meta": meta})
        except (ValueError, json.JSONDecodeError) as e:
            self._json({"ok": False, "error": str(e)}, HTTPStatus.BAD_REQUEST)

    def _video(self, path: Path):
        """Serve an mp4 with HTTP Range support (browsers need it to seek and step frames)."""
        if not path.is_file():
            return self._send(HTTPStatus.NOT_FOUND, b"video missing", "text/plain")
        size = path.stat().st_size
        start, end, status = 0, size - 1, HTTPStatus.OK
        rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        if rng and (rng.group(1) or rng.group(2)):
            if rng.group(1):
                start = int(rng.group(1))
                end = min(int(rng.group(2)), size - 1) if rng.group(2) else size - 1
            else:  # suffix range: last N bytes
                start = max(0, size - int(rng.group(2)))
            if start > end or start >= size:
                return self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "text/plain",
                                  {"Content-Range": f"bytes */{size}"})
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as f:
            f.seek(start)
            left = length
            try:
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser cancelled the request (normal when seeking)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Failure coding</title>
<style>
:root {
  --bg: #f6f6f4; --panel: #ffffff; --ink: #161615; --ink-2: #57564f; --ink-3: #8a8980; --line: #e3e2dc;
  --accent: #2a78d6; --accent-ink: #ffffff; --chip: #eeede8;
  --ok: #0f7a3d; --ok-bg: #e3f4ea; --bad: #b42318; --bad-bg: #fde8e6; --warn: #8a5a00; --warn-bg: #fff3d6;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141413; --panel: #1d1d1b; --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #8f8e86; --line: #33332f;
    --accent: #3987e5; --accent-ink: #ffffff; --chip: #2a2a27;
    --ok: #6fd39a; --ok-bg: #173323; --bad: #ff8a80; --bad-bg: #3a1a17; --warn: #ffd27a; --warn-bg: #3a2e12;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.app { display: grid; grid-template-columns: 210px 1fr 340px; height: 100vh; }
aside, .side { background: var(--panel); border-right: 1px solid var(--line); overflow-y: auto; }
.side { border-right: 0; border-left: 1px solid var(--line); padding: 16px; display: flex; flex-direction: column; gap: 14px; }
aside header { position: sticky; top: 0; background: var(--panel); padding: 14px 14px 10px; border-bottom: 1px solid var(--line); }
h1 { font-size: 15px; margin: 0 0 8px; }
.bar { height: 6px; background: var(--chip); border-radius: 3px; overflow: hidden; }
.bar > div { height: 100%; background: var(--accent); width: 0; transition: width .2s; }
.count { font-size: 12px; color: var(--ink-2); margin-top: 6px; }
.filters { display: flex; gap: 4px; margin-top: 8px; }
.filters button { flex: 1; font-size: 11px; padding: 3px 0; }
ul { list-style: none; margin: 0; padding: 6px; }
li { display: flex; align-items: center; gap: 8px; padding: 5px 8px; border-radius: 6px; cursor: pointer; font-variant-numeric: tabular-nums; }
li:hover { background: var(--chip); }
li.cur { background: var(--accent); color: var(--accent-ink); }
li .st { width: 16px; text-align: center; }
li .meta { font-size: 11px; color: var(--ink-3); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
li.cur .meta { color: var(--accent-ink); opacity: .85; }
main { display: flex; flex-direction: column; padding: 16px; gap: 10px; min-width: 0; height: 100vh; overflow: hidden; }
.top { display: flex; align-items: center; gap: 12px; }
.vid { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.badge { padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: 13px; }
.badge.ok { background: var(--ok-bg); color: var(--ok); }
.badge.bad { background: var(--bad-bg); color: var(--bad); }
.reveal { margin-left: auto; font-size: 13px; color: var(--ink-2); }
.player { flex: 1 1 0; min-height: 0; display: flex; align-items: center; justify-content: center; background: #000; border-radius: 10px; overflow: hidden; position: relative; }
video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; image-rendering: pixelated; }
.missing { color: #ddd; position: absolute; }
.controls { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.time { font-variant-numeric: tabular-nums; color: var(--ink-2); font-size: 13px; margin-left: 6px; }
button { font: inherit; color: var(--ink); background: var(--chip); border: 1px solid var(--line); border-radius: 7px; padding: 6px 10px; cursor: pointer; }
button:hover { border-color: var(--ink-3); }
button.on { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
button .k { font-size: 11px; opacity: .65; margin-right: 5px; font-variant-numeric: tabular-nums; }
.codes { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.codes button { text-align: left; position: relative; }
.codes button .ord { position: absolute; right: 7px; top: 6px; font-size: 11px; font-weight: 700; }
h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--ink-2); margin: 0 0 6px; }
.seq { min-height: 20px; font-size: 13px; color: var(--ink-2); margin-top: 6px; }
.def { font-size: 12px; color: var(--ink-2); min-height: 34px; }
.row { display: flex; gap: 6px; }
textarea { width: 100%; min-height: 70px; resize: vertical; font: inherit; color: var(--ink); background: var(--bg); border: 1px solid var(--line); border-radius: 7px; padding: 8px; }
.warn { background: var(--warn-bg); color: var(--warn); border-radius: 7px; padding: 8px 10px; font-size: 13px; display: none; }
.save { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); font-weight: 600; padding: 9px; width: 100%; }
.keys { font-size: 12px; color: var(--ink-2); border-top: 1px solid var(--line); padding-top: 10px; margin-top: auto; }
.keys kbd { font: 11px ui-monospace, monospace; background: var(--chip); border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; }
.toast { position: fixed; top: 14px; left: 50%; transform: translateX(-50%); background: var(--ink); color: var(--bg); padding: 8px 14px; border-radius: 8px; font-size: 13px; opacity: 0; transition: opacity .2s; pointer-events: none; }
.toast.show { opacity: 1; }
@media (max-width: 1100px) { .app { grid-template-columns: 160px 1fr 300px; } }
</style>
</head>
<body>
<div class="app">
  <aside>
    <header>
      <h1>Failure coding</h1>
      <div class="bar"><div id="bar"></div></div>
      <div class="count" id="count"></div>
      <div class="filters">
        <button data-f="all" class="on">All</button><button data-f="todo">To do</button><button data-f="flag">Flagged</button>
      </div>
    </header>
    <ul id="list"></ul>
  </aside>
  <main>
    <div class="top">
      <span class="vid" id="vid"></span>
      <span class="badge" id="outcome"></span>
      <span class="reveal" id="reveal"></span>
    </div>
    <div class="player"><video id="video" muted playsinline></video><div class="missing" id="missing" hidden>Video file not found on this machine</div></div>
    <div class="controls">
      <button id="play"><span class="k">Space</span>Play</button>
      <button id="back"><span class="k">←</span>Frame</button>
      <button id="fwd"><span class="k">→</span>Frame</button>
      <button id="restart"><span class="k">R</span>Restart</button>
      <span style="width:8px"></span>
      <button class="spd" data-s="0.125">⅛×</button><button class="spd on" data-s="0.25">¼×</button><button class="spd" data-s="0.5">½×</button><button class="spd" data-s="1">1×</button>
      <span class="time" id="time"></span>
    </div>
  </main>
  <section class="side">
    <div>
      <h2>Codes, in order of cause</h2>
      <div class="codes" id="codes"></div>
      <div class="seq" id="seq"></div>
      <div class="def" id="def"></div>
    </div>
    <div>
      <h2>Grasp attempts</h2>
      <div class="row" id="attempts"></div>
    </div>
    <div>
      <h2>Comment</h2>
      <textarea id="comment" placeholder="Optional (C to focus, Esc to leave)"></textarea>
    </div>
    <div class="row"><button id="flag" style="flex:1"><span class="k">F</span>Unsure / revisit</button></div>
    <div class="warn" id="warn"></div>
    <button class="save" id="save">Save &amp; next <span class="k" style="color:inherit">Enter</span></button>
    <div class="keys">
      <kbd>1</kbd>–<kbd>9</kbd> codes · <kbd>Q</kbd><kbd>W</kbd><kbd>E</kbd> attempts 1 / 2 / 3+ · <kbd>F</kbd> flag ·
      <kbd>Space</kbd> play · <kbd>←</kbd><kbd>→</kbd> frame · <kbd>Shift</kbd>+<kbd>←</kbd><kbd>→</kbd> 10 frames ·
      <kbd>S</kbd> speed · <kbd>R</kbd> restart · <kbd>[</kbd><kbd>]</kbd> prev / next · <kbd>Enter</kbd> save &amp; next.
      On a <b>success</b>, codes are near-misses (a G-MISS/G-SLIP there = recovered).
    </div>
  </section>
</div>
<div class="toast" id="toast"></div>
<script>
const CODES = __CODES__, ATTEMPTS = __ATTEMPTS__, FPS = 80;  // LeRobot LIBERO eval videos: 256x256 h264 at 80 fps (an episode lasts 1.5-3.5 s at 1x)
const $ = id => document.getElementById(id);
const video = $("video");
let items = [], idx = 0, filter = "all";
let state = { codes: [], attempts: "1", flag: false, comment: "" }, warned = "";

function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 1600); }
function metaText(m) { return `${m.method === "act" ? "ACT" : "SmolVLA"} · ${m.task} · ${m.size} demos · seed ${m.seed} · ckpt ${parseInt(m.checkpoint) / 1000}k · ep ${m.episode}`; }
function visible() { return items.map((it, i) => [it, i]).filter(([it]) => filter === "all" || (filter === "todo" && !it.coded) || (filter === "flag" && it.flag)); }

function renderList() {
  const done = items.filter(i => i.coded).length;
  $("bar").style.width = `${100 * done / items.length}%`;
  $("count").textContent = `${done} / ${items.length} coded` + (items.some(i => !i.exists) ? ` · ${items.filter(i => !i.exists).length} missing` : "");
  $("list").innerHTML = visible().map(([it, i]) => `<li data-i="${i}" class="${i === idx ? "cur" : ""}">
    <span class="st">${!it.exists ? "∅" : it.flag ? "⚑" : it.coded ? "✓" : "·"}</span>
    <span>${it.vid}</span><span class="meta">${it.coded ? (it.codes.join("+") || "clean") : ""}</span></li>`).join("");
  const cur = $("list").querySelector(".cur"); if (cur) cur.scrollIntoView({ block: "nearest" });
}

function renderPanel() {
  $("codes").querySelectorAll("button").forEach(b => {
    const k = state.codes.indexOf(b.dataset.c);
    b.classList.toggle("on", k >= 0);
    b.querySelector(".ord").textContent = k >= 0 ? k + 1 : "";
  });
  $("seq").textContent = state.codes.length ? state.codes.join(" → ") : (items[idx]?.success ? "No codes = clean success" : "No codes yet");
  $("attempts").querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.a === state.attempts));
  $("flag").classList.toggle("on", state.flag);
  if (document.activeElement !== $("comment")) $("comment").value = state.comment;
}

function load(i) {
  idx = Math.max(0, Math.min(items.length - 1, i));
  const it = items[idx];
  state = it.coded ? { codes: [...it.codes], attempts: it.attempts, flag: it.flag, comment: it.comment } : { codes: [], attempts: "1", flag: false, comment: "" };
  warned = ""; $("warn").style.display = "none";
  $("vid").textContent = it.vid;
  $("outcome").textContent = it.success ? "✓ Success (simulator)" : "✕ Failure (simulator)";
  $("outcome").className = "badge " + (it.success ? "ok" : "bad");
  $("reveal").textContent = it.coded && it.meta ? metaText(it.meta) : "Run hidden until saved";
  $("missing").hidden = it.exists;
  video.hidden = !it.exists;
  if (it.exists) { video.src = `/video/${it.vid}.mp4`; video.playbackRate = speed(); video.play().catch(() => {}); }
  else video.removeAttribute("src");
  renderPanel(); renderList();
}

function speed() { return parseFloat(document.querySelector(".spd.on").dataset.s); }
function setSpeed(s) { document.querySelectorAll(".spd").forEach(b => b.classList.toggle("on", b.dataset.s === String(s))); video.playbackRate = s; }
function toggleCode(c) { const k = state.codes.indexOf(c); if (k >= 0) state.codes.splice(k, 1); else state.codes.push(c); warned = ""; renderPanel(); }
function step(sec) { video.pause(); video.currentTime = Math.max(0, Math.min(video.duration || 0, video.currentTime + sec)); }

function checks() {
  const it = items[idx], c = state.codes, a = state.attempts;
  const miss = c.includes("G-MISS") || c.includes("G-SLIP");
  if (!it.success && !c.length) return "This is a failure but has no code. Add at least one (OTHER + comment if unsure).";
  if (it.success && miss && a === "1") return "Success after a miss/drop needs a re-grasp: set attempts to 2 or 3+.";
  if (it.success && a !== "1" && !miss) return "Success with several attempts: add G-MISS or G-SLIP for the first miss, so it counts as a recovery.";
  if (c.includes("RETRY-FAIL") && a === "1") return "RETRY-FAIL with 1 attempt: set attempts to 2 or 3+.";
  if (c.includes("NO-RETRY") && a !== "1") return "NO-RETRY with several attempts: check the attempts count.";
  if (c.includes("NO-RETRY") && c.includes("RETRY-FAIL")) return "NO-RETRY and RETRY-FAIL together: keep only one.";
  if (c.includes("OTHER") && !state.comment.trim()) return "OTHER needs a short comment.";
  return "";
}

async function save() {
  state.comment = $("comment").value;
  const w = checks();
  if (w && warned !== w) { warned = w; $("warn").textContent = w + " (Enter again to save anyway)"; $("warn").style.display = "block"; return; }
  const it = items[idx];
  const r = await fetch("/api/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ vid: it.vid, ...state }) });
  const j = await r.json();
  if (!j.ok) { toast("Not saved: " + j.error); return; }
  Object.assign(it, { coded: true, codes: [...state.codes], attempts: state.attempts, flag: state.flag, comment: state.comment, meta: j.meta });
  toast(`${it.vid} saved · ${metaText(j.meta)}`);
  const order = items.map((_, k) => (idx + 1 + k) % items.length);
  const next = order.find(k => !items[k].coded);
  load(next === undefined ? idx : next);
  if (next === undefined) toast("All videos coded. notes/failure_codes.csv is complete.");
}

function init() {
  $("codes").innerHTML = CODES.map(([c, d], k) => `<button data-c="${c}" title="${d}"><span class="k">${k + 1}</span>${c}<span class="ord"></span></button>`).join("");
  $("attempts").innerHTML = ATTEMPTS.map((a, k) => `<button data-a="${a}" style="flex:1"><span class="k">${"QWE"[k]}</span>${a}</button>`).join("");
  $("codes").addEventListener("click", e => { const b = e.target.closest("button"); if (b) toggleCode(b.dataset.c); });
  $("codes").addEventListener("mouseover", e => { const b = e.target.closest("button"); if (b) $("def").textContent = b.title; });
  $("attempts").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { state.attempts = b.dataset.a; warned = ""; renderPanel(); } });
  $("flag").onclick = () => { state.flag = !state.flag; renderPanel(); };
  $("save").onclick = save;
  $("play").onclick = () => video.paused ? video.play() : video.pause();
  $("back").onclick = () => step(-1 / FPS);
  $("fwd").onclick = () => step(1 / FPS);
  $("restart").onclick = () => { video.currentTime = 0; video.play(); };
  document.querySelectorAll(".spd").forEach(b => b.onclick = () => setSpeed(parseFloat(b.dataset.s)));
  document.querySelectorAll(".filters button").forEach(b => b.onclick = () => { filter = b.dataset.f; document.querySelectorAll(".filters button").forEach(x => x.classList.toggle("on", x === b)); renderList(); });
  $("list").addEventListener("click", e => { const li = e.target.closest("li"); if (li) load(+li.dataset.i); });
  $("comment").addEventListener("input", e => { state.comment = e.target.value; warned = ""; });
  video.addEventListener("timeupdate", () => { $("time").textContent = `frame ${Math.round(video.currentTime * FPS)} / ${Math.round((video.duration || 0) * FPS)}`; });
  video.addEventListener("ended", () => { video.currentTime = 0; video.play(); });  // loop
  video.addEventListener("play", () => $("play").innerHTML = '<span class="k">Space</span>Pause');
  video.addEventListener("pause", () => $("play").innerHTML = '<span class="k">Space</span>Play');
  document.addEventListener("keydown", e => {
    if (e.target === $("comment")) {
      if (e.key === "Escape") $("comment").blur();
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $("comment").blur(); save(); }
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (/^[1-9]$/.test(k)) toggleCode(CODES[+k - 1][0]);
    else if ("qwe".includes(k.toLowerCase()) && k.length === 1) { state.attempts = ATTEMPTS["qwe".indexOf(k.toLowerCase())]; warned = ""; renderPanel(); }
    else if (k === "f" || k === "F") { state.flag = !state.flag; renderPanel(); }
    else if (k === " ") { e.preventDefault(); video.paused ? video.play() : video.pause(); }
    else if (k === "ArrowLeft") { e.preventDefault(); step((e.shiftKey ? -10 : -1) / FPS); }
    else if (k === "ArrowRight") { e.preventDefault(); step((e.shiftKey ? 10 : 1) / FPS); }
    else if (k === "s" || k === "S") { const s = [0.125, 0.25, 0.5, 1]; setSpeed(s[(s.indexOf(speed()) + 1) % s.length]); }
    else if (k === "r" || k === "R") { video.currentTime = 0; video.play(); }
    else if (k === "c" || k === "C") { e.preventDefault(); $("comment").focus(); }
    else if (k === "[") load(idx - 1);
    else if (k === "]") load(idx + 1);
    else if (k === "Enter") { e.preventDefault(); save(); }
  });
  fetch("/api/items").then(r => r.json()).then(j => {
    items = j;
    const first = items.findIndex(i => !i.coded);
    load(first < 0 ? 0 : first);
  });
}
init();
</script>
</body>
</html>
"""


def main():
    global STORE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    STORE = Store()
    items = STORE.items()
    missing = sum(not i["exists"] for i in items)
    print(f"{len(items)} videos in the sample, {sum(i['coded'] for i in items)} already coded, {missing} missing on disk")
    print(f"open http://localhost:{a.port}  (codes are saved to {CODES_CSV.relative_to(ROOT)}; Ctrl-C to stop)")
    try:
        server = ThreadingHTTPServer((a.host, a.port), Handler)
    except OSError as e:
        raise SystemExit(f"cannot listen on port {a.port} ({e.strerror}): is the page already running? "
                         f"Use it, or stop it, or pass --port {a.port + 1}")
    server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
