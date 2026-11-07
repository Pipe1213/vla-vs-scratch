#!/usr/bin/env python3
"""Interactive play: type an instruction in a local page, watch a policy drive the LIBERO arm live.

Side tool, not part of the experiment. It loads a policy exactly like lerobot-eval does (same policy factory,
pre/post-processors and LIBERO environment factory, same settings as our evaluation: 256 px, re-plan every 10
actions) and runs one episode at a time. The live view and the controls are a page in your browser, like
code_videos.py (LeRobot installs the headless OpenCV, which cannot open windows, so we stream JPEG frames instead).

    conda activate lerobot && export MUJOCO_GL=egl
    python scripts/play.py                                    # then open http://localhost:8770 in the Windows browser
    python scripts/play.py --scene 6                          # start in the butter scene
    python scripts/play.py --policy.path=Pipe1213/smolvla_obj_t2_n10        # one of our runs (rename map added for you)
    python scripts/play.py --policy.path=Pipe1213/act_obj_t2_nall --scene 6
    python scripts/play.py --record outputs/play              # also save each finished episode as an mp4 (for a GIF or a post)
    python scripts/play.py --host 0.0.0.0                     # only if Windows cannot reach localhost

Default model: HuggingFaceVLA/smolvla_libero (trained on all LIBERO tasks, so it can follow different typed
instructions). Every flag this script does not know is passed to LeRobot's eval config (draccus), with the same
meaning as for lerobot-eval, e.g. --policy.n_action_steps=1 or --policy.device=cpu.

In the page
    instruction box + Run     run one episode (empty box = the scene's own instruction)
    scene / start             LIBERO-Object scene 0-9 (ours: 3 bbq sauce, 6 butter, 8 chocolate pudding),
                              starting state 0-49 (moves to the next one after each finished episode)
    keys (outside the box)    Space pause/resume, r restart (same start), n next start, s stop

Things to know
- "Success" is the simulator's check for the scene's own task. If you type another instruction (e.g. "pick up the
  ketchup and place it in the basket" in the bbq sauce scene), success still means the scene's task was done, so
  judge other instructions by eye.
- ACT ignores language, and our SmolVLA runs were each trained on one task, so they mostly do their own task
  whatever you type. Watching that happen is itself a useful check.
- The view shows the front camera (left) and the wrist camera (right), turned upright for display only. The policy
  gets exactly the images lerobot-eval gives it.
- Needs internet the first time a Hub model is used (download); afterwards HF_HUB_OFFLINE=1 works.
Ctrl-C in the terminal to stop.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from contextlib import nullcontext
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

# Same settings as configs/runs.yaml (evaluation.fixed_args + n_action_steps). Each can be overridden on the
# command line with the same flag.
DEFAULTS = {
    "--policy.path": "HuggingFaceVLA/smolvla_libero",
    "--env.type": "libero",
    "--env.task": "libero_object",
    "--env.init_states": "true",
    "--env.observation_height": "256",
    "--env.observation_width": "256",
    "--eval.batch_size": "1",
    "--policy.n_action_steps": "10",
}
# Our SmolVLA runs were fine-tuned from smolvla_base with these camera names (runs.yaml, methods.smolvla).
OUR_SMOLVLA_RENAME = ('{"observation.images.image": "observation.images.camera1", '
                      '"observation.images.image2": "observation.images.camera2"}')
N_STARTS = 50


def split_argv(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Separate this script's own flags from the LeRobot (draccus) flags, and add our evaluation defaults."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--scene", type=int, default=3, help="LIBERO-Object scene = sim task id (0-9)")
    ap.add_argument("--start", type=int, default=0, help="first starting state (0-49)")
    ap.add_argument("--scale", type=int, default=2, help="display zoom of the 256 px images")
    ap.add_argument("--fps", type=float, default=0, help="cap the speed at this many steps/s (0 = as fast as it runs)")
    ap.add_argument("--record", type=Path, default=None, help="folder where each episode is saved as an mp4")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("-h", "--help", action="store_true")
    own, rest = ap.parse_known_args(argv)
    if own.help:
        print(__doc__)
        sys.exit(0)
    given = {a.split("=", 1)[0] for a in rest if a.startswith("--")}
    for flag, value in DEFAULTS.items():
        if flag == "--policy.path" and "--policy.type" in given:
            continue  # --policy.type=<name> without a path = random weights (debugging only)
        if flag not in given:
            rest.append(f"{flag}={value}")
    policy_path = next((a.split("=", 1)[1] for a in rest if a.startswith("--policy.path=")), "")
    if "smolvla_obj" in policy_path and "--rename_map" not in given:
        rest.append(f"--rename_map={OUR_SMOLVLA_RENAME}")
        print("[play] one of our SmolVLA runs: adding the camera rename map from runs.yaml")
    rest = [a for a in rest if not a.startswith("--env.task_ids=")]  # the scene is chosen with --scene
    rest.append(f"--env.task_ids=[{own.scene}]")
    return own, rest


def success_from_info(info: dict) -> bool:
    """is_success from a 1-env vector env step (same cases as lerobot_eval.rollout)."""
    for src in (info.get("final_info"), info):
        if isinstance(src, dict) and "is_success" in src:
            return bool(np.asarray(src["is_success"]).reshape(-1)[0])
    return False


class Shared:
    """State shared between the simulation (main thread) and the web server threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = b""
        self.frame_id = 0
        self.state: dict = {}
        self.commands: queue.Queue = queue.Queue()

    def set_frame(self, img_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self.lock:
                self.jpeg, self.frame_id = buf.tobytes(), self.frame_id + 1

    def update(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)


class Player:
    def __init__(self, cfg, own: argparse.Namespace, shared: Shared):
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.random_utils import set_seed

        self.cfg, self.own, self.shared = cfg, own, shared
        self.device = torch.device(str(cfg.policy.device))
        set_seed(cfg.seed)
        self.env = None
        self.next_start = own.start % N_STARTS
        self.history: list[dict] = []
        self.instructions = self.scene_instructions()
        self.build_env(own.scene)  # make_policy needs the env config
        print(f"[play] loading policy {cfg.policy.pretrained_path} ...")
        self.policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
        self.publish("idle", f"ready: {cfg.policy.pretrained_path}")

    # -- environment -------------------------------------------------------------------------------------
    def scene_instructions(self) -> list[str]:
        from libero.libero import benchmark  # installed with lerobot[libero]
        suite = benchmark.get_benchmark_dict()[self.cfg.env.task]()
        return [suite.get_task(i).language for i in range(len(suite.tasks))]

    def build_env(self, scene: int) -> None:
        from lerobot.envs import close_envs, make_env
        if self.env is not None:
            close_envs({"s": {0: self.env}})
        self.cfg.env.task_ids = [scene]
        envs = make_env(self.cfg.env, n_envs=1, use_async_envs=False)
        self.env = next(iter(next(iter(envs.values())).values()))
        self.scene = scene
        self.scene_instruction = self.instructions[scene]
        print(f"[play] scene {scene}: \"{self.scene_instruction}\"")
        self.show_start()

    def reset(self, start: int):
        from lerobot.envs.utils import NEW_ROLLOUT_OPTION
        for e in self.env.envs:                      # LiberoEnv picks its starting state from init_state_id
            e.unwrapped.init_state_id = start
        raw, _ = self.env.reset(seed=[self.cfg.seed], options={NEW_ROLLOUT_OPTION: True})
        return raw

    def show_start(self) -> None:
        """Show the starting scene before anything runs."""
        raw = self.reset(self.next_start)
        self.shared.set_frame(self.frame(raw, self.scene_instruction, f"scene {self.scene} | start {self.next_start} | ready"))

    def publish(self, mode: str, message: str = "", **kw) -> None:
        self.shared.update(mode=mode, message=message, scene=self.scene, start=self.next_start,
                           scene_instruction=self.scene_instruction, instructions=self.instructions,
                           policy=str(self.cfg.policy.pretrained_path), history=self.history[-15:], **kw)

    # -- one episode -------------------------------------------------------------------------------------
    def frame(self, raw_obs: dict, text: str, status: str) -> np.ndarray:
        cams = [np.ascontiguousarray(img[0][::-1, ::-1]) for img in raw_obs["pixels"].values()]  # upright for display
        img = np.concatenate(cams, axis=1)
        img = cv2.resize(img, None, fx=self.own.scale, fy=self.own.scale, interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bar = np.zeros((48, img.shape[1], 3), np.uint8)
        cv2.putText(bar, text[:95], (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, status, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
        return np.concatenate([bar, img], axis=0)

    def episode(self, instruction: str) -> str:
        """Run one episode. Returns 'done', 'restart', 'next' or 'stop'."""
        from lerobot.utils.constants import ACTION
        from lerobot.envs import preprocess_observation

        start = self.next_start
        self.policy.reset(); self.pre.reset(); self.post.reset()
        raw = self.reset(start)
        max_steps = self.env.call("_max_episode_steps")[0]
        frames: list[np.ndarray] = []
        paused, step, success, result, t0 = False, 0, False, "done", time.time()
        amp = torch.autocast(device_type=self.device.type) if self.cfg.policy.use_amp else nullcontext()
        while step < max_steps:
            status = f"scene {self.scene} | start {start} | step {step}/{max_steps}" + (" | PAUSED" if paused else "")
            img = self.frame(raw, instruction, status)
            self.shared.set_frame(img)
            if self.own.record and not paused:
                frames.append(img)
            self.publish("paused" if paused else "running", status, instruction=instruction)
            try:
                cmd = self.shared.commands.get(timeout=0.05) if paused else self.shared.commands.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd:
                c = cmd.get("cmd")
                if c == "pause":
                    paused = not paused
                elif c in ("restart", "next", "stop"):
                    result = c
                    break
            if paused:
                continue
            # Same order as lerobot_eval.rollout(): env format -> LeRobot keys -> task text -> env processor ->
            # policy processor -> policy -> postprocessor -> env postprocessor.
            obs = preprocess_observation(raw)
            obs["task"] = [instruction]
            obs = self.pre(self.env_pre(obs))
            with torch.inference_mode(), amp:
                action = self.policy.select_action(obs)
            action = self.env_post({ACTION: self.post(action)})[ACTION]
            raw, _, terminated, truncated, info = self.env.step(action.to("cpu").numpy())
            step += 1
            success = success_from_info(info)
            if success or terminated[0] or truncated[0]:
                break
            if self.own.fps:
                time.sleep(max(0.0, step / self.own.fps - (time.time() - t0)))
        if result == "done":
            verdict = "scene task done" if success else "scene task not done (time limit)"
        else:
            verdict = {"restart": "restarted", "next": "skipped", "stop": "stopped"}[result]
        secs = time.time() - t0
        print(f"[play] scene {self.scene} start {start}: {verdict} after {step} steps ({secs:.0f} s) | {instruction}")
        final = self.frame(raw, instruction, f"scene {self.scene} | start {start} | {verdict} | {step} steps")
        self.shared.set_frame(final)
        self.history.append({"scene": self.scene, "start": start, "instruction": instruction,
                             "result": verdict, "steps": step, "seconds": round(secs)})
        if self.own.record and frames and result == "done":  # restarted / skipped / stopped episodes are not saved
            self.save_video(frames + [final] * 20, start)
        if result in ("done", "next"):
            self.next_start = (start + 1) % N_STARTS
        return result

    def save_video(self, frames: list[np.ndarray], start: int) -> None:
        self.own.record.mkdir(parents=True, exist_ok=True)
        out = self.own.record / f"{time.strftime('%Y%m%d_%H%M%S')}_scene{self.scene}_start{start}.mp4"
        h, w = frames[0].shape[:2]
        fps = 20  # LIBERO runs at 80 Hz (render_fps 80): 20 fps = quarter speed, like our GIFs
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"[play] saved {out}")

    # -- command loop (main thread) ------------------------------------------------------------------------
    def run(self) -> None:
        while True:
            cmd = self.shared.commands.get()
            c = cmd.get("cmd")
            try:
                if c == "run":
                    instruction = (cmd.get("text") or "").strip() or self.scene_instruction
                    while self.episode(instruction) in ("restart", "next"):
                        pass  # restart keeps the start state, next moves on; same instruction
                    self.publish("idle", "ready")
                elif c == "scene":
                    self.publish("busy", "loading scene ...")
                    self.build_env(int(cmd["value"]) % len(self.instructions))
                    self.publish("idle", "ready")
                elif c == "start":
                    self.next_start = int(cmd["value"]) % N_STARTS
                    self.show_start()
                    self.publish("idle", "ready")
            except Exception as e:  # keep the page alive; the traceback is in the terminal
                import traceback
                traceback.print_exc()
                self.publish("idle", f"error: {e}")


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>LIBERO play</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f5f9;--fg:#1d1b24;--mut:#6b6778;--line:#dcd8e6;--acc:#6d4fc2;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#16151b;--fg:#ece9f4;--mut:#9a95aa;--line:#2f2c38;--acc:#a58cf0;--card:#1f1d26}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:18px;margin:0 0 4px}.mut{color:var(--mut);font-size:13px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px;margin-top:12px}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
img{width:100%;image-rendering:pixelated;border-radius:6px;background:#000;display:block}
input,select,button{font:inherit;border:1px solid var(--line);border-radius:6px;padding:7px 9px;background:var(--card);color:var(--fg)}
input[type=text]{width:100%}button{cursor:pointer}button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
.row{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;align-items:center}
#status{font-weight:600}kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px}td{border-top:1px solid var(--line);padding:4px 2px;vertical-align:top}
.ok{color:#2e8b57}.no{color:#c0504d}
</style></head><body><main>
<h1>LIBERO play</h1><div class="mut" id="policy"></div>
<div class="grid"><div>
<img id="view" src="/stream" alt="Live view: front camera left, wrist camera right">
<div class="row"><span id="status">connecting</span></div>
<form id="f" class="row" style="flex-wrap:nowrap">
<input type="text" id="text" placeholder="Instruction (empty = the scene's own instruction)" autocomplete="off">
<button class="primary" type="submit">Run</button></form>
<div class="row"><button data-c="pause">Pause <kbd>Space</kbd></button><button data-c="restart">Restart <kbd>r</kbd></button>
<button data-c="next">Next start <kbd>n</kbd></button><button data-c="stop">Stop <kbd>s</kbd></button></div>
</div><div>
<div class="card"><label>Scene<select id="scene" style="display:block;width:100%;margin-top:4px"></select></label>
<div class="row">
<label>Start <input type="number" id="start" min="0" max="49" style="width:70px"></label>
<button id="setstart">Set</button></div>
<p class="mut" id="sceneinstr"></p>
<p class="mut">Success = the scene's own task is done, whatever you typed.</p></div>
<div class="card" style="margin-top:12px"><b>Episodes</b><table id="hist"></table></div>
</div></div></main>
<script>
const $=id=>document.getElementById(id);
const send=o=>fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)});
$('f').onsubmit=e=>{e.preventDefault();send({cmd:'run',text:$('text').value});};
document.querySelectorAll('[data-c]').forEach(b=>b.onclick=()=>send({cmd:b.dataset.c}));
$('scene').onchange=()=>send({cmd:'scene',value:+$('scene').value});
$('setstart').onclick=()=>send({cmd:'start',value:+$('start').value});
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
 const m={' ':'pause','r':'restart','n':'next','s':'stop'}[e.key];if(m){e.preventDefault();send({cmd:m});}});
let built=false;
async function poll(){try{const s=await (await fetch('/state')).json();
 if(!built&&s.instructions){$('scene').innerHTML=s.instructions.map((t,i)=>`<option value="${i}">${i}: ${t}</option>`).join('');built=true;}
 if(document.activeElement!==$('scene'))$('scene').value=s.scene;
 if(document.activeElement!==$('start'))$('start').value=s.start;
 $('status').textContent=(s.mode||'')+(s.message?' | '+s.message:'');
 $('policy').textContent='Policy: '+(s.policy||'');$('sceneinstr').textContent='Scene task: '+(s.scene_instruction||'');
 const busy=s.mode==='running'||s.mode==='paused'||s.mode==='busy';
 document.querySelectorAll('#f button,#scene,#setstart').forEach(x=>x.disabled=busy);
 $('hist').innerHTML=(s.history||[]).slice().reverse().map(h=>`<tr><td>${h.scene}/${h.start}</td><td>${h.instruction.replace(/</g,'&lt;')}</td><td class="${h.result==='scene task done'?'ok':'no'}">${h.result}</td></tr>`).join('');
}catch(e){$('status').textContent='disconnected';}setTimeout(poll,400);}
poll();
</script></body></html>"""


def make_handler(shared: Shared):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the terminal for episode results
            pass

        def _send(self, body: bytes, ctype: str, code=HTTPStatus.OK):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/state":
                with shared.lock:
                    body = json.dumps(shared.state).encode()
                self._send(body, "application/json")
            elif self.path.startswith("/stream"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last, sent = -1, 0.0
                try:
                    while True:
                        with shared.lock:
                            fid, jpg = shared.frame_id, shared.jpeg
                        # Browsers paint a part of the stream when the next one starts, so a still picture is
                        # re-sent a few times per second (otherwise the last frame would never show).
                        if jpg and (fid != last or time.time() - sent > 0.25):
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                            last, sent = fid, time.time()
                        time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/cmd":
                return self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            try:
                cmd = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self._send(b"bad json", "text/plain", HTTPStatus.BAD_REQUEST)
            shared.commands.put(cmd)
            self._send(b"{}", "application/json")

    return Handler


def main() -> None:
    own, rest = split_argv(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]  # draccus (and EvalPipelineConfig's policy path lookup) read sys.argv

    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig

    shared = Shared()
    try:
        server = ThreadingHTTPServer((own.host, own.port), make_handler(shared))
    except OSError:
        sys.exit(f"Port {own.port} is in use: stop the other program or pass --port {own.port + 1}")
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

    @parser.wrap()
    def _start(cfg: EvalPipelineConfig) -> None:
        player = Player(cfg, own, shared)
        print(f"[play] open http://localhost:{own.port} in your browser (Ctrl-C here to stop)")
        player.run()

    try:
        _start()
    except KeyboardInterrupt:
        print("\n[play] stopped")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
