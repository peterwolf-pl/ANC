#!/usr/bin/env python3
# UI: Sender + PNG->ACODE generator + ACODE preview + simulator overlay
# Tabs:
# - Printer: sender + jog + pen + steppers
# - Generator: PNG -> artacodepng.py -> .acode + preview + simulator
# - Text outline: text -> outline PNG -> artacodepng.py -> .acode + preview + simulator
#
# Simulator:
# - draws trajectory from ACODE (diff drive)
# - overlays sprite static/maszyna.png scaled to wheelbase_mm (with extra UI scale multiplier)

import os
import time
import uuid
import json
import math
import socket
import threading
import queue
import subprocess
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from io import BytesIO

from flask import Flask, request, render_template, Response, jsonify, send_file, abort

app = Flask(__name__, static_folder="static", static_url_path="/static")

DEFAULT_PORT = 3333
RECV_TIMEOUT_S = 0.25
LINE_ACK_TIMEOUT_S = 60.0

HERE = os.path.dirname(os.path.abspath(__file__))
ARTACODEPNG_PATH = os.path.join(HERE, "artacodepng.py")
ACODEVIZ_PATH = os.path.join(HERE, "acodeviz.py")
ACODE_PY_PATH = os.path.join(HERE, "acode.py")

WORKDIR = os.path.join(HERE, "_ui_work")
UPLOADS_DIR = os.path.join(WORKDIR, "uploads")
GEN_DIR = os.path.join(WORKDIR, "generated")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(os.path.join(HERE, "static"), exist_ok=True)

events = queue.Queue(maxsize=2000)

def push_event(kind: str, payload: dict):
    msg = {"kind": kind, "payload": payload, "ts": time.time()}
    try:
        events.put_nowait(msg)
    except queue.Full:
        try:
            events.get_nowait()
        except queue.Empty:
            pass
        try:
            events.put_nowait(msg)
        except queue.Full:
            pass

W_RE = re.compile(r"^W\s+L(-?\d+)\s+R(-?\d+)\s+F(\d+)\s*$")
P_RE = re.compile(r"^P\s+([UD])\s*$")

ESTIMATE_DEFAULTS = {
    "feed_to_sps": 0.6,
    "min_sps": 50.0,
    "max_sps": 2500.0,
    "servo_settle_s": 0.18,
    "start_sps": 90.0,
    "accel_sps2": 4000.0,
}

def estimate_motion_seconds(
    line: str,
    feed_to_sps: float,
    min_sps: float,
    max_sps: float,
    servo_settle_s: float,
    start_sps: float,
    accel_sps2: float,
) -> float:
    if line == "H":
        return 0.25
    if line == "END":
        return 0.25
    if line.startswith("E"):
        return 0.25

    if P_RE.match(line):
        return servo_settle_s + 0.25

    m = W_RE.match(line)
    if not m:
        return 0.6

    l = abs(int(m.group(1)))
    r = abs(int(m.group(2)))
    f = int(m.group(3))
    steps = max(l, r)
    if steps <= 0:
        return 0.25

    v_target = f * feed_to_sps
    if v_target < min_sps:
        v_target = min_sps
    if v_target > max_sps:
        v_target = max_sps

    v0 = start_sps
    if v0 < min_sps:
        v0 = min_sps
    if v0 > v_target:
        v0 = v_target

    a = accel_sps2
    if a < 1.0:
        a = 1.0

    ramp_steps = (v_target * v_target - v0 * v0) / (2.0 * a)
    if ramp_steps < 0.0:
        ramp_steps = 0.0

    half = steps / 2.0
    if ramp_steps > half:
        ramp_steps = half

    t_ramp = (v_target - v0) / a
    if t_ramp < 0.0:
        t_ramp = 0.0

    cruise_steps = steps - 2.0 * ramp_steps
    if cruise_steps < 0.0:
        cruise_steps = 0.0

    t_cruise = cruise_steps / v_target if v_target > 1e-9 else 9999.0

    return (2.0 * t_ramp) + t_cruise + 0.30

def estimate_total_duration(lines: List[str], params: Dict[str, float]) -> float:
    total = 0.0
    for ln in lines:
        total += estimate_motion_seconds(
            line=ln,
            feed_to_sps=params.get("feed_to_sps", ESTIMATE_DEFAULTS["feed_to_sps"]),
            min_sps=params.get("min_sps", ESTIMATE_DEFAULTS["min_sps"]),
            max_sps=params.get("max_sps", ESTIMATE_DEFAULTS["max_sps"]),
            servo_settle_s=params.get("servo_settle_s", ESTIMATE_DEFAULTS["servo_settle_s"]),
            start_sps=params.get("start_sps", ESTIMATE_DEFAULTS["start_sps"]),
            accel_sps2=params.get("accel_sps2", ESTIMATE_DEFAULTS["accel_sps2"]),
        )
    return total

@dataclass
class JobState:
    running: bool = False
    paused: bool = False
    stopping: bool = False
    host: str = "192.168.4.1"
    port: int = DEFAULT_PORT
    lines: Optional[List[str]] = None
    idx: int = 0
    last_sent: str = ""
    last_ok: bool = False
    last_ok_idx: int = 0
    last_ok_line: str = ""
    error: str = ""

state = JobState()
state_lock = threading.Lock()
worker_thread: Optional[threading.Thread] = None

@dataclass
class GenState:
    gen_id: str = ""
    png_path: str = ""
    acode_path: str = ""
    preview_path: str = ""
    acode_text: str = ""
    meta: Dict[str, Any] = None  # type: ignore[assignment]
    error: str = ""
    viz_settings: Dict[str, Any] = None  # type: ignore[assignment]

gen_state = GenState(meta={}, viz_settings={})
gen_lock = threading.Lock()

def recv_line(sock: socket.socket, timeout_s: float) -> Optional[str]:
    sock.settimeout(timeout_s)
    buf = bytearray()
    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            return None
        try:
            b = sock.recv(1)
        except socket.timeout:
            return None
        if not b:
            return None
        if b == b"\n":
            return buf.decode("utf-8", errors="replace").strip()
        if b != b"\r":
            buf.extend(b)

def send_line(sock: socket.socket, line: str):
    data = (line.strip() + "\n").encode("utf-8")
    sock.sendall(data)

def _parse_state_line(line: str) -> Optional[Dict[str, str]]:
    if not line.startswith("STATE"):
        return None
    parts = line.split(maxsplit=2)
    status = parts[1] if len(parts) > 1 else ""
    detail = parts[2] if len(parts) > 2 else ""
    return {"state": status or "?", "detail": detail}

def _handle_sideband(line: str) -> bool:
    st = _parse_state_line(line)
    if st:
        push_event("machine_state", st)
        return True
    return False

def wait_ok(sock: socket.socket, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = recv_line(sock, RECV_TIMEOUT_S)
        if not resp:
            continue
        s = resp.strip()
        if _handle_sideband(s):
            continue
        if s.startswith("OK"):
            return True
        if s.startswith("ERR"):
            with state_lock:
                state.error = s
            push_event("error", {"msg": s})
            return False
    return False

def send_one_command(host: str, port: int, line: str, timeout_s: float = LINE_ACK_TIMEOUT_S) -> Dict[str, Any]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        push_event("status", {"msg": "connected"})
        send_line(sock, line)
        ok = wait_ok(sock, timeout_s)
        return {"ok": ok, "line": line, "error": None if ok else "timeout or ERR"}
    except OSError as e:
        return {"ok": False, "line": line, "error": str(e)}
    finally:
        try:
            sock.close()
        except Exception:
            pass
        push_event("status", {"msg": "disconnected"})

def sender_worker():
    with state_lock:
        host = state.host
        port = state.port
        lines = state.lines or []
        state.last_sent = ""
        state.last_ok = False
        state.error = ""
        state.stopping = False

    push_event("status", {"msg": "connecting", "host": host, "port": port})

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        push_event("status", {"msg": "connected"})

        while True:
            with state_lock:
                if state.stopping:
                    break
                idx = state.idx
                running = state.running
                paused = state.paused
                err = state.error
                total = len(lines)

            if err:
                break

            if not running or paused:
                time.sleep(0.05)
                continue

            if idx >= total:
                push_event("done", {"msg": "finished"})
                with state_lock:
                    state.running = False
                break

            line = lines[idx].strip()
            if not line:
                with state_lock:
                    state.idx += 1
                continue

            with state_lock:
                state.last_sent = line
                state.last_ok = False

            push_event("line", {"idx": idx + 1, "total": total, "line": line})

            try:
                send_line(sock, line)
            except OSError as e:
                with state_lock:
                    state.error = f"send failed: {e}"
                push_event("error", {"msg": state.error})
                break

            ok = wait_ok(sock, LINE_ACK_TIMEOUT_S)
            with state_lock:
                state.last_ok = ok
                if ok:
                    state.last_ok_idx = idx + 1
                    state.last_ok_line = line

            if not ok:
                with state_lock:
                    if not state.error:
                        state.error = "timeout waiting for OK"
                push_event("error", {"msg": state.error})
                break

            push_event("ok", {"idx": idx + 1, "line": line})
            with state_lock:
                state.idx += 1

        push_event("status", {"msg": "stopped"})
    except OSError as e:
        with state_lock:
            state.error = f"connection failed: {e}"
        push_event("error", {"msg": state.error})
        push_event("status", {"msg": "disconnected"})
    finally:
        try:
            sock.close()
        except Exception:
            pass
        with state_lock:
            state.running = False
            state.paused = False
            state.stopping = False
        push_event("status", {"msg": "stopped"})

def _safe_float(name: str, default: float) -> float:
    try:
        return float(request.form.get(name, "").strip() or default)
    except ValueError:
        return default

def _safe_int(name: str, default: int) -> int:
    try:
        return int(request.form.get(name, "").strip() or default)
    except ValueError:
        return default

def _safe_bool(name: str, default: bool = False) -> bool:
    raw = request.form.get(name, "")
    if raw is None or raw == "":
        return default
    v = (raw.strip() or "").lower()
    return v in ("1", "true", "yes", "on", "checked")

def _safe_choice(name: str, default: str, allowed: Tuple[str, ...]) -> str:
    v = (request.form.get(name, "").strip() or default)
    return v if v in allowed else default

def _parse_line_advance(field: str) -> str:
    v = _safe_choice(field, "soft", ("soft", "turn90", "real90", "default"))
    return "soft" if v == "default" else v

def _parse_y_order(field: str) -> str:
    return _safe_choice(field, "top-down", ("top-down", "bottom-up"))

def _assert_tools_exist() -> Optional[str]:
    for p in [ARTACODEPNG_PATH, ACODEVIZ_PATH, ACODE_PY_PATH]:
        if not os.path.isfile(p):
            return f"missing file: {os.path.basename(p)} in {HERE}"
    return None

def _run_cmd(cmd: List[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

_HELP_CACHE: Dict[str, str] = {}
def _script_help_text(path: str) -> str:
    if path in _HELP_CACHE:
        return _HELP_CACHE[path]
    p = _run_cmd(["python3", path, "--help"], cwd=HERE)
    txt = (p.stdout or "") + "\n" + (p.stderr or "")
    _HELP_CACHE[path] = txt
    return txt

def _supports_arg(script_path: str, arg: str) -> bool:
    txt = _script_help_text(script_path)
    if arg in txt:
        return True
    try:
        with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
            return arg in f.read()
    except OSError:
        return False

def _split_acode_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

# ----------------------------
# Settings parsing without importing acode.py
# ----------------------------
_NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
def _extract_number(text: str, names: List[str]) -> Optional[float]:
    for n in names:
        m = re.search(rf"^\s*{re.escape(n)}\s*=\s*({_NUM_RE})\s*(?:#.*)?$", text, flags=re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None

def resolve_machine_settings(acode_py_path: str) -> Dict[str, float]:
    # Fallbacks match acode.py defaults to keep simulator in sync even if file read fails
    wheelbase = 255.0
    spmm = (200 * 8) / 175.0  # 9.142857...
    turn_gain = 7400.0 / ((wheelbase * math.pi) * spmm)

    try:
        with open(acode_py_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return {"wheelbase_mm": wheelbase, "steps_per_mm": spmm, "turn_gain": turn_gain}

    w = _extract_number(txt, ["WHEELBASE_MM", "wheelbase_mm", "WHEELBASE"])
    s = _extract_number(txt, ["STEPS_PER_MM", "steps_per_mm", "SPMM", "STEPS_MM"])
    g = _extract_number(txt, ["TURN_GAIN", "turn_gain", "TURNGAIN"])

    if w is not None:
        wheelbase = w
    if s is not None:
        spmm = s
    if g is not None:
        turn_gain = g

    return {"wheelbase_mm": wheelbase, "steps_per_mm": spmm, "turn_gain": turn_gain}

# ----------------------------
# ACODE -> trajectory model (differential drive)
# Pose: x_mm, y_mm, theta_rad
# theta=0 means facing +X
# ----------------------------
def _integrate_step(
    x: float,
    y: float,
    th: float,
    dl_mm: float,
    dr_mm: float,
    wheelbase_mm: float,
    turn_gain: float
) -> Tuple[float, float, float]:
    d = 0.5 * (dl_mm + dr_mm)
    dth = 0.0
    if wheelbase_mm > 1e-9:
        dth = ((dr_mm - dl_mm) / wheelbase_mm) * turn_gain

    if abs(dth) < 1e-9:
        x2 = x + d * math.cos(th)
        y2 = y + d * math.sin(th)
        return x2, y2, th

    R = d / dth
    cx = x - R * math.sin(th)
    cy = y + R * math.cos(th)
    th2 = th + dth
    x2 = cx + R * math.sin(th2)
    y2 = cy - R * math.cos(th2)
    return x2, y2, th2

def acode_to_trajectory(
    lines: List[str],
    wheelbase_mm: float,
    steps_per_mm: float,
    turn_gain: float,
    arc_step_mm: float,
    arc_step_deg: float,
    home_resets_pose: bool
) -> Dict[str, Any]:
    x = 0.0
    y = 0.0
    th = 0.0
    pen_down = True
    pts: List[Dict[str, float]] = [{"x": x, "y": y, "theta": th, "pen_down": pen_down}]
    line_frames: List[int] = []

    xmin = xmax = x
    ymin = ymax = y

    def add_pt(px: float, py: float, pth: float, p_pen: bool):
        nonlocal xmin, xmax, ymin, ymax
        pts.append({"x": px, "y": py, "theta": pth, "pen_down": p_pen})
        xmin = min(xmin, px)
        xmax = max(xmax, px)
        ymin = min(ymin, py)
        ymax = max(ymax, py)

    def mark_line():
        line_frames.append(max(0, len(pts) - 1))

    arc_step_rad = max(1e-6, (arc_step_deg * math.pi / 180.0))
    arc_step_mm = max(1e-6, arc_step_mm)

    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        if s == "H" and home_resets_pose:
            x, y, th = 0.0, 0.0, 0.0
            add_pt(x, y, th, pen_down)
            mark_line()
            continue

        if s.startswith("W"):
            ml = re.search(r"\bL\s*(" + _NUM_RE + r")\b", s)
            mr = re.search(r"\bR\s*(" + _NUM_RE + r")\b", s)
            if not ml or not mr:
                mark_line()
                continue
            try:
                L = float(ml.group(1))
                R = float(mr.group(1))
            except ValueError:
                mark_line()
                continue

            if steps_per_mm <= 1e-9:
                mark_line()
                continue

            dl = L / steps_per_mm
            dr = R / steps_per_mm

            d = 0.5 * (dl + dr)
            dth = 0.0
            if wheelbase_mm > 1e-9:
                dth = ((dr - dl) / wheelbase_mm) * turn_gain

            n1 = int(abs(d) / arc_step_mm) if abs(d) > 0 else 0
            n2 = int(abs(dth) / arc_step_rad) if abs(dth) > 0 else 0
            n = max(1, n1, n2)

            dl_i = dl / n
            dr_i = dr / n

            for _ in range(n):
                x, y, th = _integrate_step(x, y, th, dl_i, dr_i, wheelbase_mm, turn_gain)
                add_pt(x, y, th, pen_down)

            mark_line()
            continue

        if s.startswith("P"):
            if "U" in s.upper():
                pen_down = False
            elif "D" in s.upper():
                pen_down = True
            add_pt(x, y, th, pen_down)
            mark_line()
            continue

        mark_line()

    if abs(xmax - xmin) < 1e-6:
        xmax += 1.0
        xmin -= 1.0
    if abs(ymax - ymin) < 1e-6:
        ymax += 1.0
        ymin -= 1.0

    return {
        "points": pts,
        "bounds": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
        "settings": {
            "wheelbase_mm": wheelbase_mm,
            "steps_per_mm": steps_per_mm,
            "turn_gain": turn_gain,
            "arc_step_mm": arc_step_mm,
            "arc_step_deg": arc_step_deg,
            "home_resets_pose": home_resets_pose,
        },
        "line_frames": line_frames,
    }

# ----------------------------
# Fonts + Text outline render
# ----------------------------
def list_system_fonts(limit: int = 250) -> List[Dict[str, str]]:
    try:
        from matplotlib import font_manager
    except Exception:
        return []

    items: List[Dict[str, str]] = []
    seen = set()

    for f in getattr(font_manager, "fontManager").ttflist:
        name = getattr(f, "name", "") or ""
        path = getattr(f, "fname", "") or ""
        if not name or not path:
            continue
        key = (name, path)
        if key in seen:
            continue
        seen.add(key)
        items.append({"name": name, "path": path})
        if len(items) >= limit:
            break

    def score(it: Dict[str, str]) -> int:
        n = it["name"].lower()
        if "dejavu" in n:
            return 0
        if "noto" in n:
            return 1
        return 2

    items.sort(key=score)
    return items

def render_text_outline_png(
    text: str,
    line_width_mm: float,
    letter_height_mm: float,
    font_path: str,
    stroke_mm: float = 0.35,
    dpi: int = 600,
    padding_mm: float = 2.0,
) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import PathPatch

    px_per_mm = dpi / 25.4
    pad_px = int(max(0, padding_mm * px_per_mm))
    width_px = int(max(60, line_width_mm * px_per_mm)) + 2 * pad_px

    fp = FontProperties(fname=font_path) if font_path else FontProperties()

    tp = TextPath((0, 0), text, size=1.0, prop=fp)
    bb = tp.get_extents()

    if bb.width <= 0 or bb.height <= 0:
        fig_w_in = width_px / dpi
        fig_h_in = max(30, pad_px * 2 + 30) / dpi
        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, width_px)
        ax.set_ylim(0, fig_h_in * dpi)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), width_px, fig_h_in * dpi, facecolor="white", edgecolor="none"))
        buf = BytesIO()
        fig.canvas.print_png(buf)
        plt.close(fig)
        return buf.getvalue()

    target_h_px = max(10.0, letter_height_mm * px_per_mm)

    scale_h = target_h_px / bb.height
    width_after_h = bb.width * scale_h
    target_w_px = max(10.0, line_width_mm * px_per_mm)

    scale_w = (target_w_px / width_after_h) if width_after_h > 0 else 1.0
    scale = scale_h * min(1.0, scale_w)

    bb2_h = bb.height * scale
    height_px = int(math.ceil(bb2_h + 2 * pad_px))
    height_px = max(height_px, pad_px * 2 + 40)

    fig_w_in = width_px / dpi
    fig_h_in = height_px / dpi

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(0, height_px)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 0), width_px, height_px, facecolor="white", edgecolor="none"))

    x0 = pad_px
    y0 = pad_px - (bb.y0 * scale)

    path = tp.transformed(matplotlib.transforms.Affine2D().scale(scale).translate(x0, y0))
    stroke_px = max(1.0, stroke_mm * px_per_mm)

    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor="black",
        lw=stroke_px,
        joinstyle="round",
        capstyle="round",
        antialiased=True,
    )
    ax.add_patch(patch)

    buf = BytesIO()
    fig.canvas.print_png(buf)
    plt.close(fig)
    return buf.getvalue()

# ----------------------------
# Generator pipeline
# ----------------------------
def generate_from_png(png_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    err = _assert_tools_exist()
    if err:
        return {"ok": False, "error": err}

    gen_id = str(uuid.uuid4())[:8]
    out_acode = os.path.join(GEN_DIR, f"{gen_id}.acode")
    out_preview = os.path.join(GEN_DIR, f"{gen_id}.preview.png")

    warn: List[str] = []

    cmd = [
        "python3",
        ARTACODEPNG_PATH,
        png_path,
        "--img-width-mm", str(params["img_width_mm"]),
        "-o", out_acode,
        "--work-width-px", str(params["work_width_px"]),
        "--line-spacing-mm", str(params["line_spacing_mm"]),
        "--threshold", str(params["threshold"]),
        "--gamma", str(params["gamma"]),
        "--min-segment-mm", str(params["min_segment_mm"]),
        "--margin-mm", str(params["margin_mm"]),
        "--y-order", params["y_order"],
        "--scan", params["scan"],
        "--y-jitter-mm", str(params["y_jitter_mm"]),
        "--x-mode", params["x_mode"],
        "--x-step-mm", str(params["x_step_mm"]),
        "--row-angle-deg", str(params["row_angle_deg"]),
        "--soft-min-dy-mm", str(params["soft_min_dy_mm"]),
        "--feed-lin", str(params["feed_lin"]),
        "--feed-turn", str(params["feed_turn"]),
    ]

    if params.get("invert"):
        cmd.append("--invert")
    if params.get("flip_y"):
        cmd.append("--flip-y")
    if params.get("seed") is not None:
        cmd += ["--seed", str(params["seed"])]

    if _supports_arg(ARTACODEPNG_PATH, "--line-advance"):
        cmd += ["--line-advance", params["line_advance"]]
    elif params["line_advance"] != "soft":
        warn.append("artacodepng.py does not support --line-advance (using soft)")

    p1 = _run_cmd(cmd, cwd=HERE)
    if p1.returncode != 0 or not os.path.isfile(out_acode):
        msg = (p1.stderr or p1.stdout or "").strip() or "generator failed"
        return {"ok": False, "error": msg}

    cmd2 = [
        "python3",
        ACODEVIZ_PATH,
        out_acode,
        "-o", out_preview,
        "--acode-py", ACODE_PY_PATH,
        "--dpi", str(params["viz_dpi"]),
        "--arc-step-mm", str(params["viz_arc_step_mm"]),
        "--arc-step-deg", str(params["viz_arc_step_deg"]),
    ]
    if params.get("viz_equal"):
        cmd2.append("--equal")
    if params.get("viz_invert_y"):
        cmd2.append("--invert-y")
    if params.get("viz_home_resets_pose"):
        cmd2.append("--home-resets-pose")

    p2 = _run_cmd(cmd2, cwd=HERE)
    if p2.returncode != 0 or not os.path.isfile(out_preview):
        msg = (p2.stderr or p2.stdout or "").strip() or "preview failed"
        return {"ok": False, "error": msg}

    with open(out_acode, "r", encoding="utf-8", errors="ignore") as f:
        acode_text = f.read()

    lines_list = _split_acode_lines(acode_text)
    duration_s = estimate_total_duration(lines_list, ESTIMATE_DEFAULTS)

    meta = {
        "gen_id": gen_id,
        "warnings": warn,
        "params": params,
        "acode_lines": len(lines_list),
        "generator_stdout": (p1.stdout or "").strip(),
        "generator_stderr": (p1.stderr or "").strip(),
        "viz_stdout": (p2.stdout or "").strip(),
        "viz_stderr": (p2.stderr or "").strip(),
        "duration_s": duration_s,
    }

    return {
        "ok": True,
        "gen_id": gen_id,
        "png_path": png_path,
        "acode_path": out_acode,
        "preview_path": out_preview,
        "acode_text": acode_text,
        "meta": meta,
        "duration_s": duration_s,
    }

def generate_from_dxf(dxf_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    err = _assert_tools_exist()
    if err:
        return {"ok": False, "error": err}

    gen_id = str(uuid.uuid4())[:8]
    out_acode = os.path.join(GEN_DIR, f"{gen_id}.acode")
    out_preview = os.path.join(GEN_DIR, f"{gen_id}.preview.png")

    cmd = [
        "python3",
        ACODE_PY_PATH,
        dxf_path,
        "-o",
        out_acode,
        "--feed-lin",
        str(params["feed_lin"]),
        "--feed-turn",
        str(params["feed_turn"]),
        "--feed-arc",
        str(params["feed_arc"]),
        "--flat-step",
        str(params["flat_step"]),
        "--epsilon",
        str(params["epsilon"]),
    ]

    if params.get("layer"):
        cmd += ["--layer", params["layer"]]
    if not params.get("reorder", True):
        cmd.append("--no-reorder")

    p1 = _run_cmd(cmd, cwd=HERE)
    if p1.returncode != 0 or not os.path.isfile(out_acode):
        msg = (p1.stderr or p1.stdout or "").strip() or "DXF generator failed"
        return {"ok": False, "error": msg}

    cmd2 = [
        "python3",
        ACODEVIZ_PATH,
        out_acode,
        "-o",
        out_preview,
        "--acode-py",
        ACODE_PY_PATH,
        "--dpi",
        str(params["viz_dpi"]),
        "--arc-step-mm",
        str(params["viz_arc_step_mm"]),
        "--arc-step-deg",
        str(params["viz_arc_step_deg"]),
    ]
    if params.get("viz_equal"):
        cmd2.append("--equal")
    if params.get("viz_invert_y"):
        cmd2.append("--invert-y")
    if params.get("viz_home_resets_pose"):
        cmd2.append("--home-resets-pose")

    p2 = _run_cmd(cmd2, cwd=HERE)
    if p2.returncode != 0 or not os.path.isfile(out_preview):
        msg = (p2.stderr or p2.stdout or "").strip() or "preview failed"
        return {"ok": False, "error": msg}

    with open(out_acode, "r", encoding="utf-8", errors="ignore") as f:
        acode_text = f.read()

    lines_list = _split_acode_lines(acode_text)
    duration_s = estimate_total_duration(lines_list, ESTIMATE_DEFAULTS)

    meta = {
        "gen_id": gen_id,
        "warnings": [],
        "params": params,
        "acode_lines": len(lines_list),
        "generator_stdout": (p1.stdout or "").strip(),
        "generator_stderr": (p1.stderr or "").strip(),
        "viz_stdout": (p2.stdout or "").strip(),
        "viz_stderr": (p2.stderr or "").strip(),
        "duration_s": duration_s,
    }

    return {
        "ok": True,
        "gen_id": gen_id,
        "acode_path": out_acode,
        "preview_path": out_preview,
        "acode_text": acode_text,
        "meta": meta,
        "duration_s": duration_s,
    }

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/load", methods=["POST"])
def api_load():
    host = request.form.get("host", "").strip() or "192.168.4.1"
    port = _safe_int("port", DEFAULT_PORT)
    content = request.form.get("acode", "")
    start_line = _safe_int("start_line", 1)

    lines = [ln.rstrip("\r\n") for ln in content.splitlines()]
    lines = [ln for ln in lines if ln.strip()]

    start_idx = max(0, min(len(lines), start_line - 1))
    last_ok_line = lines[start_idx - 1] if start_idx > 0 else ""
    duration_s = estimate_total_duration(lines, ESTIMATE_DEFAULTS)

    with state_lock:
        state.host = host
        state.port = port
        state.lines = lines
        state.idx = start_idx
        state.error = ""
        state.last_sent = ""
        state.last_ok = False
        state.last_ok_idx = start_idx
        state.last_ok_line = last_ok_line

    push_event("status", {"msg": "loaded", "lines": len(lines)})
    return jsonify({"ok": True, "lines": len(lines), "start_line": start_idx + 1, "duration_s": duration_s})

@app.route("/api/start", methods=["POST"])
def api_start():
    global worker_thread
    with state_lock:
        if not state.lines:
            return jsonify({"ok": False, "error": "no ACODE loaded"}), 400
        state.running = True
        state.paused = False
        state.stopping = False

    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(target=sender_worker, daemon=True)
        worker_thread.start()

    push_event("status", {"msg": "running"})
    return jsonify({"ok": True})

@app.route("/api/pause", methods=["POST"])
def api_pause():
    with state_lock:
        state.paused = True
    push_event("status", {"msg": "paused"})
    return jsonify({"ok": True})

@app.route("/api/resume", methods=["POST"])
def api_resume():
    with state_lock:
        state.paused = False
        state.running = True
    push_event("status", {"msg": "running"})
    return jsonify({"ok": True})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        state.stopping = True
        state.running = False
        state.paused = False
    push_event("status", {"msg": "stopping"})
    return jsonify({"ok": True})

@app.route("/events")
def sse_events():
    def gen():
        with state_lock:
            snap = {
                "running": state.running,
                "paused": state.paused,
                "idx": state.idx,
                "total": len(state.lines or []),
                "last_sent": state.last_sent,
                "last_ok": state.last_ok,
                "last_ok_idx": state.last_ok_idx,
                "last_ok_line": state.last_ok_line,
                "error": state.error,
            }
        yield "data: " + json.dumps(snap) + "\n\n"

        while True:
            msg = events.get()
            yield "data: " + json.dumps(msg) + "\n\n"

    return Response(gen(), mimetype="text/event-stream")

# ----------------------------
# Jog / pen / steppers
# ----------------------------
def _busy_guard() -> Optional[str]:
    with state_lock:
        if state.running and not state.paused:
            return "busy: stop or pause printing before jog/pen/steppers"
    return None

@app.route("/api/jog", methods=["POST"])
def api_jog():
    err = _busy_guard()
    if err:
        return jsonify({"ok": False, "error": err}), 409

    direction = (request.form.get("dir", "") or "").strip().lower()
    host = request.form.get("host", "").strip() or "192.168.4.1"
    port = _safe_int("port", DEFAULT_PORT)

    jog_mm = _safe_float("jog_mm", 10.0)
    steps_per_mm = _safe_float("steps_per_mm", 9.142857)
    feed = _safe_int("feed", 1200)

    steps = int(round(jog_mm * steps_per_mm))
    if steps <= 0:
        return jsonify({"ok": False, "error": "jog_mm or steps_per_mm too small"}), 400

    if direction == "fwd":
        line = f"W L{steps} R{steps} F{feed}"
    elif direction == "back":
        line = f"W L{-steps} R{-steps} F{feed}"
    elif direction == "left":
        line = f"W L{-steps} R{steps} F{feed}"
    elif direction == "right":
        line = f"W L{steps} R{-steps} F{feed}"
    else:
        return jsonify({"ok": False, "error": "bad dir"}), 400

    push_event("line", {"idx": 0, "total": 0, "line": line})
    out = send_one_command(host, port, line, timeout_s=LINE_ACK_TIMEOUT_S)
    if out["ok"]:
        push_event("ok", {"idx": 0})
        return jsonify({"ok": True, "line": line})
    push_event("error", {"msg": out["error"] or "jog failed"})
    return jsonify({"ok": False, "error": out["error"] or "jog failed"}), 500

@app.route("/api/pen", methods=["POST"])
def api_pen():
    err = _busy_guard()
    if err:
        return jsonify({"ok": False, "error": err}), 409

    host = request.form.get("host", "").strip() or "192.168.4.1"
    port = _safe_int("port", DEFAULT_PORT)
    mode = (request.form.get("mode", "") or "").strip().lower()

    if mode == "up":
        line = "P U"
    elif mode == "down":
        line = "P D"
    else:
        return jsonify({"ok": False, "error": "bad mode"}), 400

    push_event("line", {"idx": 0, "total": 0, "line": line})
    out = send_one_command(host, port, line, timeout_s=LINE_ACK_TIMEOUT_S)
    if out["ok"]:
        push_event("ok", {"idx": 0})
        return jsonify({"ok": True, "line": line})
    push_event("error", {"msg": out["error"] or "pen failed"})
    return jsonify({"ok": False, "error": out["error"] or "pen failed"}), 500

@app.route("/api/steppers", methods=["POST"])
def api_steppers():
    err = _busy_guard()
    if err:
        return jsonify({"ok": False, "error": err}), 409

    host = request.form.get("host", "").strip() or "192.168.4.1"
    port = _safe_int("port", DEFAULT_PORT)

    cmd = (request.form.get("cmd", "") or "").strip()
    if not cmd:
        return jsonify({"ok": False, "error": "missing cmd"}), 400

    push_event("line", {"idx": 0, "total": 0, "line": cmd})
    out = send_one_command(host, port, cmd, timeout_s=LINE_ACK_TIMEOUT_S)
    if out["ok"]:
        push_event("ok", {"idx": 0})
        return jsonify({"ok": True, "line": cmd})
    push_event("error", {"msg": out["error"] or "steppers failed"})
    return jsonify({"ok": False, "error": out["error"] or "steppers failed"}), 500

# ----------------------------
# Generator: PNG
# ----------------------------
@app.route("/api/gen", methods=["POST"])
def api_gen():
    if "png" not in request.files:
        return jsonify({"ok": False, "error": "missing file field: png"}), 400

    f = request.files["png"]
    if not f.filename.lower().endswith(".png"):
        return jsonify({"ok": False, "error": "only .png accepted"}), 400

    gen_id_tmp = str(uuid.uuid4())[:8]
    png_path = os.path.join(UPLOADS_DIR, f"{gen_id_tmp}.png")
    f.save(png_path)

    params = {
        "line_advance": _parse_line_advance("line_advance"),
        "img_width_mm": _safe_float("img_width_mm", 120.0),
        "work_width_px": _safe_int("work_width_px", 1600),
        "line_spacing_mm": _safe_float("line_spacing_mm", 0.7),
        "threshold": _safe_int("threshold", 160),
        "gamma": _safe_float("gamma", 1.0),
        "min_segment_mm": _safe_float("min_segment_mm", 1.0),
        "invert": _safe_bool("invert"),
        "margin_mm": _safe_float("margin_mm", 0.0),
        "flip_y": _safe_bool("flip_y"),
        "y_order": _parse_y_order("y_order"),
        "scan": (request.form.get("scan", "serpentine") or "serpentine"),
        "y_jitter_mm": _safe_float("y_jitter_mm", 0.04),
        "seed": (None if (request.form.get("seed", "").strip() == "") else _safe_int("seed", 0)),
        "x_mode": (request.form.get("x_mode", "pixel") or "pixel"),
        "x_step_mm": _safe_float("x_step_mm", 0.25),
        "row_angle_deg": _safe_float("row_angle_deg", 18.0),
        "soft_min_dy_mm": _safe_float("soft_min_dy_mm", 0.3),
        "feed_lin": _safe_int("feed_lin", 1200),
        "feed_turn": _safe_int("feed_turn", 800),

        "viz_dpi": _safe_int("viz_dpi", 160),
        "viz_arc_step_mm": _safe_float("viz_arc_step_mm", 1.0),
        "viz_arc_step_deg": _safe_float("viz_arc_step_deg", 1.0),
        "viz_equal": _safe_bool("viz_equal"),
        "viz_invert_y": _safe_bool("viz_invert_y"),
        "viz_home_resets_pose": _safe_bool("viz_home_resets_pose"),
    }

    result = generate_from_png(png_path, params)
    if not result["ok"]:
        with gen_lock:
            gen_state.error = result["error"]
        return jsonify({"ok": False, "error": result["error"]}), 500

    machine = resolve_machine_settings(ACODE_PY_PATH)
    with gen_lock:
        gen_state.gen_id = result["gen_id"]
        gen_state.png_path = result["png_path"]
        gen_state.acode_path = result["acode_path"]
        gen_state.preview_path = result["preview_path"]
        gen_state.acode_text = result["acode_text"]
        gen_state.meta = result["meta"]
        gen_state.error = ""
        gen_state.viz_settings = {
            "wheelbase_mm": machine["wheelbase_mm"],
            "steps_per_mm": machine["steps_per_mm"],
            "turn_gain": machine["turn_gain"],
            "viz_arc_step_mm": params["viz_arc_step_mm"],
            "viz_arc_step_deg": params["viz_arc_step_deg"],
            "viz_home_resets_pose": params["viz_home_resets_pose"],
            "viz_equal": params["viz_equal"],
            "viz_invert_y": params["viz_invert_y"],
        }

    push_event("gen", {"msg": "generated", "gen_id": result["gen_id"], "lines": result["meta"]["acode_lines"]})
    return jsonify({
        "ok": True,
        "gen_id": result["gen_id"],
        "acode_lines": result["meta"]["acode_lines"],
        "warnings": result["meta"].get("warnings", []),
        "duration_s": result.get("duration_s", 0.0),
    })

# ----------------------------
# Generator: DXF
# ----------------------------
@app.route("/api/dxf", methods=["POST"])
def api_dxf():
    if "dxf" not in request.files:
        return jsonify({"ok": False, "error": "missing file field: dxf"}), 400

    f = request.files["dxf"]
    if not f.filename.lower().endswith(".dxf"):
        return jsonify({"ok": False, "error": "only .dxf accepted"}), 400

    gen_id_tmp = str(uuid.uuid4())[:8]
    dxf_path = os.path.join(UPLOADS_DIR, f"{gen_id_tmp}.dxf")
    f.save(dxf_path)

    params = {
        "layer": (request.form.get("layer", "") or "").strip(),
        "reorder": not _safe_bool("no_reorder", False),
        "feed_lin": _safe_int("feed_lin", 1200),
        "feed_turn": _safe_int("feed_turn", 800),
        "feed_arc": _safe_int("feed_arc", 800),
        "flat_step": _safe_float("flat_step", 1.0),
        "epsilon": _safe_float("epsilon", 0.25),
        "viz_dpi": _safe_int("viz_dpi", 160),
        "viz_arc_step_mm": _safe_float("viz_arc_step_mm", 1.0),
        "viz_arc_step_deg": _safe_float("viz_arc_step_deg", 1.0),
        "viz_equal": _safe_bool("viz_equal"),
        "viz_invert_y": _safe_bool("viz_invert_y"),
        "viz_home_resets_pose": _safe_bool("viz_home_resets_pose"),
    }

    result = generate_from_dxf(dxf_path, params)
    if not result["ok"]:
        with gen_lock:
            gen_state.error = result["error"]
        return jsonify({"ok": False, "error": result["error"]}), 500

    machine = resolve_machine_settings(ACODE_PY_PATH)
    with gen_lock:
        gen_state.gen_id = result["gen_id"]
        gen_state.png_path = ""
        gen_state.acode_path = result["acode_path"]
        gen_state.preview_path = result["preview_path"]
        gen_state.acode_text = result["acode_text"]
        gen_state.meta = result["meta"]
        gen_state.error = ""
        gen_state.viz_settings = {
            "wheelbase_mm": machine["wheelbase_mm"],
            "steps_per_mm": machine["steps_per_mm"],
            "turn_gain": machine["turn_gain"],
            "viz_arc_step_mm": params["viz_arc_step_mm"],
            "viz_arc_step_deg": params["viz_arc_step_deg"],
            "viz_home_resets_pose": params["viz_home_resets_pose"],
            "viz_equal": params["viz_equal"],
            "viz_invert_y": params["viz_invert_y"],
        }

    push_event(
        "gen",
        {
            "msg": "dxf_generated",
            "gen_id": result["gen_id"],
            "lines": result["meta"]["acode_lines"],
        },
    )

    return jsonify(
        {
            "ok": True,
            "gen_id": result["gen_id"],
            "acode_lines": result["meta"]["acode_lines"],
            "warnings": result["meta"].get("warnings", []),
            "duration_s": result.get("duration_s", 0.0),
        }
    )

# ----------------------------
# Generator: Text outline
# ----------------------------
@app.route("/api/fonts", methods=["GET"])
def api_fonts():
    fonts = list_system_fonts(limit=250)
    return jsonify({"ok": True, "fonts": fonts})

@app.route("/api/text_outline", methods=["POST"])
def api_text_outline():
    text = (request.form.get("text", "") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "text is empty"}), 400

    line_width_mm = _safe_float("line_width_mm", 120.0)
    letter_height_mm = _safe_float("letter_height_mm", 12.0)
    stroke_mm = _safe_float("stroke_mm", 0.35)
    font_path = (request.form.get("font_path", "") or "").strip()
    render_dpi = _safe_int("render_dpi", 600)

    gen_id_tmp = str(uuid.uuid4())[:8]
    png_path = os.path.join(UPLOADS_DIR, f"text_{gen_id_tmp}.png")

    try:
        png_bytes = render_text_outline_png(
            text=text,
            line_width_mm=line_width_mm,
            letter_height_mm=letter_height_mm,
            font_path=font_path,
            stroke_mm=stroke_mm,
            dpi=render_dpi,
            padding_mm=2.0,
        )
        with open(png_path, "wb") as f:
            f.write(png_bytes)
    except Exception as e:
        return jsonify({"ok": False, "error": f"render failed: {e}"}), 500

    params = {
        "line_advance": _parse_line_advance("line_advance"),
        "img_width_mm": line_width_mm,
        "work_width_px": _safe_int("work_width_px", 2200),
        "line_spacing_mm": _safe_float("line_spacing_mm", 0.6),
        "threshold": _safe_int("threshold", 160),
        "gamma": _safe_float("gamma", 1.0),
        "min_segment_mm": _safe_float("min_segment_mm", 0.8),
        "invert": _safe_bool("invert"),
        "margin_mm": _safe_float("margin_mm", 0.0),
        "flip_y": _safe_bool("flip_y"),
        "y_order": _parse_y_order("y_order"),
        "scan": (request.form.get("scan", "serpentine") or "serpentine"),
        "y_jitter_mm": _safe_float("y_jitter_mm", 0.02),
        "seed": (None if (request.form.get("seed", "").strip() == "") else _safe_int("seed", 0)),
        "x_mode": (request.form.get("x_mode", "pixel") or "pixel"),
        "x_step_mm": _safe_float("x_step_mm", 0.25),
        "row_angle_deg": _safe_float("row_angle_deg", 0.1),
        "soft_min_dy_mm": _safe_float("soft_min_dy_mm", 0.25),
        "feed_lin": _safe_int("feed_lin", 1200),
        "feed_turn": _safe_int("feed_turn", 800),

        "viz_dpi": _safe_int("viz_dpi", 160),
        "viz_arc_step_mm": _safe_float("viz_arc_step_mm", 1.0),
        "viz_arc_step_deg": _safe_float("viz_arc_step_deg", 1.0),
        "viz_equal": _safe_bool("viz_equal"),
        "viz_invert_y": _safe_bool("viz_invert_y"),
        "viz_home_resets_pose": _safe_bool("viz_home_resets_pose"),
    }

    result = generate_from_png(png_path, params)
    if not result["ok"]:
        with gen_lock:
            gen_state.error = result["error"]
        return jsonify({"ok": False, "error": result["error"]}), 500

    machine = resolve_machine_settings(ACODE_PY_PATH)
    with gen_lock:
        gen_state.gen_id = result["gen_id"]
        gen_state.png_path = result["png_path"]
        gen_state.acode_path = result["acode_path"]
        gen_state.preview_path = result["preview_path"]
        gen_state.acode_text = result["acode_text"]
        gen_state.meta = result["meta"]
        gen_state.error = ""
        gen_state.viz_settings = {
            "wheelbase_mm": machine["wheelbase_mm"],
            "steps_per_mm": machine["steps_per_mm"],
            "turn_gain": machine["turn_gain"],
            "viz_arc_step_mm": params["viz_arc_step_mm"],
            "viz_arc_step_deg": params["viz_arc_step_deg"],
            "viz_home_resets_pose": params["viz_home_resets_pose"],
            "viz_equal": params["viz_equal"],
            "viz_invert_y": params["viz_invert_y"],
        }

    push_event("gen", {"msg": "text_outline_generated", "gen_id": result["gen_id"], "lines": result["meta"]["acode_lines"]})
    return jsonify({
        "ok": True,
        "gen_id": result["gen_id"],
        "acode_lines": result["meta"]["acode_lines"],
        "warnings": result["meta"].get("warnings", []),
        "duration_s": result.get("duration_s", 0.0),
    })

# ----------------------------
# Preview + download + push
# ----------------------------
@app.route("/preview/<gen_id>.png", methods=["GET"])
def preview_png(gen_id: str):
    with gen_lock:
        if gen_id != gen_state.gen_id:
            abort(404)
        path = gen_state.preview_path
    if not path or not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="image/png", as_attachment=False)

@app.route("/download/<gen_id>.acode", methods=["GET"])
def download_acode(gen_id: str):
    with gen_lock:
        if gen_id != gen_state.gen_id:
            abort(404)
        path = gen_state.acode_path
    if not path or not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="text/plain", as_attachment=True, download_name=f"{gen_id}.acode")

@app.route("/api/push_to_sender", methods=["POST"])
def api_push_to_sender():
    host = request.form.get("host", "").strip()
    port_s = request.form.get("port", "").strip()

    with gen_lock:
        if not gen_state.acode_text:
            return jsonify({"ok": False, "error": "no generated ACODE"}), 400
        text = gen_state.acode_text

    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    normalized_text = "\n".join(lines) + ("\n" if lines else "")

    with state_lock:
        if host:
            state.host = host
        if port_s:
            try:
                state.port = int(port_s)
            except ValueError:
                pass
        state.lines = lines
        state.idx = 0
        state.error = ""
        state.last_sent = ""
        state.last_ok = False
        state.last_ok_idx = 0
        state.last_ok_line = ""

    push_event("status", {"msg": "loaded_from_generator", "lines": len(lines)})
    return jsonify({"ok": True, "lines": len(lines), "acode_text": normalized_text})

# ----------------------------
# Sender preview + serial tools
# ----------------------------
@app.route("/api/sender_vizdata", methods=["POST"])
def api_sender_vizdata():
    content = request.form.get("acode", "") or ""
    lines = _split_acode_lines(content)
    if not lines:
        return jsonify({"ok": False, "error": "no acode"}), 400

    arc_step_mm = _safe_float("viz_arc_step_mm", 1.0)
    arc_step_deg = _safe_float("viz_arc_step_deg", 1.0)
    home_resets_pose = _safe_bool("viz_home_resets_pose", True)
    viz_equal = _safe_bool("viz_equal", False)
    viz_invert_y = _safe_bool("viz_invert_y", False)

    machine = resolve_machine_settings(ACODE_PY_PATH)
    traj = acode_to_trajectory(
        lines=lines,
        wheelbase_mm=machine["wheelbase_mm"],
        steps_per_mm=machine["steps_per_mm"],
        turn_gain=machine["turn_gain"],
        arc_step_mm=arc_step_mm,
        arc_step_deg=arc_step_deg,
        home_resets_pose=home_resets_pose,
    )
    traj["viz"] = {"equal": viz_equal, "invert_y": viz_invert_y}

    duration_s = estimate_total_duration(lines, ESTIMATE_DEFAULTS)

    return jsonify({
        "ok": True,
        "points": traj["points"],
        "bounds": traj["bounds"],
        "settings": traj["settings"],
        "line_frames": traj.get("line_frames", []),
        "viz": traj["viz"],
        "duration_s": duration_s,
        "total_lines": len(lines),
    })

@app.route("/api/serial_send", methods=["POST"])
def api_serial_send():
    err = _busy_guard()
    if err:
        return jsonify({"ok": False, "error": err}), 409

    host = request.form.get("host", "").strip() or "192.168.4.1"
    port = _safe_int("port", DEFAULT_PORT)
    line = (request.form.get("line", "") or "").strip()
    if not line:
        return jsonify({"ok": False, "error": "missing line"}), 400

    push_event("line", {"idx": 0, "total": 0, "line": line})
    out = send_one_command(host, port, line, timeout_s=LINE_ACK_TIMEOUT_S)
    if out["ok"]:
        push_event("ok", {"idx": 0, "line": line})
        return jsonify({"ok": True, "line": line})
    push_event("error", {"msg": out["error"] or "serial failed"})
    return jsonify({"ok": False, "error": out["error"] or "serial failed"}), 500

# ----------------------------
# Simulator data endpoint
# ----------------------------
@app.route("/api/vizdata/<gen_id>", methods=["GET"])
def api_vizdata(gen_id: str):
    with gen_lock:
        if gen_id != gen_state.gen_id:
            abort(404)
        if not gen_state.acode_text:
            abort(404)
        acode_text = gen_state.acode_text
        vs = gen_state.viz_settings or {}

    wheelbase_mm = float(vs.get("wheelbase_mm", 120.0))
    steps_per_mm = float(vs.get("steps_per_mm", 9.142857))
    turn_gain = float(vs.get("turn_gain", 1.0))
    arc_step_mm = float(vs.get("viz_arc_step_mm", 1.0))
    arc_step_deg = float(vs.get("viz_arc_step_deg", 1.0))
    home_resets_pose = bool(vs.get("viz_home_resets_pose", True))
    viz_equal = bool(vs.get("viz_equal", False))
    viz_invert_y = bool(vs.get("viz_invert_y", False))

    lines = [ln.strip() for ln in acode_text.splitlines() if ln.strip()]
    traj = acode_to_trajectory(
        lines=lines,
        wheelbase_mm=wheelbase_mm,
        steps_per_mm=steps_per_mm,
        turn_gain=turn_gain,
        arc_step_mm=arc_step_mm,
        arc_step_deg=arc_step_deg,
        home_resets_pose=home_resets_pose,
    )
    traj["viz"] = {"equal": viz_equal, "invert_y": viz_invert_y}
    return jsonify(traj)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
