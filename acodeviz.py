#!/usr/bin/env python3
# acode_viz.py
#
# Visualize ANC ACODE (H, P U/D, W L.. R.. F.., END)
# Uses calibration from your acode.py:
#   WHEELBASE_MM, STEPS_PER_MM, TURN_GAIN
#
# Important:
# - Pose is continuous. Next W starts where previous W ended.
# - Arcs are rendered by subdividing each W move into small steps.

import argparse
import importlib.util
import math
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt

RE_W = re.compile(r"^\s*W\s+L(-?\d+)\s+R(-?\d+)(?:\s+F(-?\d+))?\s*$", re.IGNORECASE)
RE_P = re.compile(r"^\s*P\s+([UD])\s*$", re.IGNORECASE)

HOME = "H"
END = "END"


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # radians
    pen_down: bool = False


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def load_module_from_path(py_path: str, module_name: str = "_anc_acode_cfg") -> Any:
    py_path = os.path.abspath(py_path)
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"acode.py not found: {py_path}")
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def require_attr(mod: Any, name: str) -> Any:
    if not hasattr(mod, name):
        raise RuntimeError(f"acode.py missing required constant: {name}")
    return getattr(mod, name)


def resolve_settings_from_acode_py(acode_py_path: str) -> Tuple[float, float, float]:
    mod = load_module_from_path(acode_py_path)
    wheelbase_mm = float(require_attr(mod, "WHEELBASE_MM"))
    steps_per_mm = float(require_attr(mod, "STEPS_PER_MM"))
    turn_gain = float(require_attr(mod, "TURN_GAIN"))
    return wheelbase_mm, steps_per_mm, turn_gain


def integrate_step(
    x: float,
    y: float,
    heading: float,
    dl_mm: float,
    dr_mm: float,
    wheelbase_mm: float,
    turn_gain: float,
) -> Tuple[float, float, float]:
    """
    Mirror generator:
      dtheta = (dr - dl) / (WHEELBASE * TURN_GAIN)
      ds     = (dl + dr) / 2
    """
    ds = 0.5 * (dl_mm + dr_mm)
    dtheta = (dr_mm - dl_mm) / (wheelbase_mm * max(turn_gain, 1e-12))

    if abs(dtheta) < 1e-12:
        nx = x + ds * math.cos(heading)
        ny = y + ds * math.sin(heading)
        return nx, ny, heading

    R = ds / dtheta
    nx = x + R * (math.sin(heading + dtheta) - math.sin(heading))
    ny = y - R * (math.cos(heading + dtheta) - math.cos(heading))
    nh = wrap_pi(heading + dtheta)
    return nx, ny, nh


def compute_substeps(
    dl_mm: float,
    dr_mm: float,
    wheelbase_mm: float,
    turn_gain: float,
    arc_step_mm: float,
    arc_step_deg: float,
) -> int:
    ds_total = 0.5 * (dl_mm + dr_mm)
    dtheta_total = (dr_mm - dl_mm) / (wheelbase_mm * max(turn_gain, 1e-12))

    if abs(dtheta_total) < 1e-9:
        return 1

    arc_len = abs(ds_total)
    n_by_len = max(1, int(math.ceil(arc_len / max(arc_step_mm, 1e-9))))
    n_by_ang = max(1, int(math.ceil(abs(math.degrees(dtheta_total)) / max(arc_step_deg, 1e-9))))
    return max(n_by_len, n_by_ang)


def parse_ops(lines: List[str]) -> List[Tuple[str, Tuple]]:
    ops: List[Tuple[str, Tuple]] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith(";"):
            continue

        up = s.upper()
        if up == HOME:
            ops.append(("H", ()))
            continue
        if up == END:
            ops.append(("END", ()))
            continue

        mp = RE_P.match(s)
        if mp:
            ops.append(("P", (mp.group(1).upper(),)))
            continue

        mw = RE_W.match(s)
        if mw:
            l = int(mw.group(1))
            r = int(mw.group(2))
            f = int(mw.group(3)) if mw.group(3) is not None else 0
            ops.append(("W", (l, r, f)))
            continue

        ops.append(("UNKNOWN", (s,)))
    return ops


def simulate_to_polyline(
    ops: List[Tuple[str, Tuple]],
    wheelbase_mm: float,
    spmm: float,
    turn_gain: float,
    home_resets_pose: bool,
    arc_step_mm: float,
    arc_step_deg: float,
) -> Tuple[List[Tuple[Optional[float], Optional[float]]], dict, State]:
    st = State()
    poly: List[Tuple[Optional[float], Optional[float]]] = []

    unknown = 0
    w_cnt = 0

    minx = maxx = st.x
    miny = maxy = st.y

    def upd_bounds(x: float, y: float):
        nonlocal minx, maxx, miny, maxy
        minx = min(minx, x)
        maxx = max(maxx, x)
        miny = min(miny, y)
        maxy = max(maxy, y)

    def break_line():
        if poly and poly[-1] != (None, None):
            poly.append((None, None))

    for op, args in ops:
        if op == "UNKNOWN":
            unknown += 1
            continue

        if op == "END":
            break

        if op == "H":
            if home_resets_pose:
                st.x, st.y, st.heading = 0.0, 0.0, 0.0
                upd_bounds(st.x, st.y)
                # teleport breaks drawing
                break_line()
            continue

        if op == "P":
            new_down = (args[0] == "D")
            if new_down != st.pen_down:
                # switching pen state breaks polyline
                break_line()
                st.pen_down = new_down
            continue

        if op == "W":
            w_cnt += 1
            l_steps, r_steps, _feed = args

            dl_mm = l_steps / spmm
            dr_mm = r_steps / spmm

            n = compute_substeps(dl_mm, dr_mm, wheelbase_mm, turn_gain, arc_step_mm, arc_step_deg)
            dl = dl_mm / n
            dr = dr_mm / n

            if st.pen_down:
                # Ensure polyline starts at current point
                if not poly or poly[-1] == (None, None):
                    poly.append((st.x, st.y))

                for _ in range(n):
                    st.x, st.y, st.heading = integrate_step(
                        st.x, st.y, st.heading, dl, dr, wheelbase_mm, turn_gain
                    )
                    poly.append((st.x, st.y))
                    upd_bounds(st.x, st.y)
            else:
                for _ in range(n):
                    st.x, st.y, st.heading = integrate_step(
                        st.x, st.y, st.heading, dl, dr, wheelbase_mm, turn_gain
                    )
                    upd_bounds(st.x, st.y)

            continue

    stats = {
        "unknown_lines": unknown,
        "w_commands": w_cnt,
        "minx": minx,
        "maxx": maxx,
        "miny": miny,
        "maxy": maxy,
        "width_mm": (maxx - minx),
        "height_mm": (maxy - miny),
    }
    return poly, stats, st


def render_polyline(
    poly: List[Tuple[Optional[float], Optional[float]]],
    out_path: str,
    dpi: int,
    equal: bool,
    title: str,
    invert_y: bool,
    show: bool,
):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if poly:
        ax.plot(xs, ys, linewidth=1.0)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)

    if equal:
        ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()

    ax.grid(True, linewidth=0.3)

    ext = os.path.splitext(out_path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    elif ext in [".svg", ".pdf"]:
        fig.savefig(out_path, bbox_inches="tight")
    else:
        fig.savefig(out_path + ".png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize ACODE path (pen-down only) with continuous pose and arc rendering.")
    ap.add_argument("acode", help="Input .acode file")
    ap.add_argument("-o", "--out", default="acode_preview.png", help="Output image path (png/svg/pdf)")
    ap.add_argument("--acode-py", default=None, help="Path to acode.py (default: sibling of this script)")

    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--equal", action="store_true")
    ap.add_argument("--invert-y", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--title", default=None)

    ap.add_argument("--home-resets-pose", action="store_true", help="Treat H as teleport to (0,0,0,0)")

    ap.add_argument("--arc-step-mm", type=float, default=1.0, help="Max distance per segment when drawing arcs")
    ap.add_argument("--arc-step-deg", type=float, default=1.0, help="Max angle per segment when drawing arcs")

    args = ap.parse_args()

    if args.acode_py is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.acode_py = os.path.join(here, "acode.py")

    wheelbase_mm, spmm, turn_gain = resolve_settings_from_acode_py(args.acode_py)

    with open(args.acode, "r", encoding="utf-8", errors="ignore") as f:
        ops = parse_ops(f.read().splitlines())

    poly, stats, st = simulate_to_polyline(
        ops=ops,
        wheelbase_mm=wheelbase_mm,
        spmm=spmm,
        turn_gain=turn_gain,
        home_resets_pose=args.home_resets_pose,
        arc_step_mm=args.arc_step_mm,
        arc_step_deg=args.arc_step_deg,
    )

    title = args.title if args.title else os.path.basename(args.acode)

    render_polyline(
        poly=poly,
        out_path=args.out,
        dpi=args.dpi,
        equal=args.equal,
        title=title,
        invert_y=args.invert_y,
        show=args.show,
    )

    print("ACODE visualization done.")
    print(f"- input: {args.acode}")
    print(f"- acode.py: {args.acode_py}")
    print(f"- output: {args.out}")
    print(f"- wheelbase_mm: {wheelbase_mm}")
    print(f"- steps_per_mm: {spmm}")
    print(f"- turn_gain: {turn_gain}")
    print(f"- W commands: {stats['w_commands']}")
    print(f"- unknown lines: {stats['unknown_lines']}")
    print(f"- size: {stats['width_mm']:.3f} x {stats['height_mm']:.3f} mm")
    print(f"- final pose: x={st.x:.3f} y={st.y:.3f} heading={math.degrees(st.heading):.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())