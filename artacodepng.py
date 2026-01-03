#!/usr/bin/env python3
# artacodepng.py
# PNG -> ACODE, horyzontalne scanline (tylko poziome odcinki)
# Format: H, P U/D, W L.. R.. F.., END
#
# Fix: brak "pocięcia w pionie"
# - segmenty wykrywane na pełnej siatce X (pixel mode)
# - opcjonalny sampling w X dla szybkości (step mode)

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

Point = Tuple[float, float]

# ----------------------------
# Machine calibration
# ----------------------------
WHEELBASE_MM = 255.0
WHEEL_CIRCUM_MM = 175.0
STEPS_PER_REV = 200
MICROSTEPS = 8

STEPS_PER_MM = (STEPS_PER_REV * MICROSTEPS) / WHEEL_CIRCUM_MM

TURN_STEPS_360_MEASURED = 7400
_steps_nominal_360 = (WHEELBASE_MM * math.pi) * STEPS_PER_MM
TURN_GAIN = TURN_STEPS_360_MEASURED / _steps_nominal_360

# ----------------------------
# ACODE commands
# ----------------------------
HOME_CMD = "H"
PEN_UP_CMD = "P U"
PEN_DOWN_CMD = "P D"
END_CMD = "END"

# ----------------------------
# Helpers
# ----------------------------
def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def steps_from_mm(mm: float) -> int:
    return int(round(mm * STEPS_PER_MM))

def emit_w(out: List[str], l_steps: int, r_steps: int, feed: int):
    if l_steps == 0 and r_steps == 0:
        return
    out.append(f"W L{l_steps} R{r_steps} F{feed}")

def emit_turn_in_place(out: List[str], dtheta: float, feed_turn: int):
    dl = -(WHEELBASE_MM / 2.0) * dtheta * TURN_GAIN
    dr = +(WHEELBASE_MM / 2.0) * dtheta * TURN_GAIN
    emit_w(out, steps_from_mm(dl), steps_from_mm(dr), feed_turn)

def emit_straight(out: List[str], ds: float, feed_lin: int):
    s = steps_from_mm(ds)
    emit_w(out, s, s, feed_lin)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ----------------------------
# PNG processing
# ----------------------------
def load_grayscale(path: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.uint8)

def invert_if(gray: np.ndarray, invert: bool) -> np.ndarray:
    if not invert:
        return gray
    return (255 - gray).astype(np.uint8)

def resize_to_width_px(gray: np.ndarray, target_width_px: int) -> np.ndarray:
    h, w = gray.shape
    if target_width_px <= 0:
        raise ValueError("work_width_px must be > 0")
    if w == target_width_px:
        return gray
    scale = target_width_px / float(w)
    target_height_px = max(1, int(round(h * scale)))
    im = Image.fromarray(gray, mode="L")
    im2 = im.resize((target_width_px, target_height_px), resample=Image.Resampling.LANCZOS)
    return np.asarray(im2, dtype=np.uint8)

def mm_per_px_from_width(width_mm: float, width_px: int) -> float:
    if width_mm <= 0:
        raise ValueError("img_width_mm must be > 0")
    if width_px <= 0:
        raise ValueError("width_px must be > 0")
    return width_mm / float(width_px)

def runs_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []
    m = mask.astype(np.uint8)
    diff = np.diff(m, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))

def build_scanline_rows_segments_mm(
    gray: np.ndarray,
    mm_per_px: float,
    line_spacing_mm: float,
    threshold: int,
    gamma: float,
    min_segment_mm: float,
    y_jitter_mm: float,
    seed: Optional[int],
    x_mode: str,          # "pixel" | "step"
    x_step_mm: float,     # used only in "step"
) -> List[Tuple[float, List[Tuple[float, float]]]]:
    """
    [
      (y_mm, [(x1_mm, x2_mm), ...]),
      ...
    ]
    """
    if seed is not None:
        random.seed(seed)

    h, w = gray.shape

    min_segment_px = max(1, int(round(min_segment_mm / mm_per_px)))
    line_spacing_px = max(1, int(math.ceil(line_spacing_mm / mm_per_px)))

    g = gray.astype(np.float32) / 255.0
    if gamma != 1.0:
        g = np.power(g, gamma)

    t = float(threshold) / 255.0
    rows: List[Tuple[float, List[Tuple[float, float]]]] = []

    if x_mode not in ("pixel", "step"):
        raise ValueError("x_mode must be 'pixel' or 'step'")

    if x_mode == "step":
        if x_step_mm <= 0:
            raise ValueError("x_step_mm must be > 0 for x_mode=step")
        x_step_px = max(1, int(round(x_step_mm / mm_per_px)))

    for y_px in range(0, h, line_spacing_px):
        row = g[y_px, :]

        if x_mode == "pixel":
            draw_mask = row < t
            runs = runs_from_mask(draw_mask)
            segs: List[Tuple[float, float]] = []
            for x1, x2 in runs:
                if (x2 - x1) < min_segment_px:
                    continue
                segs.append((x1 * mm_per_px, x2 * mm_per_px))
        else:
            xs = np.arange(0, w, x_step_px, dtype=int)
            sampled = row[xs]
            draw_mask_s = sampled < t
            runs_s = runs_from_mask(draw_mask_s)
            segs = []
            for s, e in runs_s:
                x1 = int(xs[s])
                x2 = int(xs[e])
                x2 = min(w - 1, x2 + (x_step_px - 1))
                if (x2 - x1) < min_segment_px:
                    continue
                segs.append((x1 * mm_per_px, x2 * mm_per_px))

        if not segs:
            continue

        base_y_mm = y_px * mm_per_px
        jitter = random.uniform(-y_jitter_mm, y_jitter_mm) if y_jitter_mm > 0 else 0.0
        rows.append((base_y_mm + jitter, segs))

    return rows

# ----------------------------
# Geometry
# ----------------------------
@dataclass
class PrimLine:
    p0: Point
    p1: Point

def build_scanline_paths_no_x_drift(
    rows: List[Tuple[float, List[Tuple[float, float]]]],
    img_width_mm: float,
    margin_mm: float,
    flip_y: bool,
    height_mm: float,
    scan: str,  # "serpentine" | "ltr" | "ltr+rtl"
    y_order: str,  # "top-down" | "bottom-up"
) -> List[List[PrimLine]]:
    if not rows:
        return []

    ys: List[float] = []
    segs_rows: List[List[Tuple[float, float]]] = []
    for y_mm, segs in rows:
        y2 = height_mm - y_mm if flip_y else y_mm
        y2 += margin_mm
        ys.append(y2)
        segs_rows.append(segs)

    order = sorted(range(len(ys)), key=lambda i: ys[i])
    ys = [ys[i] for i in order]
    segs_rows = [segs_rows[i] for i in order]

    if y_order == "bottom-up":
        ys.reverse()
        segs_rows.reverse()

    x_lo = margin_mm
    x_hi = margin_mm + img_width_mm

    paths: List[List[PrimLine]] = []

    for i, (y_mm, segs) in enumerate(zip(ys, segs_rows)):
        segs2 = [(x1 + margin_mm, x2 + margin_mm) for (x1, x2) in segs]
        segs2.sort(key=lambda a: min(a[0], a[1]))

        serp = (scan == "serpentine")
        bounce = (scan == "ltr+rtl")
        left_to_right = True if (not serp) else (i % 2 == 0)

        seg_iter = segs2 if left_to_right else list(reversed(segs2))

        for x1, x2 in seg_iter:
            if left_to_right:
                p0x, p1x = x1, x2
            else:
                p0x, p1x = x2, x1

            p0x = clamp(p0x, x_lo, x_hi)
            p1x = clamp(p1x, x_lo, x_hi)

            paths.append([PrimLine((p0x, y_mm), (p1x, y_mm))])

        if bounce:
            for x1, x2 in reversed(segs2):
                p0x = clamp(x2, x_lo, x_hi)
                p1x = clamp(x1, x_lo, x_hi)
                paths.append([PrimLine((p0x, y_mm), (p1x, y_mm))])

    return paths

def lines_to_acode(
    paths: List[List[PrimLine]],
    feed_lin: int,
    feed_turn: int,
    row_angle_deg: float,
    soft_min_dy_mm: float,
    line_advance: str,
    scan: str,
) -> List[str]:
    # Serpentine już zawiera optymalny kierunek końcowy/następny start,
    # więc przejazdy „ukośne” między wierszami są niepożądane.
    # W tym trybie erzac miękkiego przejścia zastępujemy osiowym ruchem (jak turn90).
    soft_rows = (line_advance == "soft") and (scan != "serpentine")
    real_90_rows = (line_advance == "real90")
    if soft_rows:
        if row_angle_deg <= 0 or row_angle_deg >= 90:
            raise ValueError("row_angle_deg should be in (0, 90)")

        alpha = math.radians(row_angle_deg)
        tan_a = math.tan(alpha)

    x = 0.0
    y = 0.0
    heading = 0.0
    pen_down = False

    out: List[str] = [HOME_CMD, PEN_UP_CMD]

    def set_pen(down: bool):
        nonlocal pen_down
        if down and not pen_down:
            out.append(PEN_DOWN_CMD)
            pen_down = True
        if (not down) and pen_down:
            out.append(PEN_UP_CMD)
            pen_down = False

    def go_to_point(nx: float, ny: float):
        nonlocal x, y, heading
        dx, dy = nx - x, ny - y
        ds = math.hypot(dx, dy)
        if ds < 1e-9:
            return
        target = math.atan2(dy, dx)
        dtheta = wrap_pi(target - heading)
        if abs(dtheta) > 1e-9:
            emit_turn_in_place(out, dtheta, feed_turn)
            heading = wrap_pi(heading + dtheta)
        emit_straight(out, ds, feed_lin)
        x, y = nx, ny

    def go_to_point_soft_row(nx: float, ny: float):
        nonlocal x, y
        dx = nx - x
        dy = ny - y

        if abs(dy) < soft_min_dy_mm:
            go_to_point(nx, ny)
            return

        dy_half = dy / 2.0
        base = abs(dy_half) / tan_a
        a = dx / 2.0

        s = 1.0 if a >= 0 else -1.0
        D = base + abs(a)

        wx = (x + nx) / 2.0 + s * D
        wy = (y + ny) / 2.0

        go_to_point(wx, wy)
        go_to_point(nx, ny)

    def go_to_point_turn90(nx: float, ny: float):
        nonlocal x, y, heading

        dy = ny - y
        if abs(dy) > 1e-9:
            target_heading = snap_axis_heading(math.pi / 2.0 if dy > 0 else -math.pi / 2.0)
            dtheta = wrap_pi(target_heading - heading)
            if abs(dtheta) > 1e-9:
                emit_turn_in_place(out, dtheta, feed_turn)
                heading = target_heading
            heading = target_heading
            emit_straight(out, abs(dy), feed_lin)
            y = ny

        dx = nx - x
        if abs(dx) > 1e-9:
            target_heading = snap_axis_heading(0.0 if dx > 0 else math.pi)
            dtheta = wrap_pi(target_heading - heading)
            if abs(dtheta) > 1e-9:
                emit_turn_in_place(out, dtheta, feed_turn)
                heading = target_heading
            heading = target_heading
            emit_straight(out, abs(dx), feed_lin)
            x = nx

    def snap_axis_heading(theta: float) -> float:
        axes = [0.0, math.pi, math.pi / 2.0, -math.pi / 2.0]
        return min(axes, key=lambda a: abs(wrap_pi(theta - a)))

    def _smoketest_real90() -> None:
        """Minimal sanity check for REAL 90 orthogonality."""
        sample_paths = [
            [PrimLine((0.0, 0.0), (20.0, 0.0))],
            [PrimLine((20.0, 5.0), (0.0, 5.0))],
            [PrimLine((0.0, 10.0), (20.0, 10.0))],
        ]
        ac = lines_to_acode(
            paths=sample_paths,
            feed_lin=1200,
            feed_turn=800,
            row_angle_deg=row_angle_deg,
            soft_min_dy_mm=soft_min_dy_mm,
            line_advance="real90",
            scan="ltr",
        )
        print("\n".join(ac))

    def axis_move_to(nx: float, ny: float, preferred_heading: float):
        """Axis-aligned move for 90-degree modes: Y first, then X."""
        nonlocal x, y, heading

        dy = ny - y
        if abs(dy) > 1e-9:
            axis_y = snap_axis_heading(math.pi / 2.0 if dy > 0 else -math.pi / 2.0)
            dtheta = wrap_pi(axis_y - heading)
            if abs(dtheta) > 1e-9:
                emit_turn_in_place(out, dtheta, feed_turn)
            heading = axis_y
            emit_straight(out, abs(dy), feed_lin)
            y = ny

        dx = nx - x
        axis_x = snap_axis_heading(0.0 if dx >= 0 else math.pi)
        dtheta = wrap_pi(axis_x - heading)
        if abs(dtheta) > 1e-9:
            emit_turn_in_place(out, dtheta, feed_turn)
        heading = axis_x

        if abs(dx) > 1e-9:
            emit_straight(out, abs(dx), feed_lin)
            x = nx
        elif abs(wrap_pi(preferred_heading - heading)) > 1e-9:
            # snap heading to preferred row direction even if no X move is needed
            axis_pref = snap_axis_heading(preferred_heading)
            dtheta2 = wrap_pi(axis_pref - heading)
            if abs(dtheta2) > 1e-9:
                emit_turn_in_place(out, dtheta2, feed_turn)
            heading = axis_pref

    def go_to_point_real90(nx: float, ny: float, target_heading: float):
        nonlocal x, y, heading

        axis_move_to(nx, ny, target_heading)

    for path in paths:
        row_heading = 0.0
        if path:
            dx_row = path[0].p1[0] - path[0].p0[0]
            row_heading = 0.0 if dx_row >= 0 else math.pi

        set_pen(False)
        if soft_rows:
            go_to_point_soft_row(path[0].p0[0], path[0].p0[1])
            set_pen(True)
            for ln in path:
                go_to_point(ln.p1[0], ln.p1[1])
        else:
            axis_move_to(path[0].p0[0], path[0].p0[1], row_heading)
            set_pen(True)
            for ln in path:
                axis_move_to(ln.p1[0], ln.p1[1], row_heading)

    set_pen(False)
    out.append(END_CMD)
    return out

# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="ART PNG -> ACODE (horizontal scanline, faithful to source)")
    ap.add_argument("png", help="Input PNG path")
    ap.add_argument("--img-width-mm", type=float, required=True, help="Target width in mm")
    ap.add_argument("-o", "--out", help="Output .acode path")

    ap.add_argument("--work-width-px", type=int, default=1600, help="Internal working width in pixels")
    ap.add_argument("--line-spacing-mm", type=float, default=0.7, help="Scanline spacing in mm")
    ap.add_argument("--threshold", type=int, default=160, help="0..255 (lower draws more)")
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma correction")
    ap.add_argument("--min-segment-mm", type=float, default=1.0, help="Skip short segments")
    ap.add_argument("--invert", action="store_true", help="Invert grayscale before processing")

    ap.add_argument("--margin-mm", type=float, default=0.0, help="Offset everything by margin (mm)")
    ap.add_argument("--flip-y", action="store_true", help="Flip Y axis")
    ap.add_argument("--y-order", choices=["top-down", "bottom-up"], default="top-down", help="Row traversal order")

    ap.add_argument("--scan", choices=["serpentine", "ltr", "ltr+rtl"], default="serpentine", help="Row direction strategy")

    # Antybanding
    ap.add_argument("--y-jitter-mm", type=float, default=0.04, help="Random Y offset per row. Use 0 to disable.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for repeatable jitter")

    # X detection mode
    ap.add_argument("--x-mode", choices=["pixel", "step"], default="pixel", help="pixel keeps edges faithful, step is faster")
    ap.add_argument("--x-step-mm", type=float, default=0.25, help="Used only when x-mode=step")

    # Fast row change without 90-degree turn
    ap.add_argument("--row-angle-deg", type=float, default=18.0, help="Max angle to X for row-advance legs (15-20)")
    ap.add_argument("--soft-min-dy-mm", type=float, default=0.3, help="Apply soft row-advance only if |dy| >= this value")

    ap.add_argument("--line-advance", choices=["soft", "turn90", "real90"], default="soft", help="Row change mode")

    ap.add_argument("--feed-lin", type=int, default=1200)
    ap.add_argument("--feed-turn", type=int, default=800)
    args = ap.parse_args()

    out_path = args.out or (os.path.splitext(args.png)[0] + ".acode")

    gray = load_grayscale(args.png)
    gray = invert_if(gray, args.invert)
    gray = resize_to_width_px(gray, args.work_width_px)

    h_px, w_px = gray.shape
    mm_per_px = mm_per_px_from_width(args.img_width_mm, w_px)
    height_mm = h_px * mm_per_px

    rows = build_scanline_rows_segments_mm(
        gray=gray,
        mm_per_px=mm_per_px,
        line_spacing_mm=args.line_spacing_mm,
        threshold=args.threshold,
        gamma=args.gamma,
        min_segment_mm=args.min_segment_mm,
        y_jitter_mm=args.y_jitter_mm,
        seed=args.seed,
        x_mode=args.x_mode,
        x_step_mm=args.x_step_mm,
    )

    paths = build_scanline_paths_no_x_drift(
        rows=rows,
        img_width_mm=args.img_width_mm,
        margin_mm=args.margin_mm,
        flip_y=args.flip_y,
        height_mm=height_mm,
        scan=args.scan,
        y_order=args.y_order,
    )

    # Jazda w dół->góra z miękkim przejściem potrafi przecinać istniejące linie.
    # Wymuś osiowy przebieg między wierszami (turn90) w tym trybie, chyba że użytkownik wybierze coś innego.
    line_advance_effective = args.line_advance
    if args.y_order == "bottom-up" and args.line_advance == "soft":
        line_advance_effective = "turn90"

    acode = lines_to_acode(
        paths=paths,
        feed_lin=args.feed_lin,
        feed_turn=args.feed_turn,
        row_angle_deg=args.row_angle_deg,
        soft_min_dy_mm=args.soft_min_dy_mm,
        line_advance=line_advance_effective,
        scan=args.scan,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(acode) + "\n")

    print("OK")
    print("written:", out_path)
    print("img_width_mm:", f"{args.img_width_mm:.3f}")
    print("img_height_mm:", f"{height_mm:.3f}")
    print("work_px:", w_px, "x", h_px)
    print("rows:", len(rows))
    print("scan:", args.scan)
    print("x_mode:", args.x_mode)
    if args.x_mode == "step":
        print("x_step_mm:", f"{args.x_step_mm:.3f}")
    print("row_angle_deg:", f"{args.row_angle_deg:.2f}")
    print("y_jitter_mm:", f"{args.y_jitter_mm:.3f}")
    if args.seed is not None:
        print("seed:", args.seed)
    print("soft_min_dy_mm:", f"{args.soft_min_dy_mm:.3f}")
    print("wheelbase_mm:", WHEELBASE_MM)
    print("steps_per_mm:", f"{STEPS_PER_MM:.9f}")
    print("turn_gain:", f"{TURN_GAIN:.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
