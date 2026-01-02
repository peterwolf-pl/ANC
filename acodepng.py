#!/usr/bin/env python3
# PNG -> ACODE for differential drive plotter
# Output format matches acode.py:
# Commands: H, P U/P D, W L.. R.. F.., END
#
# Requires: pillow, numpy
# Install: pip install pillow numpy

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

Point = Tuple[float, float]

# ----------------------------
# Machine calibration (copied from acode.py)
# ----------------------------
WHEELBASE_MM = 255.0
WHEEL_CIRCUM_MM = 175.0
STEPS_PER_REV = 200
MICROSTEPS = 8

STEPS_PER_MM = (STEPS_PER_REV * MICROSTEPS) / WHEEL_CIRCUM_MM  # 9.142857...

TURN_STEPS_360_MEASURED = 7400
_steps_nominal_360 = (WHEELBASE_MM * math.pi) * STEPS_PER_MM
TURN_GAIN = TURN_STEPS_360_MEASURED / _steps_nominal_360

# ----------------------------
# ACODE commands (exact)
# ----------------------------
HOME_CMD = "H"
PEN_UP_CMD = "P U"
PEN_DOWN_CMD = "P D"
END_CMD = "END"

# ----------------------------
# Helpers (copied-style)
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

def build_scanline_segments_mm(
    gray: np.ndarray,
    mm_per_px: float,
    line_spacing_mm: float,
    x_step_mm: float,
    threshold: int,
    gamma: float,
    min_segment_mm: float,
) -> List[Tuple[float, float, float, float]]:
    """
    returns segments (x1_mm, y_mm, x2_mm, y_mm)
    """
    h, w = gray.shape

    line_spacing_px = max(1, int(round(line_spacing_mm / mm_per_px)))
    x_step_px = max(1, int(round(x_step_mm / mm_per_px)))
    min_segment_px = max(1, int(round(min_segment_mm / mm_per_px)))

    g = gray.astype(np.float32) / 255.0
    if gamma != 1.0:
        g = np.power(g, gamma)

    t = float(threshold) / 255.0
    segments: List[Tuple[float, float, float, float]] = []

    for y in range(0, h, line_spacing_px):
        row = g[y, :]

        xs = np.arange(0, w, x_step_px, dtype=int)
        sampled = row[xs]

        draw_mask = sampled < t
        runs = runs_from_mask(draw_mask)

        for s, e in runs:
            x1 = int(xs[s])
            x2 = int(xs[e])
            x2 = min(w - 1, x2 + (x_step_px - 1))

            if (x2 - x1) < min_segment_px:
                continue

            x1_mm = x1 * mm_per_px
            x2_mm = x2 * mm_per_px
            y_mm = y * mm_per_px
            segments.append((x1_mm, y_mm, x2_mm, y_mm))

    return segments

def serpentine_order(
    segments: List[Tuple[float, float, float, float]],
    line_spacing_mm: float,
) -> List[Tuple[float, float, float, float]]:
    if not segments:
        return []
    eps = max(1e-6, line_spacing_mm * 0.5)

    segs = sorted(segments, key=lambda s: (s[1], min(s[0], s[2])))

    rows: List[List[Tuple[float, float, float, float]]] = []
    current: List[Tuple[float, float, float, float]] = []
    cur_y: Optional[float] = None

    for seg in segs:
        y = seg[1]
        if cur_y is None or abs(y - cur_y) <= eps:
            current.append(seg)
            if cur_y is None:
                cur_y = y
        else:
            rows.append(current)
            current = [seg]
            cur_y = y
    if current:
        rows.append(current)

    out: List[Tuple[float, float, float, float]] = []
    for i, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda s: min(s[0], s[2]))
        if i % 2 == 0:
            out.extend(row_sorted)
        else:
            for s in reversed(row_sorted):
                out.append((s[2], s[1], s[0], s[3]))
    return out

# ----------------------------
# Geometry to ACODE motion (line-only)
# ----------------------------
@dataclass
class PrimLine:
    p0: Point
    p1: Point

def reorder_nearest_line_paths(paths: List[List[PrimLine]]) -> List[List[PrimLine]]:
    # Prosty nearest-neighbor po start/end, jak w acode.py
    def dist(a: Point, b: Point) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def path_start(path: List[PrimLine]) -> Point:
        return path[0].p0

    def path_end(path: List[PrimLine]) -> Point:
        return path[-1].p1

    def reverse_path(path: List[PrimLine]) -> List[PrimLine]:
        rev = []
        for ln in reversed(path):
            rev.append(PrimLine(ln.p1, ln.p0))
        return rev

    cur = (0.0, 0.0)
    left = paths[:]
    out: List[List[PrimLine]] = []

    while left:
        best_i = 0
        best_d = 1e30
        flip = False

        for i, path in enumerate(left):
            d0 = dist(cur, path_start(path))
            d1 = dist(cur, path_end(path))
            if d0 < best_d:
                best_d = d0
                best_i = i
                flip = False
            if d1 < best_d:
                best_d = d1
                best_i = i
                flip = True

        path = left.pop(best_i)
        if flip:
            path = reverse_path(path)
        out.append(path)
        cur = path_end(path)

    return out

def group_paths_scanline(
    paths: List[List[PrimLine]],
    line_spacing_mm: float,
    scan_direction: str,  # "ltr" | "serpentine"
) -> List[List[PrimLine]]:
    """
    Układa ścieżki linia po linii (rosnące Y).
    Nie robi nearest-neighbor.
    """
    if not paths:
        return []

    eps = max(1e-6, line_spacing_mm * 0.5)

    # Każdy path ma 1 segment, ale zostawiamy logikę ogólną
    def y_of_path(p: List[PrimLine]) -> float:
        return p[0].p0[1]

    def x_min_of_path(p: List[PrimLine]) -> float:
        ln = p[0]
        return min(ln.p0[0], ln.p1[0])

    def x_max_of_path(p: List[PrimLine]) -> float:
        ln = p[0]
        return max(ln.p0[0], ln.p1[0])

    # sort by y then x
    ps = sorted(paths, key=lambda p: (y_of_path(p), x_min_of_path(p)))

    rows: List[List[List[PrimLine]]] = []
    cur_row: List[List[PrimLine]] = []
    cur_y: Optional[float] = None

    for p in ps:
        y = y_of_path(p)
        if cur_y is None or abs(y - cur_y) <= eps:
            cur_row.append(p)
            if cur_y is None:
                cur_y = y
        else:
            rows.append(cur_row)
            cur_row = [p]
            cur_y = y
    if cur_row:
        rows.append(cur_row)

    out: List[List[PrimLine]] = []
    for i, row in enumerate(rows):
        # zawsze sortujemy w ramach wiersza po X
        row_sorted = sorted(row, key=lambda p: x_min_of_path(p))

        if scan_direction == "ltr":
            out.extend(row_sorted)
            continue

        # serpentine: parzyste LTR, nieparzyste RTL (odwracamy segment)
        if i % 2 == 0:
            out.extend(row_sorted)
        else:
            for p in reversed(row_sorted):
                # odwróć segment, żeby kończyć po prawej -> jechać na lewo
                ln = p[0]
                out.append([PrimLine(ln.p1, ln.p0)])

    return out

def lines_to_acode(
    paths: List[List[PrimLine]],
    feed_lin: int,
    feed_turn: int,
) -> List[str]:
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

    for path in paths:
        set_pen(False)
        go_to_point(path[0].p0[0], path[0].p0[1])

        set_pen(True)
        for ln in path:
            go_to_point(ln.p1[0], ln.p1[1])

    set_pen(False)
    out.append(END_CMD)
    return out

# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="PNG -> ACODE (H, P U/D, W L.. R.. F.., END)")
    ap.add_argument("png", help="Input PNG path")
    ap.add_argument("--img-width-mm", type=float, required=True, help="Target width in mm")
    ap.add_argument("-o", "--out", help="Output .acode path")

    ap.add_argument("--work-width-px", type=int, default=1200, help="Internal working width in pixels")
    ap.add_argument("--line-spacing-mm", type=float, default=0.6, help="Scanline spacing in mm")
    ap.add_argument("--x-step-mm", type=float, default=0.25, help="Sampling step in mm")
    ap.add_argument("--threshold", type=int, default=160, help="0..255 (lower draws more)")
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma correction")
    ap.add_argument("--min-segment-mm", type=float, default=1.0, help="Skip short segments")
    ap.add_argument("--invert", action="store_true", help="Invert grayscale before processing")

    ap.add_argument("--margin-mm", type=float, default=0.0, help="Offset everything by margin (mm)")
    ap.add_argument("--flip-y", action="store_true", help="Flip Y axis")

    # New ordering controls
    ap.add_argument(
        "--path-order",
        choices=["nearest", "scanline"],
        default="nearest",
        help="Path ordering strategy: nearest (old) or scanline (line-by-line)",
    )
    ap.add_argument(
        "--scan-direction",
        choices=["ltr", "serpentine"],
        default="serpentine",
        help="When --path-order scanline: ltr (always left->right) or serpentine (zigzag)",
    )

    # Kept for backward compatibility
    ap.add_argument("--no-reorder", action="store_true", help="Alias: force no nearest reorder (ignored in scanline)")

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

    segments = build_scanline_segments_mm(
        gray=gray,
        mm_per_px=mm_per_px,
        line_spacing_mm=args.line_spacing_mm,
        x_step_mm=args.x_step_mm,
        threshold=args.threshold,
        gamma=args.gamma,
        min_segment_mm=args.min_segment_mm,
    )

    # Segment list order is not critical now; ordering is done on paths
    # Keep old serpentine segment generation as it affects segment direction for "nearest" mode only.
    segments_serp = serpentine_order(segments, args.line_spacing_mm)

    # Convert segments -> paths of PrimLine (each segment is its own path)
    paths: List[List[PrimLine]] = []
    for x1, y, x2, _ in segments_serp:
        y2 = y

        if args.flip_y:
            y = height_mm - y
            y2 = height_mm - y2

        x1 += args.margin_mm
        x2 += args.margin_mm
        y += args.margin_mm
        y2 += args.margin_mm

        paths.append([PrimLine((x1, y), (x2, y2))])

    # Ordering
    if args.path_order == "scanline":
        paths = group_paths_scanline(
            paths=paths,
            line_spacing_mm=args.line_spacing_mm,
            scan_direction=args.scan_direction,
        )
    else:
        # nearest (old behavior) unless user forced no-reorder
        if not args.no_reorder:
            paths = reorder_nearest_line_paths(paths)

    acode = lines_to_acode(
        paths=paths,
        feed_lin=args.feed_lin,
        feed_turn=args.feed_turn,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(acode) + "\n")

    print("OK")
    print("written:", out_path)
    print("img_width_mm:", f"{args.img_width_mm:.3f}")
    print("img_height_mm:", f"{height_mm:.3f}")
    print("work_px:", w_px, "x", h_px)
    print("segments:", len(segments))
    print("path_order:", args.path_order)
    if args.path_order == "scanline":
        print("scan_direction:", args.scan_direction)
    print("wheelbase_mm:", WHEELBASE_MM)
    print("steps_per_mm:", f"{STEPS_PER_MM:.9f}")
    print("turn_gain:", f"{TURN_GAIN:.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())