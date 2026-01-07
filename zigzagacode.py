#!/usr/bin/env python3
"""PNG -> ACODE generator with explicit ZigZag motion.

This script was written from scratch to drive the machine in a true zig-zag:
- Forward pass goes left→right (heading +X).
- Return pass drives right→left while keeping the same heading (+X) and
  moving backwards along X (no 180° turn between rows).
- Row changes use two 90° turns (up/down) to step between rows.
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# ----------------------------
# Machine / kinematics
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
# ACODE helpers
# ----------------------------
HOME_CMD = "H"
PEN_UP_CMD = "P U"
PEN_DOWN_CMD = "P D"
END_CMD = "END"


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


def emit_straight_signed(out: List[str], ds: float, feed_lin: int):
    """Straight move that can go forward (ds>0) or backward (ds<0)."""
    s = steps_from_mm(ds)
    emit_w(out, s, s, feed_lin)


# ----------------------------
# Image helpers (minimal subset)
# ----------------------------
def mm_per_px_from_width(img_width_mm: float, width_px: int) -> float:
    if width_px <= 0:
        raise ValueError("width_px must be > 0")
    return img_width_mm / float(width_px)


def load_grayscale(path: str) -> np.ndarray:
    return Image.open(path).convert("L")


def invert_if(im: Image.Image, invert: bool) -> Image.Image:
    if not invert:
        return im
    return Image.fromarray(255 - np.asarray(im))


def resize_to_width_px(im: Image.Image, target_width_px: int) -> Image.Image:
    w, h = im.size
    if w == target_width_px:
        return im
    ratio = target_width_px / float(w)
    new_h = max(1, int(round(h * ratio)))
    return im.resize((target_width_px, new_h), resample=Image.BILINEAR)


def runs_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start = i
            in_run = True
        if (not v) and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def build_scanline_rows_segments_mm(
    gray: np.ndarray,
    mm_per_px: float,
    line_spacing_mm: float,
    threshold: int,
    gamma: float,
    min_segment_mm: float,
    y_jitter_mm: float,
    seed: Optional[int],
    x_mode: str,
    x_step_mm: float,
) -> List[Tuple[float, List[Tuple[float, float]]]]:
    """
    Returns [(y_mm, [(x1_mm, x2_mm), ...]), ...]
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
# ZigZag ACODE builder
# ----------------------------
def zigzag_acode_from_rows(
    rows: List[Tuple[float, List[Tuple[float, float]]]],
    img_width_mm: float,
    height_mm: float,
    margin_mm: float,
    flip_y: bool,
    y_order: str,
    feed_lin: int,
    feed_turn: int,
) -> List[str]:
    if not rows:
        return [HOME_CMD, END_CMD]

    # Normalize row order and coordinates
    ys: List[float] = []
    seg_rows: List[List[Tuple[float, float]]] = []
    for y_mm, segs in rows:
        y2 = height_mm - y_mm if flip_y else y_mm
        y2 += margin_mm
        ys.append(y2)
        seg_rows.append(segs)

    order = sorted(range(len(ys)), key=lambda i: ys[i])
    ys = [ys[i] for i in order]
    seg_rows = [seg_rows[i] for i in order]

    if y_order == "bottom-up":
        ys.reverse()
        seg_rows.reverse()

    out: List[str] = [HOME_CMD, PEN_UP_CMD]

    x = 0.0
    y = 0.0
    heading = 0.0  # keep heading along +X for all rows
    pen_down = False

    x_lo = margin_mm
    x_hi = margin_mm + img_width_mm

    def set_pen(down: bool):
        nonlocal pen_down
        if down and not pen_down:
            out.append(PEN_DOWN_CMD)
            pen_down = True
        if (not down) and pen_down:
            out.append(PEN_UP_CMD)
            pen_down = False

    def turn_to(theta: float):
        nonlocal heading
        dtheta = wrap_pi(theta - heading)
        if abs(dtheta) > 1e-9:
            emit_turn_in_place(out, dtheta, feed_turn)
            heading = wrap_pi(heading + dtheta)

    def move_y(target_y: float):
        nonlocal x, y, heading
        dy = target_y - y
        if abs(dy) < 1e-9:
            return
        turn_to(math.pi / 2.0 if dy > 0 else -math.pi / 2.0)
        emit_straight_signed(out, dy, feed_lin)
        turn_to(0.0)
        y = target_y

    def move_x_signed(target_x: float):
        nonlocal x, heading
        dx = target_x - x
        if abs(dx) < 1e-9:
            return
        # Keep heading at 0; sign of dx decides direction (forward/backward).
        turn_to(0.0)
        emit_straight_signed(out, dx, feed_lin)
        x = target_x

    for idx_row, (y_row, segs) in enumerate(zip(ys, seg_rows)):
        # Clamp segments into workspace with margin.
        segs2 = [(max(x_lo, min(x_hi, a)), max(x_lo, min(x_hi, b))) for (a, b) in segs]
        if not segs2:
            continue

        forward = (idx_row % 2 == 0)
        seg_order = segs2 if forward else list(reversed(segs2))

        # Move to row Y
        set_pen(False)
        move_y(y_row)

        # Choose starting X based on direction
        start_x = seg_order[0][0] if forward else seg_order[0][1]
        move_x_signed(start_x)

        # Draw row
        for (x1, x2) in seg_order:
            a, b = (x1, x2) if forward else (x2, x1)
            move_x_signed(a)
            set_pen(True)
            move_x_signed(b)
            set_pen(False)

        # End of row: stay at last X ready for next row; Y will move in next iteration.

    set_pen(False)
    out.append(END_CMD)
    return out


# ----------------------------
# CLI
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="PNG -> ACODE (true ZigZag, reverse return)")
    ap.add_argument("png", help="Input PNG path")
    ap.add_argument("--img-width-mm", type=float, required=True, help="Target width in mm")
    ap.add_argument("-o", "--out", help="Output .acode path")

    ap.add_argument("--work-width-px", type=int, default=1600, help="Internal working width in pixels")
    ap.add_argument("--line-spacing-mm", type=float, default=0.7, help="Scanline spacing in mm")
    ap.add_argument("--threshold", type=int, default=160, help="0..255 (lower draws more)")
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma correction")
    ap.add_argument("--min-segment-mm", type=float, default=1.0, help="Skip short segments")
    ap.add_argument("--invert", action="store_true", help="Invert grayscale before processing")

    ap.add_argument("--margin-mm", type=float, default=0.0, help="Margin added around drawing")
    ap.add_argument("--flip-y", action="store_true", help="Flip Y axis")
    ap.add_argument("--y-order", choices=["top-down", "bottom-up"], default="top-down", help="Row order")

    ap.add_argument("--y-jitter-mm", type=float, default=0.02, help="Random Y jitter per row (0 to disable)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for jitter")
    ap.add_argument("--x-mode", choices=["pixel", "step"], default="pixel", help="X sampling mode")
    ap.add_argument("--x-step-mm", type=float, default=0.25, help="Used only for x-mode=step")

    ap.add_argument("--feed-lin", type=int, default=1200, help="Feed for straight moves")
    ap.add_argument("--feed-turn", type=int, default=800, help="Feed for in-place turns")

    args = ap.parse_args()

    out_path = args.out or (os.path.splitext(args.png)[0] + ".zigzag.acode")

    im = load_grayscale(args.png)
    im = invert_if(im, args.invert)
    im = resize_to_width_px(im, args.work_width_px)

    gray = np.asarray(im)
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

    acode = zigzag_acode_from_rows(
        rows=rows,
        img_width_mm=args.img_width_mm,
        height_mm=height_mm,
        margin_mm=args.margin_mm,
        flip_y=args.flip_y,
        y_order=args.y_order,
        feed_lin=args.feed_lin,
        feed_turn=args.feed_turn,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(acode) + "\n")

    print(f"Wrote {len(acode)} ACODE lines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
