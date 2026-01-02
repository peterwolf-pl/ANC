#!/usr/bin/env python3
# DXF (mm) -> ACODE for differential drive plotter
# Commands: H, P U/P D, W L.. R.. F.., END
#
# Features:
# - Lines: one-shot W
# - ARC/CIRCLE: one-shot differential W
# - LWPOLYLINE bulge: converted to one-shot ARC per segment
# - TURN_GAIN calibration (measured in-place turn)

import argparse
import math
import os
from typing import List, Tuple, Optional, Union

import ezdxf
import numpy as np

Point = Tuple[float, float]

# ----------------------------
# Machine calibration
# ----------------------------
WHEELBASE_MM = 255.0
WHEEL_CIRCUM_MM = 175.0
STEPS_PER_REV = 200
MICROSTEPS = 8

STEPS_PER_MM = (STEPS_PER_REV * MICROSTEPS) / WHEEL_CIRCUM_MM  # 9.142857...

# Measured in-place 360 deg turn steps per wheel
TURN_STEPS_360_MEASURED = 7400
_steps_nominal_360 = (WHEELBASE_MM * math.pi) * STEPS_PER_MM
TURN_GAIN = TURN_STEPS_360_MEASURED / _steps_nominal_360

# ----------------------------
# ACODE
# ----------------------------
HOME_CMD = "H"
PEN_UP_CMD = "P U"
PEN_DOWN_CMD = "P D"
END_CMD = "END"

# ----------------------------
# Simplification
# ----------------------------
DEFAULT_FLAT_STEP_MM = 1.0
DEFAULT_EPSILON_MM = 0.25

# ----------------------------
# Helpers
# ----------------------------
def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])

def rdp(points: List[Point], eps: float) -> List[Point]:
    if len(points) <= 2:
        return points

    p0 = np.array(points[0], dtype=float)
    p1 = np.array(points[-1], dtype=float)
    v = p1 - p0
    vlen = float(np.linalg.norm(v))
    if vlen < 1e-12:
        return [points[0], points[-1]]

    vu = v / vlen
    max_d = -1.0
    max_i = -1
    for i in range(1, len(points) - 1):
        pi = np.array(points[i], dtype=float)
        w = pi - p0
        proj = float(np.dot(w, vu))
        perp = w - proj * vu
        d = float(np.linalg.norm(perp))
        if d > max_d:
            max_d = d
            max_i = i

    if max_d <= eps:
        return [points[0], points[-1]]

    left = rdp(points[: max_i + 1], eps)
    right = rdp(points[max_i:], eps)
    return left[:-1] + right

def steps_from_mm(mm: float) -> int:
    return int(round(mm * STEPS_PER_MM))

# ----------------------------
# Path primitives
# ----------------------------
class PrimLine:
    __slots__ = ("p0", "p1")
    def __init__(self, p0: Point, p1: Point):
        self.p0 = p0
        self.p1 = p1

class PrimArc:
    __slots__ = ("center", "radius", "a0", "dtheta")
    # a0 in radians, dtheta signed (CCW positive)
    def __init__(self, center: Point, radius: float, a0: float, dtheta: float):
        self.center = center
        self.radius = radius
        self.a0 = a0
        self.dtheta = dtheta

Primitive = Union[PrimLine, PrimArc]

def prim_start(p: Primitive) -> Point:
    if isinstance(p, PrimLine):
        return p.p0
    return (p.center[0] + p.radius * math.cos(p.a0),
            p.center[1] + p.radius * math.sin(p.a0))

def prim_end(p: Primitive) -> Point:
    if isinstance(p, PrimLine):
        return p.p1
    a1 = p.a0 + p.dtheta
    return (p.center[0] + p.radius * math.cos(a1),
            p.center[1] + p.radius * math.sin(a1))

def path_start(path: List[Primitive]) -> Point:
    return prim_start(path[0])

def path_end(path: List[Primitive]) -> Point:
    return prim_end(path[-1])

def reverse_primitive(p: Primitive) -> Primitive:
    if isinstance(p, PrimLine):
        return PrimLine(p.p1, p.p0)
    # reverse arc: start becomes old end, sweep negated
    return PrimArc(
        center=p.center,
        radius=p.radius,
        a0=p.a0 + p.dtheta,
        dtheta=-p.dtheta,
    )

def reorder_nearest(paths: List[List[Primitive]]) -> List[List[Primitive]]:
    cur = (0.0, 0.0)
    left = paths[:]
    out: List[List[Primitive]] = []

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
            path = [reverse_primitive(p) for p in reversed(path)]
        out.append(path)
        cur = path_end(path)

    return out

# ----------------------------
# Bulge conversion
# ----------------------------
def bulge_to_arc(p0: Point, p1: Point, bulge: float) -> Optional[PrimArc]:
    # bulge = tan(theta/4), theta signed (CCW positive)
    if abs(bulge) < 1e-12:
        return None

    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    c = math.hypot(dx, dy)
    if c < 1e-12:
        return None

    dtheta = 4.0 * math.atan(bulge)  # signed
    adtheta = abs(dtheta)

    # radius
    s = math.sin(adtheta / 2.0)
    if abs(s) < 1e-12:
        return None
    R = c / (2.0 * s)

    # midpoint
    mx = (x0 + x1) / 2.0
    my = (y0 + y1) / 2.0

    # unit chord vector
    ux = dx / c
    uy = dy / c
    # left normal
    nx = -uy
    ny = ux

    # distance from midpoint to center
    h = math.sqrt(max(0.0, R * R - (c / 2.0) * (c / 2.0)))

    # For CCW (bulge > 0), center is to the left of chord
    if bulge > 0:
        cx = mx + nx * h
        cy = my + ny * h
    else:
        cx = mx - nx * h
        cy = my - ny * h

    a0 = math.atan2(y0 - cy, x0 - cx)
    return PrimArc(center=(cx, cy), radius=R, a0=a0, dtheta=dtheta)

# ----------------------------
# DXF extraction
# ----------------------------
def sample_spline(e, step: float) -> List[Point]:
    tool = e.construction_tool()
    length = tool.approximate_length(segments=200)
    n = max(8, int(math.ceil(length / max(step, 1e-9))))
    pts = [tool.point(t) for t in np.linspace(0, 1, n + 1)]
    return [(float(p.x), float(p.y)) for p in pts]

def arc_from_dxf_entity(center: Point, radius: float, a0: float, a1: float) -> PrimArc:
    # DXF ARC is CCW from start_angle to end_angle (mod 360)
    d = a1 - a0
    while d < 0.0:
        d += 2.0 * math.pi
    while d >= 2.0 * math.pi:
        d -= 2.0 * math.pi
    return PrimArc(center=center, radius=radius, a0=a0, dtheta=+d)

def extract_primitives(doc, layer: Optional[str], flat_step: float, epsilon: float) -> List[List[Primitive]]:
    msp = doc.modelspace()
    paths: List[List[Primitive]] = []

    for e in msp:
        if layer and e.dxf.layer != layer:
            continue

        t = e.dxftype()

        if t == "LINE":
            p0 = (float(e.dxf.start.x), float(e.dxf.start.y))
            p1 = (float(e.dxf.end.x), float(e.dxf.end.y))
            paths.append([PrimLine(p0, p1)])

        elif t == "LWPOLYLINE":
            # Handle bulge per segment
            pts_raw = list(e.get_points("xyb"))  # (x,y,bulge)
            if len(pts_raw) < 2:
                continue

            pts: List[Tuple[float, float, float]] = [(float(x), float(y), float(b)) for x, y, b in pts_raw]
            closed = bool(e.closed)

            prims: List[Primitive] = []
            n = len(pts)
            last = n if not closed else n + 1

            for i in range(1, last):
                x0, y0, b0 = pts[i - 1]
                x1, y1, _b1 = pts[i % n]
                p0 = (x0, y0)
                p1 = (x1, y1)

                if abs(b0) < 1e-12:
                    prims.append(PrimLine(p0, p1))
                else:
                    arc = bulge_to_arc(p0, p1, b0)
                    if arc is None:
                        prims.append(PrimLine(p0, p1))
                    else:
                        prims.append(arc)

            paths.append(prims)

        elif t == "POLYLINE":
            pts: List[Point] = []
            for v in e.vertices:
                pts.append((float(v.dxf.location.x), float(v.dxf.location.y)))
            closed = bool(e.is_closed)

            if len(pts) >= 2:
                if closed and pts[0] != pts[-1]:
                    pts.append(pts[0])
                pts = rdp(pts, epsilon)
                prims: List[Primitive] = []
                for i in range(1, len(pts)):
                    prims.append(PrimLine(pts[i - 1], pts[i]))
                paths.append(prims)

        elif t == "ARC":
            c = (float(e.dxf.center.x), float(e.dxf.center.y))
            r = float(e.dxf.radius)
            a0 = math.radians(float(e.dxf.start_angle))
            a1 = math.radians(float(e.dxf.end_angle))
            paths.append([arc_from_dxf_entity(c, r, a0, a1)])

        elif t == "CIRCLE":
            c = (float(e.dxf.center.x), float(e.dxf.center.y))
            r = float(e.dxf.radius)
            paths.append([PrimArc(center=c, radius=r, a0=0.0, dtheta=2.0 * math.pi)])

        elif t == "SPLINE":
            pts = sample_spline(e, flat_step)
            if len(pts) >= 2:
                pts = rdp(pts, epsilon)
                prims: List[Primitive] = []
                for i in range(1, len(pts)):
                    prims.append(PrimLine(pts[i - 1], pts[i]))
                paths.append(prims)

    return [p for p in paths if p]

# ----------------------------
# ACODE emit
# ----------------------------
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

def emit_arc_one_shot(out: List[str], radius: float, dtheta: float, feed_arc: int):
    # One-shot arc:
    # dL = R*theta - TURN_GAIN*(W/2)*theta
    # dR = R*theta + TURN_GAIN*(W/2)*theta
    dl = (radius * dtheta) - (TURN_GAIN * (WHEELBASE_MM / 2.0) * dtheta)
    dr = (radius * dtheta) + (TURN_GAIN * (WHEELBASE_MM / 2.0) * dtheta)
    emit_w(out, steps_from_mm(dl), steps_from_mm(dr), feed_arc)

def tangent_heading_for_arc(a0: float, dtheta: float) -> float:
    ccw = dtheta > 0
    return wrap_pi(a0 + (math.pi / 2.0 if ccw else -math.pi / 2.0))

def primitives_to_acode(
    paths: List[List[Primitive]],
    feed_lin: int,
    feed_turn: int,
    feed_arc: int,
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
        sx, sy = path_start(path)
        go_to_point(sx, sy)

        set_pen(True)

        for prim in path:
            if isinstance(prim, PrimLine):
                go_to_point(prim.p1[0], prim.p1[1])
                continue

            # ARC (including bulge arcs)
            start_tan = tangent_heading_for_arc(prim.a0, prim.dtheta)
            dtheta_align = wrap_pi(start_tan - heading)
            if abs(dtheta_align) > 1e-9:
                emit_turn_in_place(out, dtheta_align, feed_turn)
                heading = wrap_pi(heading + dtheta_align)

            if abs(prim.dtheta) > 1e-9:
                emit_arc_one_shot(out, prim.radius, prim.dtheta, feed_arc)

            ex, ey = prim_end(prim)
            x, y = ex, ey
            heading = wrap_pi(heading + prim.dtheta)

    set_pen(False)
    out.append(END_CMD)
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="DXF(mm) -> ACODE with ARC one-shot and LWPOLYLINE bulge support")
    ap.add_argument("dxf")
    ap.add_argument("-o", "--out")
    ap.add_argument("--layer")
    ap.add_argument("--no-reorder", action="store_true")

    ap.add_argument("--feed-lin", type=int, default=1200)
    ap.add_argument("--feed-turn", type=int, default=800)
    ap.add_argument("--feed-arc", type=int, default=800)

    ap.add_argument("--flat-step", type=float, default=DEFAULT_FLAT_STEP_MM)
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON_MM)
    args = ap.parse_args()

    out_path = args.out or (os.path.splitext(args.dxf)[0] + ".acode")

    doc = ezdxf.readfile(args.dxf)

    paths = extract_primitives(doc, args.layer, args.flat_step, args.epsilon)
    if not paths:
        raise SystemExit("Brak geometrii w modelspace.")

    if not args.no_reorder:
        paths = reorder_nearest(paths)

    lines = primitives_to_acode(
        paths=paths,
        feed_lin=args.feed_lin,
        feed_turn=args.feed_turn,
        feed_arc=args.feed_arc,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("OK")
    print("written:", out_path)
    print("wheelbase_mm:", WHEELBASE_MM)
    print("steps_per_mm:", f"{STEPS_PER_MM:.9f}")
    print("turn_gain:", f"{TURN_GAIN:.6f}")

if __name__ == "__main__":
    main()