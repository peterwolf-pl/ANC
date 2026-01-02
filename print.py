#!/usr/bin/env python3
# ACODE sender for ESP32 over TCP (stream view)
# - Shows only the currently sent line
# - After OK prints "OK" and moves to next line
# - Robust OK waiting with polling
# - Expected duration for W uses accel/decel model

import argparse
import socket
import sys
import time
import re
from typing import List

W_RE = re.compile(r"^W\s+L(-?\d+)\s+R(-?\d+)\s+F(\d+)\s*$")
P_RE = re.compile(r"^P\s+([UD])\s*$")

def read_acode(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip("\r\n").strip()
            if not line:
                continue
            out.append(line)
    return out

def connect_tcp(host: str, port: int, connect_timeout: float) -> socket.socket:
    s = socket.create_connection((host, port), timeout=connect_timeout)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s

def recv_line_poll(sock: socket.socket, slice_timeout: float) -> str | None:
    sock.settimeout(slice_timeout)
    buf = getattr(recv_line_poll, "_buf", bytearray())
    try:
        while True:
            b = sock.recv(1)
            if not b:
                raise ConnectionError("Connection closed by peer")
            if b == b"\n":
                line = bytes(buf).decode("utf-8", errors="replace").strip("\r")
                buf.clear()
                recv_line_poll._buf = buf
                return line.strip()
            if b != b"\r":
                buf.extend(b)
            if len(buf) > 400:
                line = bytes(buf).decode("utf-8", errors="replace")
                buf.clear()
                recv_line_poll._buf = buf
                return line.strip()
    except TimeoutError:
        recv_line_poll._buf = buf
        return None

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

    # ramp distance in steps: (v^2 - v0^2) / (2a)
    ramp_steps = (v_target * v_target - v0 * v0) / (2.0 * a)
    if ramp_steps < 0.0:
        ramp_steps = 0.0

    half = steps / 2.0
    if ramp_steps > half:
        ramp_steps = half  # triangle profile

    # time accel v0->v
    t_ramp = (v_target - v0) / a
    if t_ramp < 0.0:
        t_ramp = 0.0

    cruise_steps = steps - 2.0 * ramp_steps
    if cruise_steps < 0.0:
        cruise_steps = 0.0

    t_cruise = cruise_steps / v_target if v_target > 1e-9 else 9999.0

    return (2.0 * t_ramp) + t_cruise + 0.30

def wait_ok(
    sock: socket.socket,
    line: str,
    expected_s: float,
    hard_cap_s: float,
    poll_slice_s: float,
) -> None:
    # Deadline with safe margin
    wait_s = max(4.0, expected_s * 3.0 + 2.0)
    wait_s = min(hard_cap_s, wait_s)
    deadline = time.time() + wait_s

    while True:
        if time.time() > deadline:
            raise TimeoutError(f"no OK within {wait_s:.1f}s (expected {expected_s:.1f}s) for: {line}")

        resp = recv_line_poll(sock, poll_slice_s)
        if resp is None:
            continue

        if resp == "OK":
            return
        if resp.startswith("ERR"):
            raise RuntimeError(resp)
        # ignore noise

def print_line_status(i: int, n: int, line: str, status: str) -> None:
    # Single line output. No full list.
    sys.stdout.write(f"{i:04d}/{n:04d}  {line}  {status}\n")
    sys.stdout.flush()

def main() -> int:
    ap = argparse.ArgumentParser(description="Send ACODE to ESP32 (stream view)")
    ap.add_argument("file")
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--connect-timeout", type=float, default=33.0)

    # must match firmware
    ap.add_argument("--feed-to-sps", type=float, default=0.6)
    ap.add_argument("--min-sps", type=float, default=50.0)
    ap.add_argument("--max-sps", type=float, default=2500.0)
    ap.add_argument("--servo-settle-ms", type=int, default=180)

    # accel model (match ESP32 defaults)
    ap.add_argument("--start-sps", type=float, default=120.0)
    ap.add_argument("--accel-sps2", type=float, default=8000.0)

    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--delay-ms", type=int, default=0)

    ap.add_argument("--poll-slice", type=float, default=0.5)
    ap.add_argument("--hard-cap", type=float, default=180.0)
    ap.add_argument("--auto-eoff", action="store_true")
    args = ap.parse_args()

    lines = read_acode(args.file)
    if not lines:
        print("Empty file")
        return 2

    idx0 = max(1, args.start) - 1
    n = len(lines)

    sock = connect_tcp(args.host, args.port, args.connect_timeout)

    try:
        print(f"Connected to {args.host}:{args.port}")
        for i in range(idx0, n):
            line = lines[i]

            expected = estimate_motion_seconds(
                line=line,
                feed_to_sps=args.feed_to_sps,
                min_sps=args.min_sps,
                max_sps=args.max_sps,
                servo_settle_s=args.servo_settle_ms / 1000.0,
                start_sps=args.start_sps,
                accel_sps2=args.accel_sps2,
            )

            print_line_status(i + 1, n, line, f"SEND (est {expected:.1f}s)")
            sock.sendall((line + "\n").encode("utf-8"))

            wait_ok(
                sock=sock,
                line=line,
                expected_s=expected,
                hard_cap_s=args.hard_cap,
                poll_slice_s=args.poll_slice,
            )

            print_line_status(i + 1, n, line, "OK")

            if args.delay_ms > 0:
                time.sleep(args.delay_ms / 1000.0)

            if line == "END":
                if args.auto_eoff:
                    e_line = "E 0"
                    print_line_status(i + 1, n, e_line, "SEND")
                    sock.sendall((e_line + "\n").encode("utf-8"))
                    wait_ok(
                        sock=sock,
                        line=e_line,
                        expected_s=0.3,
                        hard_cap_s=20.0,
                        poll_slice_s=args.poll_slice,
                    )
                    print_line_status(i + 1, n, e_line, "OK")
                print("DONE")
                return 0

        print("DONE")
        return 0

    finally:
        try:
            sock.close()
        except Exception:
            pass

if __name__ == "__main__":
    raise SystemExit(main())