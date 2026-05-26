#!/usr/bin/env python3
"""
t2m_client.py — client for t2m_infer --server (cross-platform, LE protocol).

Protocol (all little-endian, loopback only):

  request:
    [u32]    prompt_len_bytes
    [bytes]  prompt utf-8
    [f32]    duration_seconds
    [u32]    seed

  response ok (status==0):
    [u32]    status (0)
    [u32]    T  (frames)
    [u32]    D  (= 135)
    [bytes]  motion fp32 LE   (T * D * 4)
    [f32]    hidden_ms
    [f32]    refiner_ms
    [f32]    dit_total_ms
    [f32]    dit_step0_ms
    [f32]    dit_step1_3_avg_ms

  response err (status==1):
    [u32]    status (1)
    [u32]    msg_len
    [bytes]  msg utf-8

Usage:
  python t2m_client.py "A person walks forward"
  python t2m_client.py --bench prompts.txt --warmup 2
  python t2m_client.py "..." --duration 3.0 --seed 42 --out motion.bin
"""

import argparse
import socket
import struct
import sys
import time

import numpy as np


def call_t2m(host, port, prompt, duration=3.0, seed=42):
    """
    Returns:
        motion: (T, 135) float32
        timing: dict with hidden_ms, refiner_ms, dit_total_ms,
                          dit_step0_ms, dit_step1_3_avg_ms
    """
    s = socket.create_connection((host, port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        # ---- request ----
        pb = prompt.encode("utf-8")
        s.sendall(struct.pack("<I", len(pb)))    # u32 LE
        s.sendall(pb)
        s.sendall(struct.pack("<f", duration))   # f32 LE
        s.sendall(struct.pack("<I", seed))       # u32 LE (NOT u64)

        def recv_exact(n):
            buf = b""
            while len(buf) < n:
                chunk = s.recv(n - len(buf))
                if not chunk:
                    raise ConnectionError("server closed mid-read")
                buf += chunk
            return buf

        # ---- response ----
        (status,) = struct.unpack("<I", recv_exact(4))
        if status != 0:
            (mlen,) = struct.unpack("<I", recv_exact(4))
            err = recv_exact(mlen).decode("utf-8", errors="replace")
            raise RuntimeError(f"server error: {err}")

        (T,) = struct.unpack("<I", recv_exact(4))
        (D,) = struct.unpack("<I", recv_exact(4))
        if D != 135:
            raise RuntimeError(f"expected motion_dim=135, got {D}")

        data = recv_exact(T * D * 4)
        motion = np.frombuffer(data, dtype="<f4").reshape(T, D).copy()

        (hidden_ms,)     = struct.unpack("<f", recv_exact(4))
        (refiner_ms,)    = struct.unpack("<f", recv_exact(4))
        (dit_total_ms,)  = struct.unpack("<f", recv_exact(4))
        (dit_step0_ms,)  = struct.unpack("<f", recv_exact(4))
        (dit_rest_avg,)  = struct.unpack("<f", recv_exact(4))

        return motion, {
            "hidden_ms":          hidden_ms,
            "refiner_ms":         refiner_ms,
            "dit_total_ms":       dit_total_ms,
            "dit_step0_ms":       dit_step0_ms,
            "dit_step1_3_avg_ms": dit_rest_avg,
        }
    finally:
        s.close()


def fmt_timing(t):
    return (f"hidden={t['hidden_ms']:5.1f}  "
            f"refiner={t['refiner_ms']:5.2f}  "
            f"dit_total={t['dit_total_ms']:6.2f}  "
            f"step0={t['dit_step0_ms']:5.2f}  "
            f"rest_avg={t['dit_step1_3_avg_ms']:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", help="single prompt (or use --bench)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8423)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="motion.bin")
    ap.add_argument("--bench", help="path to prompts file (one per line)")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    if args.bench:
        with open(args.bench) as f:
            prompts = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        print(f"loaded {len(prompts)} prompts from {args.bench}")
        if not prompts:
            print("no prompts, abort"); sys.exit(1)

        print(f"warmup {args.warmup}x...")
        for i in range(args.warmup):
            p = prompts[i % len(prompts)]
            t0 = time.perf_counter()
            _, t = call_t2m(args.host, args.port, p, args.duration, args.seed)
            wc = (time.perf_counter() - t0) * 1000
            print(f"  warmup {i+1}: client_wall={wc:6.1f}ms  {fmt_timing(t)}")

        print(f"\ntimed {len(prompts)} runs...")
        rows = []
        for i, p in enumerate(prompts):
            t0 = time.perf_counter()
            _, t = call_t2m(args.host, args.port, p, args.duration, args.seed)
            wc = (time.perf_counter() - t0) * 1000
            rows.append((wc, t["hidden_ms"], t["refiner_ms"], t["dit_total_ms"]))
            print(f"  [{i+1:3d}/{len(prompts)}] wall={wc:6.1f}  {fmt_timing(t)}  "
                  f"\"{p[:50]}{'...' if len(p)>50 else ''}\"")

        a = np.array(rows)
        print("\n" + "=" * 70)
        print(f"stats over {len(rows)} runs (ms):")
        for i, name in enumerate(["client_wall", "hidden", "refiner", "dit_total"]):
            col = a[:, i]
            print(f"  {name:12s}  mean={col.mean():7.2f}  p50={np.median(col):7.2f}  "
                  f"p95={np.percentile(col, 95):7.2f}  "
                  f"min={col.min():7.2f}  max={col.max():7.2f}")
        print("=" * 70)
        return

    if not args.prompt:
        print("ERROR: prompt required (or use --bench)"); sys.exit(1)

    print(f"calling {args.host}:{args.port} ...")
    t0 = time.perf_counter()
    motion, t = call_t2m(args.host, args.port, args.prompt, args.duration, args.seed)
    wc = (time.perf_counter() - t0) * 1000

    print(f"motion: shape={motion.shape}  dtype={motion.dtype}")
    print(f"  abs_max={np.abs(motion).max():.3f}  "
          f"mean={motion.mean():.4f}  std={motion.std():.3f}")
    print(f"timing:")
    print(f"  client_wall:   {wc:7.2f} ms")
    print(f"  hidden:        {t['hidden_ms']:7.2f} ms")
    print(f"  refiner:       {t['refiner_ms']:7.2f} ms")
    print(f"  dit_total:     {t['dit_total_ms']:7.2f} ms")
    print(f"    step0:       {t['dit_step0_ms']:7.2f} ms")
    print(f"    rest_avg:    {t['dit_step1_3_avg_ms']:7.2f} ms")

    motion.astype("<f4").tofile(args.out)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()