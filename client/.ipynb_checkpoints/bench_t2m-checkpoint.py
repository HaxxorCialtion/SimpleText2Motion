#!/usr/bin/env python3
"""
bench_t2m.py — benchmark the t2m_infer server.

Runs <warmup> warm calls (timings discarded) then <n> timed calls, and
reports mean/p50/p95/min/max for each stage:
  client_wall   end-to-end as the client sees it (includes TCP RTT)
  hidden        fused_server LLM forward (port 8421)
  refiner       birefiner.onnx (text_hidden -> text_tokens, pooled)
  dit_total     4-step Euler DiT loop
  dit_step0     first DiT step (often slower; arena/mem pattern setup)
  dit_rest_avg  average of steps 1..3

Usage:
  python client/bench_t2m.py                          # 5 prompts × 3 warmup + 20 timed
  python client/bench_t2m.py --n 50 --warmup 5
  python bench_t2m.py --prompts prompts.txt
  python bench_t2m.py --duration 3.0 --seed 42 --csv bench.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the production client to avoid protocol drift
from t2m_client import call_t2m


DEFAULT_PROMPTS = [
    "A person walks forward.",
    "A person waves their right hand.",
    "A person jumps in place.",
    "A person turns left and looks around.",
    "A person sits down slowly.",
]


def load_prompts(path):
    if path and Path(path).exists():
        with open(path) as f:
            ps = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        if not ps:
            print(f"prompts file {path} is empty, using defaults", file=sys.stderr)
            return DEFAULT_PROMPTS
        return ps
    return DEFAULT_PROMPTS


def summarize(name, values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "name":  name,
        "n":     len(a),
        "mean":  float(a.mean()),
        "p50":   float(np.median(a)),
        "p95":   float(np.percentile(a, 95)),
        "min":   float(a.min()),
        "max":   float(a.max()),
        "std":   float(a.std()),
    }


def print_stats_table(stats_rows):
    print()
    print("=" * 78)
    print(f"  {'stage':<14s} {'mean':>9s} {'p50':>9s} {'p95':>9s} "
          f"{'min':>9s} {'max':>9s} {'std':>9s}")
    print("-" * 78)
    for s in stats_rows:
        print(f"  {s['name']:<14s} {s['mean']:>8.1f}ms {s['p50']:>8.1f}ms "
              f"{s['p95']:>8.1f}ms {s['min']:>8.1f}ms {s['max']:>8.1f}ms "
              f"{s['std']:>8.1f}ms")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8423)
    ap.add_argument("--prompts", default="prompts.txt",
                    help="prompts file (one per line). Falls back to a built-in set.")
    ap.add_argument("--n", type=int, default=20, help="timed runs")
    ap.add_argument("--warmup", type=int, default=3, help="warmup runs")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None,
                    help="optional CSV output of every timed run")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    print(f"server   : {args.host}:{args.port}")
    print(f"prompts  : {len(prompts)} loaded from "
          f"{args.prompts if Path(args.prompts).exists() else 'built-in'}")
    print(f"warmup   : {args.warmup}")
    print(f"timed    : {args.n}")
    print(f"duration : {args.duration}s   seed: {args.seed}")
    print()

    # ---- warmup ----
    print(f"warming up ({args.warmup} runs)...")
    for i in range(args.warmup):
        p = prompts[i % len(prompts)]
        t0 = time.perf_counter()
        try:
            _, t = call_t2m(args.host, args.port, p, args.duration, args.seed)
        except Exception as e:
            print(f"  warmup #{i+1} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
        wc = (time.perf_counter() - t0) * 1000
        print(f"  #{i+1}  wall={wc:6.1f}ms  "
              f"hidden={t['hidden_ms']:5.1f}  "
              f"dit={t['dit_total_ms']:5.1f}")

    # ---- timed ----
    print(f"\ntimed ({args.n} runs)...")
    rows = []  # (wall, hidden, refiner, dit_total, dit_step0, dit_rest)
    csv_lines = ["i,prompt,wall,hidden,refiner,dit_total,dit_step0,dit_rest_avg"]
    for i in range(args.n):
        p = prompts[i % len(prompts)]
        t0 = time.perf_counter()
        try:
            _, t = call_t2m(args.host, args.port, p, args.duration, args.seed)
        except Exception as e:
            print(f"  #{i+1} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
        wc = (time.perf_counter() - t0) * 1000
        rows.append((wc, t["hidden_ms"], t["refiner_ms"],
                     t["dit_total_ms"], t["dit_step0_ms"],
                     t["dit_step1_3_avg_ms"]))
        if args.csv:
            esc = p.replace('"', '""')
            csv_lines.append(f'{i+1},"{esc}",{wc:.3f},{t["hidden_ms"]:.3f},'
                             f'{t["refiner_ms"]:.3f},{t["dit_total_ms"]:.3f},'
                             f'{t["dit_step0_ms"]:.3f},{t["dit_step1_3_avg_ms"]:.3f}')
        print(f"  #{i+1:3d}  wall={wc:6.1f}ms  "
              f"hidden={t['hidden_ms']:5.1f}  "
              f"refiner={t['refiner_ms']:4.2f}  "
              f"dit={t['dit_total_ms']:5.1f} "
              f"(s0={t['dit_step0_ms']:4.1f} rest={t['dit_step1_3_avg_ms']:4.1f})  "
              f"\"{p[:36]}{'..' if len(p)>36 else ''}\"")

    a = np.array(rows, dtype=np.float64)
    stats = [
        summarize("client_wall",  a[:, 0]),
        summarize("hidden",       a[:, 1]),
        summarize("refiner",      a[:, 2]),
        summarize("dit_total",    a[:, 3]),
        summarize("dit_step0",    a[:, 4]),
        summarize("dit_rest_avg", a[:, 5]),
    ]
    print_stats_table(stats)

    # quick health checks
    print()
    if stats[0]["mean"] < 200:
        print(f"  client wall mean {stats[0]['mean']:.0f}ms — healthy.")
    else:
        print(f"  client wall mean {stats[0]['mean']:.0f}ms — slow; check CPU load.")
    print(f"  reference (Windows i5-12600KF + CPU EP): wall ~114ms / "
          f"dit_total ~97ms / dit step ~19ms")

    if args.csv:
        Path(args.csv).write_text("\n".join(csv_lines) + "\n")
        print(f"\n  wrote {args.csv}")


if __name__ == "__main__":
    main()
