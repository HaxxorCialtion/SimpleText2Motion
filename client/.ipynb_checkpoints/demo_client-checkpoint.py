#!/usr/bin/env python3
"""
demo_client.py — end-to-end SimpleLove client.

Flow:
  user query
    └─ simpletool (port 8422)
         └─ function, arg1..argN, content     ← LLM decides what to do
              └─ extract motion description
                   └─ t2m (port 8423)
                        └─ motion (T, 135) float32
                             └─ write three artifacts:
                                  - motion.bin   raw SMPL-H rot6d
                                  - motion.npy   same data, numpy-friendly
                                  - motion.vrma  (via ./build/motion_to_vrma)

If simpletool returns no usable function (null, unknown tool, or simpletool
unavailable), falls back to feeding the raw user query to t2m directly. The
goal is: always produce a motion artifact unless something is genuinely
broken downstream.

Usage:
  python client/demo_client.py                                    # interactive REPL
  python client/demo_client.py "A person turns down."  # single shot
  python demo_client.py --out-dir runs/demo1 ...
  python demo_client.py --no-vrma ...                      # skip vrma step
  python demo_client.py --no-simpletool ...                # skip LLM, use raw query
"""

import argparse
import datetime
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from t2m_client import call_t2m

# Reuse the testing harness as a library — it already contains the canonical
# system prompt + tool definitions + cold/hot rendering + wire protocol.
# If you renamed the file, adjust the import.
st = None  # simpletool path will be unavailable unless one of these imports works
for _mod in ("simpletool_client", "test_simpletool_v2", "test_simpletool"):
    try:
        st = __import__(_mod)
        break
    except ImportError:
        continue


# ===========================================================================
# Helpers
# ===========================================================================

c_b = "\033[1m"; c_d = "\033[2m"; c_g = "\033[32m"; c_y = "\033[33m"
c_r = "\033[31m"; c_c = "\033[36m"; c_off = "\033[0m"

def title(s):  print(f"\n{c_b}{s}{c_off}")
def info(s):   print(f"  {s}")
def dim(s):    print(f"  {c_d}{s}{c_off}")
def good(s):   print(f"  {c_g}✓{c_off} {s}")
def warn(s):   print(f"  {c_y}!{c_off} {s}")
def bad(s):    print(f"  {c_r}✗{c_off} {s}")


def find_motion_to_vrma():
    """Locate the motion_to_vrma binary, preferring local build."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "build" / "motion_to_vrma",
        here / "build" / "Release" / "motion_to_vrma",
        Path("./build/motion_to_vrma"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    # PATH fallback
    found = shutil.which("motion_to_vrma")
    return Path(found) if found else None


# ===========================================================================
# Step 1: ask simpletool what to do
# ===========================================================================

def call_simpletool(query, environment="[]", history="[]",
                    host="127.0.0.1", port=8422, verbose=False):
    """Returns dict with keys: function, content, args[0..5], raw."""
    if st is None:
        raise RuntimeError("simpletool_client module not found; "
                           "can't run simpletool step")
    cold = st.render_cold(st.DEFAULT_SYSTEM_PROMPT, st.DEFAULT_TOOLS)
    hot  = st.render_hot(environment, history, query)
    resp = st.simpletool_request(cold, hot, host=host, port=port)
    parsed = st.parse_heads(resp["heads"])
    parsed["raw"] = resp
    if verbose:
        st.print_response(resp)
    return parsed


# ===========================================================================
# Step 2: derive a motion prompt from simpletool decision
# ===========================================================================

def motion_prompt_from_decision(decision, fallback_query):
    """
    SimpleTool returns structured tool calls; we have to translate those into
    a natural-language motion prompt the t2m model can use.

    Heuristic per known tool:
      action(verb, manner?, object?, location?) -> "{verb} {manner} {object} {location}"
      move_to(target)                           -> "walks to {target}"
      pass()                                    -> None (no motion)
      <unknown>                                 -> fallback to user query
    """
    fn = decision.get("function")
    args = decision.get("args", [None]*6)
    nonnull = [a for a in args if a]

    if not fn:
        # Null function -> no decision, fall back
        return fallback_query, "no function (null) → using raw user query"

    fn_low = fn.strip().lower()

    if fn_low == "pass":
        return None, "function=pass → no motion needed"

    if fn_low == "move_to" and nonnull:
        target = nonnull[0]
        return f"A person walks to {target}.", f"move_to({target!r})"

    if fn_low == "action":
        verb     = args[0] or ""
        manner   = args[1] or ""
        obj      = args[2] or ""
        location = args[3] or ""
        # build "A person {verb} the {obj} {manner} {location}"
        chunks = ["A person", verb.strip()]
        if obj.strip():
            # don't double-article: "the apple", "his hand" both fine
            chunks.append(obj.strip())
        if manner.strip():
            chunks.append(manner.strip())
        if location.strip():
            chunks.append(location.strip())
        sentence = " ".join(c for c in chunks if c).strip().rstrip(".") + "."
        return sentence, (f"action(verb={verb!r}, manner={manner!r}, "
                          f"obj={obj!r}, loc={location!r})")

    # Unknown function — fall back
    return fallback_query, f"unknown function {fn!r} → using raw user query"


# ===========================================================================
# Step 3: generate motion via t2m
# ===========================================================================

def run_t2m(prompt, host="127.0.0.1", port=8423,
            duration=3.0, seed=42):
    t0 = time.perf_counter()
    motion, t = call_t2m(host, port, prompt, duration, seed)
    wall = (time.perf_counter() - t0) * 1000
    t["client_wall_ms"] = wall
    return motion, t


# ===========================================================================
# Step 4: write artifacts
# ===========================================================================

def write_artifacts(motion, out_dir, prompt, simpletool_decision,
                    motion_to_vrma_bin=None, make_vrma=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. raw .bin (canonical, what every other tool consumes)
    bin_path = out_dir / "motion.bin"
    motion.astype("<f4").tofile(bin_path)
    good(f"wrote {bin_path}  ({bin_path.stat().st_size} bytes, "
         f"shape={motion.shape})")

    # 2. .npy (debug-friendly: keeps shape & dtype metadata)
    npy_path = out_dir / "motion.npy"
    np.save(npy_path, motion.astype("<f4"))
    good(f"wrote {npy_path}")

    # 3. prompt + decision metadata
    meta_path = out_dir / "meta.txt"
    with open(meta_path, "w") as f:
        f.write(f"prompt: {prompt}\n")
        f.write(f"shape:  {motion.shape}\n")
        f.write(f"dtype:  {motion.dtype}\n")
        f.write(f"abs_max: {float(np.abs(motion).max()):.4f}\n")
        f.write(f"mean:    {float(motion.mean()):.4f}\n")
        f.write(f"std:     {float(motion.std()):.4f}\n")
        if simpletool_decision:
            f.write("\nsimpletool decision:\n")
            f.write(f"  function: {simpletool_decision.get('function')!r}\n")
            f.write(f"  content:  {simpletool_decision.get('content')!r}\n")
            for i, a in enumerate(simpletool_decision.get("args", [])):
                if a:
                    f.write(f"  arg{i+1}: {a!r}\n")
    good(f"wrote {meta_path}")

    # 4. .vrma (call subprocess; pure C++ binary already validated)
    if make_vrma:
        if motion_to_vrma_bin is None:
            warn("motion_to_vrma binary not found; skipping .vrma "
                 "(run `cmake --build build` first)")
            return
        vrma_path = out_dir / "motion.vrma"
        cmd = [str(motion_to_vrma_bin),
               "--in",  str(bin_path),
               "--out", str(vrma_path),
               "--title", prompt[:60]]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            bad("motion_to_vrma timeout (>15s)")
            return
        if r.returncode != 0:
            bad(f"motion_to_vrma rc={r.returncode}")
            if r.stderr:
                for line in r.stderr.strip().splitlines():
                    dim(f"  {line}")
            return
        good(f"wrote {vrma_path}  "
             f"({vrma_path.stat().st_size} bytes)")
        # show 1-2 useful lines from motion_to_vrma's stdout
        for line in r.stdout.strip().splitlines():
            if line.startswith("[hips]") or line.startswith("[out]"):
                dim(line)


# ===========================================================================
# One-shot pipeline
# ===========================================================================

def run_one(query, args, motion_to_vrma_bin, out_dir):
    title(f"query: {query!r}")

    # Step 1: simpletool decision (optional)
    decision = None
    motion_prompt = query
    reason = "(simpletool skipped)"
    if not args.no_simpletool:
        try:
            decision = call_simpletool(query, args.environment, args.history,
                                       host=args.host, port=args.simpletool_port,
                                       verbose=args.verbose)
            fn = decision.get("function")
            content = decision.get("content")
            argstrs = [f"arg{i+1}={a!r}" for i, a in
                       enumerate(decision.get("args", [])) if a]
            info(f"{c_c}simpletool{c_off}: "
                 f"function={fn!r}  {' '.join(argstrs)}")
            if content:
                short = content if len(content) < 60 else content[:57] + "..."
                dim(f"  content: {short!r}")
            mp, reason = motion_prompt_from_decision(decision, query)
            if mp is None:
                warn(f"{reason} → no motion artifact will be produced")
                return
            motion_prompt = mp
            dim(f"  derived motion prompt: {motion_prompt!r}  ({reason})")
        except Exception as e:
            warn(f"simpletool failed ({type(e).__name__}: {e}); "
                 f"falling back to raw query")

    # Step 2: t2m
    info(f"{c_c}t2m{c_off}: generating motion...")
    try:
        motion, t = run_t2m(motion_prompt,
                            host=args.host, port=args.t2m_port,
                            duration=args.duration, seed=args.seed)
    except Exception as e:
        bad(f"t2m failed: {type(e).__name__}: {e}")
        return
    info(f"  motion shape={motion.shape}  "
         f"abs_max={np.abs(motion).max():.3f}")
    info(f"  timing: wall={t['client_wall_ms']:.1f}ms  "
         f"hidden={t['hidden_ms']:.1f}  "
         f"dit={t['dit_total_ms']:.1f}")

    # Step 3: artifacts
    info(f"{c_c}artifacts{c_off}: writing to {out_dir}")
    write_artifacts(motion, out_dir, motion_prompt, decision,
                    motion_to_vrma_bin=motion_to_vrma_bin,
                    make_vrma=not args.no_vrma)


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("query", nargs="?",
                    help="single query (omit for interactive REPL)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--simpletool-port", type=int, default=8422)
    ap.add_argument("--t2m-port",        type=int, default=8423)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--environment", default="[]")
    ap.add_argument("--history",     default="[]")
    ap.add_argument("--no-simpletool", action="store_true",
                    help="skip LLM step, feed raw query to t2m")
    ap.add_argument("--no-vrma", action="store_true",
                    help="don't run motion_to_vrma (only write .bin / .npy)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="dump simpletool raw heads")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default ./out/<timestamp>/)")
    ap.add_argument("--motion-to-vrma", default=None,
                    help="path to motion_to_vrma binary (auto-detected)")
    args = ap.parse_args()

    # find motion_to_vrma
    mtv = Path(args.motion_to_vrma) if args.motion_to_vrma else find_motion_to_vrma()
    if mtv:
        dim(f"motion_to_vrma: {mtv}")
    else:
        if not args.no_vrma:
            warn("motion_to_vrma not found in ./build/ or PATH; .vrma step will fail")

    # warn if simpletool module isn't importable
    if not args.no_simpletool and st is None:
        warn("simpletool_client.py not found in sys.path; "
             "auto-enabling --no-simpletool")
        args.no_simpletool = True

    # ---- one-shot ----
    if args.query:
        out_dir = args.out_dir or f"./out/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_one(args.query, args, mtv, out_dir)
        title(f"done → {out_dir}")
        return

    # ---- interactive REPL ----
    title("SimpleLove demo client (Ctrl-D or 'quit' to exit)")
    info(f"simpletool: {args.host}:{args.simpletool_port}  "
         f"{'(disabled)' if args.no_simpletool else ''}")
    info(f"t2m       : {args.host}:{args.t2m_port}")
    print()

    n = 0
    while True:
        try:
            q = input(f"{c_b}> {c_off}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        n += 1
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.out_dir or f"./out/{ts}_{n:03d}"
        run_one(q, args, mtv, out_dir)


if __name__ == "__main__":
    main()
