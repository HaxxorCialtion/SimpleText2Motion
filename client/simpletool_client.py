#!/usr/bin/env python3
"""
SimpleTool multi-head decode tester (no t2m / no vrma dependencies).

Mirrors the production prompt format used during training:
  cold = <|im_start|>system\n {SimpleTool system prompt + tool list} <|im_end|>\n
  hot  = <|im_start|>user\n environment: ... \n history: ... \n\n {query} <|im_end|>\n
         <|im_start|>assistant\n

The model is expected to emit, per head:
  head 0 (content) : "<content>response text</content>"
  head 1 (function): "<function>tool_name</function>" or "<function><|null|></function>"
  head 2..7 (argN) : "<argN>value</argN>" or "<argN><|null|></argN>"

Usage:
  python client/simpletool_client.py                  # run 4 default scenarios
  python test_simpletool_v2.py --query "..."    # one custom query
  python test_simpletool_v2.py --cache-bench    # measure HIT vs RESET savings
  python test_simpletool_v2.py --heads function,arg1   # head_mask filter
"""

import argparse
import socket
import struct
import sys
from dataclasses import dataclass, field
from typing import List, Optional


# ===========================================================================
# Wire constants
# ===========================================================================

SIMPLETOOL_HOST = "127.0.0.1"
SIMPLETOOL_PORT = 8422

HEAD_NAMES = ["content", "function", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6"]
COLD_ACTION_NAMES = {0: "HIT", 1: "EXTEND", 2: "RESET"}


# ===========================================================================
# Tool registry  (must match training format)
# ===========================================================================

@dataclass
class Tool:
    name: str
    description: str
    arg_hints: List[str] = field(default_factory=list)
    n_required: int = 0

    @property
    def n_args(self) -> int:
        return len(self.arg_hints)


DEFAULT_TOOLS: List[Tool] = [
    Tool(
        name="move_to",
        description="Move the character to a specified location in the scene.",
        arg_hints=["target location (e.g. 'the chair', 'the window')"],
        n_required=1,
    ),
    Tool(
        name="action",
        description="Perform a physical action described by verb + optional modifiers.",
        arg_hints=[
            "verb or verb phrase (e.g. 'pick', 'dance', 'sit')",
            "adjective / manner (e.g. 'slowly', 'joyfully'; optional)",
            "object or target noun (e.g. 'apple', 'the camera'; optional)",
            "adverb or prepositional phrase (e.g. 'from the ground', 'to the left'; optional)",
        ],
        n_required=1,
    ),
    Tool(
        name="pass",
        description="Stay silent / do nothing. Use when no physical action is needed.",
        arg_hints=[],
        n_required=0,
    ),
]


DEFAULT_SYSTEM_PROMPT = (
    "You are a multi-head parallel function calling model. \n"
    "## Output Heads\n"
    "\n"
    "**Head 0 - <content>**: Natural language response\n"
    "- Format: <content>response text</content>\n"
    "- Answer what you want to say while you are calling a function\n"
    "\n"
    "**Head 1 - <function>**: Function names to call\n"
    "- Format: <function>name</function>\n"
    "- Name: must match tool defined name\n"
    "\n"
    "**Head 2-7 - <arg1>、<arg2>、<arg3>、<arg4>、<arg5>、<arg6>**: Function arguments by position\n"
    "- Format: <argN>value</argN> \n"
    "- Strictly fill in according to the parameter order of the tool you intend to call\n"
    "- Note the special restrictions of parameter definitions for corresponding positions\n"
    "- If the corresponding tool definition has required parameters, these must be filled in\n"
    "- Infer the user's actual needs.\n"
    "- If Unnecessary: <argN><|null|></argN>\n"
    "\n"
    "**Environment - The information you have.\n"
    "**History - The tools you have called.\n"
)


# ===========================================================================
# Prompt rendering  (must match training format byte-for-byte)
# ===========================================================================

def _build_tools_json(tools: List[Tool]) -> str:
    lines = []
    for t in tools:
        props = []
        for a in range(t.n_args):
            hint = t.arg_hints[a] if a < len(t.arg_hints) else ""
            props.append(
                f'"arg{a+1}": {{"type": "string", "description": "{hint}"}}'
            )
        params = '{"type": "object", "properties": {' + ", ".join(props) + "}"
        if t.n_required > 0:
            req = ", ".join(f'"arg{a+1}"' for a in range(t.n_required))
            params += f', "required": [{req}]'
        params += "}"
        lines.append(
            '{"type": "function", "function": '
            f'{{"name": "{t.name}", "description": "{t.description}", '
            f'"parameters": {params}}}}}'
        )
    return "\n".join(lines)


def render_cold(system_prompt: str, tools: List[Tool]) -> str:
    """Cold prefix: stable across requests with the same tool config.
    The server hashes these exact bytes for cold-cache hit detection."""
    return (
        "<|im_start|>system\n"
        + system_prompt
        + "\n## Available Tools:\n\n"
        + _build_tools_json(tools)
        + "<|im_end|>\n"
    )


def render_hot(environment: str, history: str, query: str) -> str:
    """Hot suffix: per-request user turn + assistant marker."""
    if not environment:
        environment = "[]"
    if not history:
        history = "[]"
    return (
        "<|im_start|>user\n"
        f"environment: {environment}\n"
        f"history: {history}\n"
        f"\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ===========================================================================
# Wire I/O
# ===========================================================================

def _recv_n(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"server closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def simpletool_request(
    cold: str,
    hot: str,
    head_mask: int = 0xFF,
    max_gen_tokens: int = 0,
    host: str = SIMPLETOOL_HOST,
    port: int = SIMPLETOOL_PORT,
) -> dict:
    """Send one simpletool request and parse the response.

    Args:
        cold: rendered cold prefix (utf-8). Stable across requests.
        hot:  rendered hot suffix (utf-8). Per-request.
        head_mask: bit i = include head i. 0xFF = all 8 heads.
                   Examples: 0x02 = function only, 0x03 = content+function.
        max_gen_tokens: per-head decode cap (1..255). 0 = server default (64).
    """
    cold_b = cold.encode("utf-8")
    hot_b  = hot.encode("utf-8")

    with socket.create_connection((host, port)) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # ---- request ----
        s.sendall(struct.pack(">I", len(cold_b)) + cold_b)
        s.sendall(struct.pack(">I", len(hot_b))  + hot_b)
        s.sendall(struct.pack("BB", head_mask & 0xFF, max_gen_tokens & 0xFF))

        # ---- response: status ----
        (status,) = struct.unpack(">i", _recv_n(s, 4))
        if status != 0:
            (err_len,) = struct.unpack(">H", _recv_n(s, 2))
            err_msg = _recv_n(s, err_len).decode("utf-8", errors="replace") if err_len else ""
            raise RuntimeError(f"simpletool error status={status}: {err_msg}")

        # ---- response: per-head ----
        (n_heads,) = struct.unpack("B", _recv_n(s, 1))
        heads = []
        for _ in range(n_heads):
            head_idx, is_null = struct.unpack("BB", _recv_n(s, 2))
            (text_len,) = struct.unpack(">H", _recv_n(s, 2))
            text = _recv_n(s, text_len).decode("utf-8", errors="replace") if text_len else ""
            heads.append({
                "head_idx": head_idx,
                "name":     HEAD_NAMES[head_idx] if head_idx < len(HEAD_NAMES) else f"head{head_idx}",
                "is_null":  bool(is_null),
                "text":     text,
            })

        # ---- response: cache stats + timings ----
        n_cold_tokens, n_hot_tokens = struct.unpack(">II", _recv_n(s, 8))
        (cold_action,) = struct.unpack("B", _recv_n(s, 1))
        timings = struct.unpack("<5d", _recv_n(s, 40))

    return {
        "heads":             heads,
        "n_cold_tokens":     n_cold_tokens,
        "n_hot_tokens":      n_hot_tokens,
        "cold_action":       cold_action,
        "cold_action_str":   COLD_ACTION_NAMES.get(cold_action, f"?{cold_action}"),
        "dt_cold_ms":        timings[0],
        "dt_hot_prefill_ms": timings[1],
        "dt_signal_ms":      timings[2],
        "dt_decode_ms":      timings[3],
        "dt_total_ms":       timings[4],
    }


def parse_heads(heads: List[dict]) -> dict:
    """Turn flat head list into structured {function, content, args[0..5]}.
    Null heads -> None in args; missing heads -> None."""
    out = {"function": "", "content": "", "args": [None] * 6}
    for h in heads:
        idx = h["head_idx"]
        if 0 <= idx < len(HEAD_NAMES):
            name = HEAD_NAMES[idx]
            if name == "content":
                out["content"] = h["text"]
            elif name == "function":
                out["function"] = None if h["is_null"] else h["text"]
            elif name.startswith("arg"):
                ai = int(name[3:]) - 1
                out["args"][ai] = None if h["is_null"] else h["text"]
    return out


# ===========================================================================
# Pretty printing
# ===========================================================================

def print_response(resp: dict, label: str = "") -> None:
    if label:
        print(f"\n--- {label} ---")
    parsed = parse_heads(resp["heads"])

    # Function call summary
    fn = parsed["function"]
    if fn:
        args_str = []
        for i, a in enumerate(parsed["args"]):
            if a is not None:
                args_str.append(f"arg{i+1}={a!r}")
        call = f"{fn}({', '.join(args_str)})" if args_str else f"{fn}()"
        print(f"  call:    {call}")
    else:
        print(f"  call:    <null>  (no function)")

    if parsed["content"]:
        c = parsed["content"]
        if len(c) > 80: c = c[:77] + "..."
        print(f"  content: {c!r}")

    print(f"  cache:   {resp['cold_action_str']:<6s} "
          f"n_cold={resp['n_cold_tokens']}  n_hot={resp['n_hot_tokens']}")
    print(f"  timing:  total={resp['dt_total_ms']:6.1f}ms  "
          f"(cold={resp['dt_cold_ms']:5.1f}  "
          f"hot={resp['dt_hot_prefill_ms']:5.1f}  "
          f"sig={resp['dt_signal_ms']:5.1f}  "
          f"dec={resp['dt_decode_ms']:5.1f})")


def print_raw_heads(resp: dict) -> None:
    """For debugging: show every head's raw text + null flag."""
    print(f"  raw heads ({len(resp['heads'])}):")
    for h in resp["heads"]:
        marker = "[NULL]" if h["is_null"] else "      "
        t = h["text"].replace("\n", "\\n")
        if len(t) > 60: t = t[:57] + "..."
        print(f"    {marker}  {h['name']:9s} : {t!r}")


# ===========================================================================
# Scenarios
# ===========================================================================

DEFAULT_SCENARIOS = [
    {
        "label": "move_to: walk to chair",
        "query": "Walk over to the chair please.",
        "environment": "['chair', 'desk', 'window']",
    },
    {
        "label": "action: dance joyfully",
        "query": "Dance joyfully for me!",
        "environment": "[]",
    },
    {
        "label": "action: pick apple slowly",
        "query": "Pick the apple from the ground, slowly.",
        "environment": "['apple', 'tree']",
    },
    {
        "label": "pass: no action needed",
        "query": "Hmm, I don't really need you to move right now.",
        "environment": "[]",
    },
]


# ===========================================================================
# Modes
# ===========================================================================

def mode_scenarios(args, cold: str):
    """Run the default scenarios sequentially. First call should be RESET,
    rest should be HIT on cold cache."""
    for i, sc in enumerate(DEFAULT_SCENARIOS):
        hot = render_hot(sc["environment"], "[]", sc["query"])
        print(f"\n[{i+1}/{len(DEFAULT_SCENARIOS)}]  {sc['label']}")
        print(f"    query: {sc['query']!r}")
        print(f"    env:   {sc['environment']}")
        try:
            resp = simpletool_request(
                cold, hot,
                head_mask=args.head_mask,
                max_gen_tokens=args.max_gen,
            )
            print_response(resp)
            if args.verbose:
                print_raw_heads(resp)
        except Exception as e:
            print(f"    X failed: {type(e).__name__}: {e}")


def mode_single(args, cold: str):
    hot = render_hot(args.environment, args.history, args.query)
    print(f"\nquery: {args.query!r}")
    print(f"env:   {args.environment}")
    print(f"hist:  {args.history}")
    resp = simpletool_request(
        cold, hot,
        head_mask=args.head_mask,
        max_gen_tokens=args.max_gen,
    )
    print_response(resp)
    print_raw_heads(resp)


def mode_cache_bench(args, cold: str):
    """Demonstrate the cold-cache speedup. First call resets, subsequent
    calls with same cold should be HIT and skip cold prefill entirely."""
    print("\nCache benchmark: same cold across N calls.")
    print("Expect call #1 = RESET (slow), calls #2.. = HIT (fast).\n")

    hot = render_hot("[]", "[]", "Walk to the chair.")
    times = []
    for i in range(args.cache_bench_n):
        resp = simpletool_request(cold, hot, head_mask=args.head_mask,
                                  max_gen_tokens=args.max_gen)
        action = resp["cold_action_str"]
        print(f"  #{i+1:2d}  {action:<6s}  "
              f"cold={resp['dt_cold_ms']:6.1f}ms  "
              f"hot={resp['dt_hot_prefill_ms']:5.1f}ms  "
              f"sig={resp['dt_signal_ms']:5.1f}ms  "
              f"dec={resp['dt_decode_ms']:5.1f}ms  "
              f"|  total={resp['dt_total_ms']:6.1f}ms")
        times.append((action, resp["dt_total_ms"]))

    # Quick stats: first vs median of rest
    if len(times) >= 2:
        first_total = times[0][1]
        rest = sorted([t for _, t in times[1:]])
        median_rest = rest[len(rest) // 2]
        print(f"\n  first call : {first_total:.1f}ms")
        print(f"  median rest: {median_rest:.1f}ms")
        print(f"  speedup    : {first_total / max(median_rest, 0.01):.1f}x")


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=SIMPLETOOL_HOST)
    ap.add_argument("--port", type=int, default=SIMPLETOOL_PORT)
    ap.add_argument("--max-gen", type=int, default=0,
                    help="per-head max gen tokens (0 = server default 64)")
    ap.add_argument("--heads", default="content,function,arg1,arg2,arg3,arg4,arg5,arg6",
                    help="comma-separated head names")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="also print raw per-head output")

    # Mode selectors (mutually exclusive)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--query", help="run a single custom query")
    mode.add_argument("--cache-bench", action="store_true",
                      help="benchmark cold-cache HIT vs RESET timing")
    # default mode (no flag): run scenarios

    ap.add_argument("--environment", default="[]",
                    help="environment for --query mode")
    ap.add_argument("--history", default="[]",
                    help="history for --query mode")
    ap.add_argument("--cache-bench-n", type=int, default=5,
                    help="number of calls in --cache-bench mode")
    args = ap.parse_args()

    # Build head_mask
    selected = [h.strip() for h in args.heads.split(",") if h.strip()]
    mask = 0
    for h in selected:
        if h not in HEAD_NAMES:
            print(f"unknown head: {h!r}  (valid: {HEAD_NAMES})", file=sys.stderr)
            sys.exit(1)
        mask |= 1 << HEAD_NAMES.index(h)
    args.head_mask = mask

    # Build the (shared) cold prefix
    cold = render_cold(DEFAULT_SYSTEM_PROMPT, DEFAULT_TOOLS)

    print(f"server     : {args.host}:{args.port}")
    print(f"head_mask  : 0x{mask:02x}  heads={selected}")
    print(f"max_gen    : {args.max_gen} ({'server default' if args.max_gen == 0 else 'custom'})")
    print(f"cold size  : {len(cold)} bytes ({len(cold.encode('utf-8'))} utf-8 bytes)")

    if args.cache_bench:
        mode_cache_bench(args, cold)
    elif args.query:
        mode_single(args, cold)
    else:
        mode_scenarios(args, cold)


if __name__ == "__main__":
    main()