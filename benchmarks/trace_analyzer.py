"""Trace analyzer: merge all events from one or two traces onto a single timeline.

Usage:
    # Single trace - full timeline
    python3 trace_analyzer.py trace.json

    # Two traces - unified timeline
    python3 trace_analyzer.py traceA.json --compare traceB.json --label-a V1+CG --label-b Bypass

Options:
    --compare FILE    Second trace to merge onto the same timeline
    --label-a NAME    Label for first trace (default: A)
    --label-b NAME    Label for second trace (default: B)
    --filter PATTERN  Regex filter on event name
    --cuda-only       Show only CUDA kernel events
    --cpu-only        Show only CPU events
    --start-ms MS     Start offset in ms
    --end-ms MS       End offset in ms
    --top N           Also print kernel summary (top N)
    --out FILE        Write timeline to file instead of stdout
    --csv FILE        Write timeline as CSV
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict


def load_trace(path):
    print(f"Loading {path}...", end=" ", flush=True)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        events = data.get("traceEvents", [])
    elif isinstance(data, list):
        events = data
    else:
        events = []
    print(f"{len(events)} raw events")
    return events


def classify_event(ev):
    cat = ev.get("cat", "")
    if "kernel" in cat.lower():
        return "cuda_kernel"
    if "cuda_runtime" in cat.lower() or "runtime" in cat.lower():
        return "cuda_runtime"
    if cat in ("cpu_op", "user_annotation", "python_function", "Trace"):
        return "cpu"
    if "gpu" in cat.lower() or "cuda" in cat.lower():
        return "cuda_kernel"
    return "cpu"


def parse_events(raw_events, source_label, cuda_only=False, cpu_only=False, pattern=None):
    parsed = []
    for ev in raw_events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        tid = ev.get("tid", 0)
        pid = ev.get("pid", 0)
        cat = ev.get("cat", "")
        kind = classify_event(ev)

        if cuda_only and kind != "cuda_kernel":
            continue
        if cpu_only and kind != "cpu":
            continue
        if pattern and not re.search(pattern, name, re.IGNORECASE):
            continue

        parsed.append({
            "name": name,
            "ts_us": ts,
            "dur_us": dur,
            "end_us": ts + dur,
            "tid": tid,
            "pid": pid,
            "cat": cat,
            "kind": kind,
            "source": source_label,
        })
    return parsed


def print_kernel_summary(events, label, top_n=20):
    by_name = defaultdict(lambda: {"total_us": 0, "count": 0, "min_us": float("inf"), "max_us": 0})
    total_cuda = 0
    for ev in events:
        if ev["kind"] != "cuda_kernel":
            continue
        s = by_name[ev["name"]]
        s["total_us"] += ev["dur_us"]
        s["count"] += 1
        s["min_us"] = min(s["min_us"], ev["dur_us"])
        s["max_us"] = max(s["max_us"], ev["dur_us"])
        total_cuda += ev["dur_us"]

    ranked = sorted(by_name.items(), key=lambda x: -x[1]["total_us"])

    print(f"\n{'='*120}")
    print(f"[{label}] CUDA Kernel Summary (top {top_n}, total CUDA: {total_cuda/1e6:.3f}s)")
    print(f"{'='*120}")
    print(f"{'#':>3} {'Kernel':<60} {'Total':>10} {'%':>6} {'Calls':>7} {'Avg':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*3} {'-'*60} {'-'*10} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")

    for i, (name, s) in enumerate(ranked[:top_n]):
        avg = s["total_us"] / s["count"]
        pct = s["total_us"] / total_cuda * 100 if total_cuda > 0 else 0
        short = name[:60] if len(name) <= 60 else name[:57] + "..."
        print(f"{i+1:>3} {short:<60} {s['total_us']/1e3:>9.1f}ms {pct:>5.1f}% {s['count']:>7} {avg:>9.1f}us {s['min_us']:>9.1f}us {s['max_us']:>9.1f}us")


def write_timeline(events, out, fmt="text"):
    """Write all events sorted by start time."""
    events.sort(key=lambda e: (e["ts_us"], -e["dur_us"]))

    if not events:
        print("No events.", file=out)
        return

    t0 = events[0]["ts_us"]
    total = len(events)

    if fmt == "text":
        print(f"Total events: {total}", file=out)
        print(f"Time span: {(events[-1]['end_us'] - t0)/1e6:.3f}s", file=out)
        print(file=out)
        print(f"{'#':>7} {'Source':<8} {'Start':>12} {'Duration':>12} {'End':>12} {'Kind':<12} {'TID':>6} {'Name'}", file=out)
        print(f"{'-'*7} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*6} {'-'*80}", file=out)

        for i, ev in enumerate(events):
            rel_start = (ev["ts_us"] - t0) / 1000
            dur_ms = ev["dur_us"] / 1000
            rel_end = rel_start + dur_ms
            name = ev["name"][:100] if len(ev["name"]) <= 100 else ev["name"][:97] + "..."
            print(f"{i+1:>7} {ev['source']:<8} {rel_start:>10.3f}ms {dur_ms:>10.3f}ms {rel_end:>10.3f}ms {ev['kind']:<12} {ev['tid']:>6} {name}", file=out)

    elif fmt == "csv":
        writer = csv.writer(out)
        writer.writerow(["idx", "source", "start_ms", "dur_ms", "end_ms", "kind", "tid", "pid", "cat", "name"])
        for i, ev in enumerate(events):
            rel_start = (ev["ts_us"] - t0) / 1000
            dur_ms = ev["dur_us"] / 1000
            rel_end = rel_start + dur_ms
            writer.writerow([i+1, ev["source"], f"{rel_start:.3f}", f"{dur_ms:.3f}", f"{rel_end:.3f}",
                             ev["kind"], ev["tid"], ev["pid"], ev["cat"], ev["name"]])


def main():
    parser = argparse.ArgumentParser(description="Trace timeline analyzer")
    parser.add_argument("trace", help="Chrome trace JSON")
    parser.add_argument("--compare", type=str, default=None, help="Second trace to merge")
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--filter", type=str, default=None, help="Regex filter")
    parser.add_argument("--cuda-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--start-ms", type=float, default=None)
    parser.add_argument("--end-ms", type=float, default=None)
    parser.add_argument("--top", type=int, default=0, help="Print kernel summary (top N)")
    parser.add_argument("--out", type=str, default=None, help="Write timeline to file")
    parser.add_argument("--csv", type=str, default=None, help="Write timeline as CSV")
    args = parser.parse_args()

    # Load and parse trace A
    raw_a = load_trace(args.trace)
    events_a = parse_events(raw_a, args.label_a, cuda_only=args.cuda_only,
                            cpu_only=args.cpu_only, pattern=args.filter)
    print(f"  [{args.label_a}] {len(events_a)} events after filtering")

    # Normalize timestamps: both start at 0
    if events_a:
        t0_a = min(e["ts_us"] for e in events_a)
        for e in events_a:
            e["ts_us"] -= t0_a
            e["end_us"] = e["ts_us"] + e["dur_us"]

    all_events = list(events_a)

    # Load and parse trace B if comparing
    if args.compare:
        raw_b = load_trace(args.compare)
        events_b = parse_events(raw_b, args.label_b, cuda_only=args.cuda_only,
                                cpu_only=args.cpu_only, pattern=args.filter)
        print(f"  [{args.label_b}] {len(events_b)} events after filtering")

        if events_b:
            t0_b = min(e["ts_us"] for e in events_b)
            for e in events_b:
                e["ts_us"] -= t0_b
                e["end_us"] = e["ts_us"] + e["dur_us"]

        all_events.extend(events_b)

    # Time window filter
    if args.start_ms is not None:
        all_events = [e for e in all_events if e["end_us"] >= args.start_ms * 1000]
    if args.end_ms is not None:
        all_events = [e for e in all_events if e["ts_us"] <= args.end_ms * 1000]

    all_events.sort(key=lambda e: (e["ts_us"], -e["dur_us"]))

    print(f"\nTotal events on timeline: {len(all_events)}")

    # Kernel summary
    if args.top > 0:
        if args.compare:
            print_kernel_summary(events_a, args.label_a, args.top)
            print_kernel_summary(events_b, args.label_b, args.top)
        else:
            print_kernel_summary(all_events, args.label_a, args.top)

    # Write timeline
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            write_timeline(all_events, f, fmt="csv")
        print(f"\nCSV written to {args.csv}")
    elif args.out:
        with open(args.out, "w") as f:
            write_timeline(all_events, f, fmt="text")
        print(f"\nTimeline written to {args.out}")
    else:
        write_timeline(all_events, sys.stdout, fmt="text")


if __name__ == "__main__":
    main()
