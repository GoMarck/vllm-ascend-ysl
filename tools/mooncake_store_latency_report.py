#!/usr/bin/env python3
import argparse
import math
import re
import sys
from collections import defaultdict


LOG_RE = re.compile(
    r"Store backend=(?P<backend>\w+) op=(?P<op>\S+) completed in "
    r"(?P<ms>[0-9.]+) ms \(keys=(?P<keys>\d+), buffers=(?P<buffers>\d+), "
    r"bytes=(?P<bytes>\d+)\)")


class StatBucket:
    def __init__(self) -> None:
        self.durations_ms: list[float] = []
        self.keys_total = 0
        self.buffers_total = 0
        self.bytes_total = 0

    def add(self, ms: float, keys: int, buffers: int, byte_count: int) -> None:
        self.durations_ms.append(ms)
        self.keys_total += keys
        self.buffers_total += buffers
        self.bytes_total += byte_count

    def merge(self, other: "StatBucket") -> None:
        self.durations_ms.extend(other.durations_ms)
        self.keys_total += other.keys_total
        self.buffers_total += other.buffers_total
        self.bytes_total += other.bytes_total


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    weight = idx - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def parse_lines(lines, buckets):
    matched = 0
    for line in lines:
        match = LOG_RE.search(line)
        if not match:
            continue
        backend = match.group("backend")
        op = match.group("op")
        ms = float(match.group("ms"))
        keys = int(match.group("keys"))
        buffers = int(match.group("buffers"))
        byte_count = int(match.group("bytes"))
        buckets[(backend, op)].add(ms, keys, buffers, byte_count)
        matched += 1
    return matched


def read_files(paths: list[str]) -> int:
    buckets: dict[tuple[str, str], StatBucket] = defaultdict(StatBucket)
    matched_total = 0
    if not paths:
        matched_total += parse_lines(sys.stdin, buckets)
    else:
        for path in paths:
            if path == "-":
                matched_total += parse_lines(sys.stdin, buckets)
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                matched_total += parse_lines(f, buckets)
    report(buckets, matched_total)
    return 0


def report(buckets: dict[tuple[str, str], StatBucket], matched_total: int) -> None:
    if matched_total == 0:
        print("No mooncake_store latency logs matched.", file=sys.stderr)
        return
    bytes_per_mb = 1024 * 1024
    rows = []
    for (backend, op), bucket in buckets.items():
        rows.append((backend, op, bucket))
    rows.sort(key=lambda x: (x[0], x[1]))

    header = [
        "backend",
        "op",
        "count",
        "mean_ms",
        "p99_ms",
        "mb_avg",
        "keys_avg",
    ]
    print("\t".join(header))
    for backend, op, bucket in rows:
        count = len(bucket.durations_ms)
        total_ms = sum(bucket.durations_ms)
        mean_ms = total_ms / count if count else float("nan")
        p99 = percentile(bucket.durations_ms, 0.99)
        bytes_avg = bucket.bytes_total / count if count else float("nan")
        mb_avg = bytes_avg / bytes_per_mb if count else float("nan")
        keys_avg = bucket.keys_total / count if count else float("nan")
        print("\t".join([
            backend,
            op,
            str(count),
            format_float(mean_ms),
            format_float(p99),
            format_float(mb_avg),
            format_float(keys_avg),
        ]))

    summary = defaultdict(StatBucket)
    for (backend, _op), bucket in buckets.items():
        summary[backend].merge(bucket)
    print("\nbackend summary")
    print("\t".join([
        "backend",
        "count",
        "mean_ms",
        "p99_ms",
        "mb_avg",
        "keys_avg",
    ]))
    for backend, bucket in sorted(summary.items()):
        count = len(bucket.durations_ms)
        total_ms = sum(bucket.durations_ms)
        mean_ms = total_ms / count if count else float("nan")
        p99 = percentile(bucket.durations_ms, 0.99)
        bytes_avg = bucket.bytes_total / count if count else float("nan")
        mb_avg = bytes_avg / bytes_per_mb if count else float("nan")
        keys_avg = bucket.keys_total / count if count else float("nan")
        print("\t".join([
            backend,
            str(count),
            format_float(mean_ms),
            format_float(p99),
            format_float(mb_avg),
            format_float(keys_avg),
        ]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Parse Mooncake store latency logs and summarize per "
                     "backend/op."))
    parser.add_argument(
        "paths",
        nargs="*",
        help=("Log file paths. If empty or '-', reads from stdin."),
    )
    args = parser.parse_args()
    return read_files(args.paths)


if __name__ == "__main__":
    raise SystemExit(main())
