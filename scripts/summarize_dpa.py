#!/usr/bin/env python3
"""Recompute DPA v1/v2 kernel statistics from archived JSONL (Python 3.10+)."""

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_records(path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON: {error.msg}"
                ) from error


def shape_key(record):
    shape = record["shape"]
    return shape["visible_tokens"], shape["block_size"]


def path_key(record):
    return (*shape_key(record), record["path"], record.get("num_splits", 0))


def percentile(values, fraction):
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    weight = index - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def path_statistics(per_repeat):
    pooled = [value for values in per_repeat for value in values]
    median = statistics.median(pooled)
    mean = statistics.mean(pooled)
    cv = statistics.stdev(pooled) / mean * 100 if len(pooled) > 1 else 0.0
    repeat_medians = [statistics.median(values) for values in per_repeat]
    spread = max(abs(value - median) / median * 100 for value in repeat_medians)
    return {
        "n": len(pooled),
        "median_ms": median,
        "p10_ms": percentile(pooled, 0.1),
        "p90_ms": percentile(pooled, 0.9),
        "mean_ms": mean,
        "cv_percent": cv,
        "repeat_medians_ms": repeat_medians,
        "repeat_median_spread_percent": spread,
        "converged": cv <= 10 and spread <= 10,
    }


def summarize(records):
    records = list(records)
    counts = Counter(record["type"] for record in records)
    if counts["provenance"] != 1:
        raise ValueError("expected exactly one provenance record")
    provenance = next(record for record in records if record["type"] == "provenance")
    if provenance["schema"] not in {
        "tllm-dpa-kernel-bench-v1",
        "tllm-dpa-kernel-bench-v2",
    }:
        raise ValueError(f"unsupported raw schema: {provenance['schema']}")
    allowed = {
        "provenance",
        "equivalence",
        "equivalence_splitkv",
        "sample",
        "path_stats",
        "shape_summary",
    }
    if set(counts) - allowed:
        raise ValueError(f"unknown record types: {set(counts) - allowed}")

    shapes = provenance["shapes"]
    shape_keys = {(shape["visible_tokens"], shape["block_size"]) for shape in shapes}
    if not shapes or len(shape_keys) != len(shapes):
        raise ValueError("provenance shapes must be nonempty and unique")
    repeats, batch, reps = (provenance[field] for field in ("repeats", "batch", "reps"))
    if min(repeats, batch, reps) <= 0:
        raise ValueError("repeats, batch and reps must be positive")
    rounds = max(1, reps // batch)  # Matches src/kernel_bench.cpp.
    specs = [
        (name, 0) for name in ("legacy", "contiguous", "direct", "gather_k", "gather_v")
    ]
    sweep = provenance.get("num_splits_sweep", [])
    if provenance["schema"].endswith("v2") and not sweep:
        raise ValueError("v2 provenance must declare num_splits_sweep")
    if len(set(sweep)) != len(sweep) or any(value < 1 or value > 32 for value in sweep):
        raise ValueError("invalid num_splits_sweep")
    specs += [
        (f"{base}_splitkv", splits)
        for splits in sweep
        for base in ("contiguous", "direct", "legacy")
    ]
    expected = {
        (*shape, path, splits) for shape in shape_keys for path, splits in specs
    }
    expected_split = {key for key in expected if key[-1] > 0}
    equivalence, split_equivalence, logged_stats, logged_shapes = {}, {}, {}, {}
    samples = defaultdict(lambda: defaultdict(list))

    for record in records:
        kind = record["type"]
        if kind == "provenance":
            continue
        if shape_key(record) not in shape_keys:
            raise ValueError(f"undeclared shape: {shape_key(record)}")
        if kind == "sample":
            key = path_key(record)
            if key not in expected or record["repeat"] not in range(repeats):
                raise ValueError(f"unexpected sample path/repeat: {key}")
            if not math.isfinite(record["ms"]) or record["ms"] <= 0:
                raise ValueError(f"sample latency must be finite and positive: {key}")
            samples[key][record["repeat"]].append(record["ms"])
            continue
        if kind in {"equivalence", "shape_summary"}:
            target = equivalence if kind == "equivalence" else logged_shapes
            key = shape_key(record)
        else:
            target = (
                split_equivalence if kind == "equivalence_splitkv" else logged_stats
            )
            key = path_key(record)
        if key in target:
            raise ValueError(f"duplicate {kind}: {key}")
        target[key] = record

    if set(equivalence) != shape_keys or set(logged_shapes) != shape_keys:
        raise ValueError("missing equivalence or completed shape_summary records")
    if set(split_equivalence) != expected_split:
        raise ValueError("split-KV correctness coverage does not match declared sweep")
    if set(samples) != expected or set(logged_stats) != expected:
        raise ValueError("sample/path_stats coverage does not match declared paths")
    for key, record in equivalence.items():
        if not (
            record["legacy_vs_direct_bitwise_equal"]
            and record["legacy_vs_contiguous_bitwise_equal"]
        ):
            raise ValueError(f"single-pass correctness failed: {key}")
        completed = logged_shapes[key]
        if not completed["equiv_bitwise"] or not completed.get("equiv_ok", True):
            raise ValueError(f"completed shape reports correctness failure: {key}")
    for key, record in split_equivalence.items():
        diff = record["max_abs_diff"]
        valid = record["bitwise_equal"] if key[-1] == 1 else 0 <= diff <= 2e-3
        if not math.isfinite(diff) or not record["within_tolerance"] or not valid:
            raise ValueError(f"split-KV correctness failed: {key}")

    computed = {}
    for key in sorted(expected):
        per_repeat = samples[key]
        if set(per_repeat) != set(range(repeats)) or any(
            len(values) != rounds for values in per_repeat.values()
        ):
            raise ValueError(f"incomplete sample count per repeat: {key}")
        stats = path_statistics([per_repeat[repeat] for repeat in range(repeats)])
        logged = logged_stats[key]
        if logged["n"] != stats["n"]:
            raise ValueError(f"logged sample count disagrees: {key}")
        # raw samples are rounded to 6 decimals; CV is printed to 3 decimals.
        for field in ("median_ms", "p10_ms", "p90_ms", "mean_ms", "cv_percent"):
            tolerance = 0.05 if field == "cv_percent" else 1.1e-6
            if not math.isclose(
                logged[field], stats[field], rel_tol=0, abs_tol=tolerance
            ):
                raise ValueError(f"logged {field} disagrees with samples: {key}")
        if len(logged["repeat_medians_ms"]) != repeats or any(
            not math.isclose(actual, recorded, rel_tol=0, abs_tol=1.1e-6)
            for actual, recorded in zip(
                stats["repeat_medians_ms"], logged["repeat_medians_ms"]
            )
        ):
            raise ValueError(f"logged repeat_medians_ms disagrees with samples: {key}")
        computed[key] = stats

    result_shapes = []
    for shape in shapes:
        key = (shape["visible_tokens"], shape["block_size"])
        paths = [
            {"path": path, "num_splits": splits, **computed[(*key, path, splits)]}
            for path, splits in specs
        ]
        converged = all(
            path["converged"]
            for path in paths
            if not path["path"].startswith("gather_")
        )
        medians = {
            (path["path"], path["num_splits"]): path["median_ms"] for path in paths
        }
        ratios = {
            "direct_vs_legacy": medians["legacy", 0] / medians["direct", 0]
            if converged
            else None,
            "direct_vs_contiguous": medians["contiguous", 0] / medians["direct", 0]
            if converged
            else None,
        }
        if sweep:
            ratios["splitkv_vs_direct"] = {
                str(splits): medians["direct", 0] / medians["direct_splitkv", splits]
                if converged
                else None
                for splits in sweep
            }
            ratios["direct_splitkv_vs_legacy_splitkv"] = {
                str(splits): medians["legacy_splitkv", splits]
                / medians["direct_splitkv", splits]
                if converged
                else None
                for splits in sweep
            }
        result_shapes.append(
            {
                "shape": shape,
                "paths": paths,
                "converged": converged,
                "recorded_converged": logged_shapes[key]["converged"],
                "gathers_converged": all(
                    path["converged"]
                    for path in paths
                    if path["path"].startswith("gather_")
                ),
                "speedup": ratios,
            }
        )
    return {
        "schema": "tllm-dpa-recomputed-summary-v1",
        "provenance": provenance,
        "record_counts": dict(sorted(counts.items())),
        "equivalence": {
            "single_pass_shapes": len(equivalence),
            "splitkv_records": len(split_equivalence),
            "failures": 0,
        },
        "convergence": {
            "total_shapes": len(shapes),
            "converged_shapes": sum(shape["converged"] for shape in result_shapes),
        },
        "limitations": [
            "Kernel-only; no TTFT/TPOT/Serving claim; no new GPU experiment.",
            "Statistics recomputed from six-decimal batch-mean samples, not individual-call samples.",
            "Speedup is null when comparison paths do not all converge (CV/spread <= 10%).",
            "Historical summary JSON and profiler interpretations are not regenerated by this tool.",
        ],
        "shapes": result_shapes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="archived DPA v1/v2 JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        help="new JSON file; existing files are never overwritten",
    )
    args = parser.parse_args()
    try:
        summary = summarize(load_records(args.raw))
        summary["source_sha256"] = hashlib.sha256(args.raw.read_bytes()).hexdigest()
        payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        else:
            sys.stdout.write(payload)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
