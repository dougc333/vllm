#!/usr/bin/env python3
"""Graph a normalized CSV produced by benchmark_random.py."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("/content/results_a100/summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/content/results_a100/qwen-benchmark.png"),
    )
    args = parser.parse_args()

    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("summary CSV has no benchmark rows")

    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        key = (
            row["model"],
            int(float(row["target_input_tokens"])),
        )
        groups[key].append(row)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    panels = [
        ("request_throughput_rps", "Achieved requests/s"),
        ("p95_ttft_ms", "p95 TTFT (ms)"),
        ("p95_itl_ms", "p95 inter-token latency (ms)"),
        ("output_throughput_tps", "Output tokens/s"),
        ("median_input_tokens", "Observed median input tokens"),
        ("p95_e2el_ms", "p95 end-to-end latency (ms)"),
    ]

    for ax, (field, title) in zip(axes.flat, panels):
        for (model, input_len), group in sorted(groups.items()):
            ordered = sorted(
                group,
                key=lambda row: number(row, "offered_rps"),
            )

            x = [number(row, "offered_rps") for row in ordered]
            y = [number(row, field) for row in ordered]

            ax.plot(
                x,
                y,
                marker="o",
                label=f"{model.split('/')[-1]} · {input_len} input tokens",
            )

        ax.set_title(title)
        ax.set_xlabel("Offered requests/s")
        ax.grid(alpha=0.25)

    # Reference lines
    axes[0, 0].plot(
        [0, 10],
        [0, 10],
        linestyle="--",
        color="grey",
        label="offered = achieved",
    )
    axes[0, 1].axhline(
        800,
        linestyle="--",
        color="red",
        label="800 ms TTFT target",
    )
    axes[1, 1].axhline(3000, linestyle=":", color="grey")
    axes[1, 1].axhline(8000, linestyle=":", color="grey")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    fig.suptitle("Qwen vLLM random-dataset benchmark", fontsize=16)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
