#!/usr/bin/env python3
"""
Generate charts from gpu_profiler telemetry CSVs.
Usage: python scripts/plots.py data/run_*.csv
Outputs PNG files alongside each CSV.
"""
import sys
import pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


METRICS = [
    ("power_w",         "Power (W)",       "tab:red"),
    ("temp_c",          "Temp (°C)",        "tab:orange"),
    ("clock_sm_mhz",    "SM Clock (MHz)",   "tab:blue"),
    ("clock_mem_mhz",   "Mem Clock (MHz)",  "tab:cyan"),
    ("utilization_gpu", "SM Util (%)",      "tab:green"),
    ("utilization_mem", "Mem Ctrl Util (%)", "tab:purple"),
    ("used_mem_gb",     "VRAM Used (GB)",   "tab:brown"),
]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_s"] = df["timestamp_ms"] / 1000.0
    df["power_w"]     = df["power_mw"]    / 1000.0
    df["used_mem_gb"] = df["used_memory_bytes"] / 1024**3
    return df


def plot_overview(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    n   = len(METRICS)
    fig = plt.figure(figsize=(14, 3 * n))
    gs  = gridspec.GridSpec(n, 1, hspace=0.5)
    t   = df["timestamp_s"]

    for i, (col, label, color) in enumerate(METRICS):
        ax = fig.add_subplot(gs[i])
        ax.plot(t, df[col], color=color, linewidth=0.8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlim(t.iloc[0], t.iloc[-1])
        ax.grid(True, linestyle=":", alpha=0.5)
        if i < n - 1:
            ax.set_xticklabels([])

    fig.text(0.5, 0.02, "Time (s)", ha="center", fontsize=10)
    fig.suptitle(out_path.stem, fontsize=12, y=1.01)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_throttle(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Dedicated throttling chart: SM clock + temperature on dual axes.

    Annotates the first moment the clock drops ≥9% below its peak,
    which is the threshold that corresponds to thermal throttling.
    """
    peak_clock        = df["clock_sm_mhz"].max()
    throttle_thresh   = peak_clock * 0.91
    throttled         = df[df["clock_sm_mhz"] < throttle_thresh]
    temp_thresh_c     = 75

    t = df["timestamp_s"]

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    # SM clock (left axis)
    ax1.plot(t, df["clock_sm_mhz"], color="tab:blue", linewidth=1.0, label="SM Clock")
    ax1.axhline(throttle_thresh, color="tab:blue", linestyle="--", linewidth=0.8,
                label=f"Throttle threshold ({throttle_thresh:.0f} MHz = 91% of peak)")
    ax1.set_ylabel("SM Clock (MHz)", color="tab:blue", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(bottom=df["clock_sm_mhz"].min() * 0.95)

    # Temperature (right axis)
    ax2.plot(t, df["temp_c"], color="tab:orange", linewidth=1.0, alpha=0.85, label="Temp")
    ax2.axhline(temp_thresh_c, color="tab:orange", linestyle="--", linewidth=0.8,
                label=f"{temp_thresh_c} °C threshold")
    ax2.set_ylabel("Temperature (°C)", color="tab:orange", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Shade throttled region
    if not throttled.empty:
        ax1.fill_between(t, throttle_thresh, df["clock_sm_mhz"],
                         where=df["clock_sm_mhz"] < throttle_thresh,
                         color="tab:blue", alpha=0.15, label="Throttled region")

        first_t    = throttled["timestamp_s"].iloc[0]
        first_clk  = throttled["clock_sm_mhz"].iloc[0]
        first_temp = df.loc[throttled.index[0], "temp_c"]
        drop_pct   = (1.0 - first_clk / peak_clock) * 100.0

        ax1.annotate(
            f"Throttle onset\nt={first_t:.0f}s  {first_clk:.0f} MHz\n"
            f"({drop_pct:.1f}% drop)  {first_temp}°C",
            xy=(first_t, first_clk),
            xytext=(first_t + t.iloc[-1] * 0.03, throttle_thresh * 1.01),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )
    else:
        ax1.text(0.5, 0.5, "No throttling detected",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=12, color="green", alpha=0.6)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")

    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.set_xlim(t.iloc[0], t.iloc[-1])
    ax1.grid(True, linestyle=":", alpha=0.4)
    fig.suptitle(f"{out_path.stem} — Throttle Analysis", fontsize=12)

    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(pathlib.Path("data").glob("run_*.csv"))
        if not paths:
            sys.exit("No CSV files found.")

    for p in map(pathlib.Path, paths):
        df = load(str(p))
        plot_overview(df, p.with_suffix(".png"))
        plot_throttle(df, p.with_name(p.stem + "_throttle.png"))


if __name__ == "__main__":
    main()
