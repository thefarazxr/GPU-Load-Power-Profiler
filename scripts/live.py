#!/usr/bin/env python3
"""
Live terminal monitor + real-time plot for gpu_profiler telemetry.
Run alongside the profiler in a second terminal:

  python scripts/live.py                    # auto-picks newest CSV in data/
  python scripts/live.py data/run_xyz.csv   # explicit path
  python scripts/live.py --no-plot          # terminal table only

Refreshes every 500ms. Press Ctrl+C to exit.
"""
import os
import pathlib
import sys
import time
import threading

import pandas as pd

REFRESH_S       = 0.5
WINDOW_S        = 120     # rolling plot window (seconds)
THROTTLE_THRESH = 0.91
TEMP_WARN_C     = 75

TERM_METRICS = [
    ("Power",       "power_w",          "W",   1),
    ("Temperature", "temp_c",           "°C",  0),
    ("SM Clock",    "clock_sm_mhz",     "MHz", 0),
    ("Mem Clock",   "clock_mem_mhz",    "MHz", 0),
    ("SM Util",     "utilization_gpu",  "%",   0),
    ("Mem Util",    "utilization_mem",  "%",   0),
    ("VRAM Used",   "used_mem_gb",      "GB",  2),
]

PLOT_METRICS = [
    ("power_w",         "Power (W)",      "tab:red"),
    ("temp_c",          "Temp (°C)",      "tab:orange"),
    ("clock_sm_mhz",    "SM Clock (MHz)", "tab:blue"),
    ("utilization_gpu", "SM Util (%)",    "tab:green"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def find_latest_csv() -> pathlib.Path | None:
    csvs = sorted(pathlib.Path("data").glob("run_*.csv"), key=os.path.getmtime)
    return csvs[-1] if csvs else None


def load(path: pathlib.Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.empty:
            return df
        df["timestamp_s"] = df["timestamp_ms"] / 1000.0
        df["power_w"]     = df["power_mw"]     / 1000.0
        df["used_mem_gb"] = df["used_memory_bytes"] / 1024**3
        return df
    except Exception:
        return pd.DataFrame()


# ── Terminal table ────────────────────────────────────────────────────────────

def render(df: pd.DataFrame, path: pathlib.Path) -> str:
    if df.empty:
        return "  Waiting for data..."

    last       = df.iloc[-1]
    peak_clock = df["clock_sm_mhz"].max()
    throttling = last["clock_sm_mhz"] < peak_clock * THROTTLE_THRESH
    overtemp   = last["temp_c"] >= TEMP_WARN_C

    lines = []
    lines.append(f"  File    : {path.name}")
    lines.append(f"  Samples : {len(df)}   Runtime : {last['timestamp_s']:.1f}s")
    lines.append("")
    lines.append(f"  {'Metric':<20} {'Now':>9} {'Mean':>9} {'Max':>9}")
    lines.append("  " + "-" * 50)

    for label, col, unit, dec in TERM_METRICS:
        cur  = f"{last[col]:.{dec}f}{unit}"
        mean = f"{df[col].mean():.{dec}f}{unit}"
        mx   = f"{df[col].max():.{dec}f}{unit}"
        lines.append(f"  {label:<20} {cur:>9} {mean:>9} {mx:>9}")

    lines.append("")
    status_parts = []
    if throttling:
        drop = (1 - last["clock_sm_mhz"] / peak_clock) * 100
        status_parts.append(f"THROTTLING ({drop:.1f}% clock drop)")
    if overtemp:
        status_parts.append(f"TEMP >= {TEMP_WARN_C}°C")
    lines.append("  Status  : " + ("  |  ".join(status_parts) if status_parts else "OK"))

    return "\n".join(lines)


def clear_lines(n: int) -> None:
    if n > 0:
        print(f"\033[{n}A\033[J", end="", flush=True)


def run_terminal(path_ref: list, stop: threading.Event) -> None:
    prev_lines = 0
    while not stop.is_set():
        if path_ref[0] is None:
            path_ref[0] = find_latest_csv()

        if path_ref[0] is None:
            msg = "  Waiting for profiler to start..."
            clear_lines(prev_lines)
            print(msg, flush=True)
            prev_lines = 1
        else:
            df  = load(path_ref[0])
            out = render(df, path_ref[0])
            clear_lines(prev_lines)
            print(out, flush=True)
            prev_lines = out.count("\n") + 1

        time.sleep(REFRESH_S)


# ── Live plot (matplotlib) ────────────────────────────────────────────────────

def run_plot(path_ref: list, stop: threading.Event) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, axes = plt.subplots(len(PLOT_METRICS), 1, figsize=(12, 8), sharex=True)
    fig.suptitle("GPU Profiler — Live", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    lines = []
    for ax, (col, label, color) in zip(axes, PLOT_METRICS):
        line, = ax.plot([], [], color=color, linewidth=0.9)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        lines.append((line, col, ax))

    axes[-1].set_xlabel("Time (s)", fontsize=9)

    def update(_):
        if path_ref[0] is None:
            path_ref[0] = find_latest_csv()
        if path_ref[0] is None:
            return [l for l, _, _ in lines]

        df = load(path_ref[0])
        if df.empty:
            return [l for l, _, _ in lines]

        t_max  = df["timestamp_s"].iloc[-1]
        t_min  = max(0.0, t_max - WINDOW_S)
        window = df[df["timestamp_s"] >= t_min]

        for line, col, ax in lines:
            line.set_data(window["timestamp_s"], window[col])
            ax.relim()
            ax.autoscale_view()

        axes[0].set_xlim(t_min, t_max + 1)

        # mark throttle threshold on SM clock panel (index 2)
        clock_ax = axes[2]
        peak     = df["clock_sm_mhz"].max()
        thresh   = peak * THROTTLE_THRESH
        for coll in clock_ax.collections:
            coll.remove()
        clock_ax.axhline(thresh, color="tab:blue", linestyle="--",
                         linewidth=0.7, alpha=0.6)

        return [l for l, _, _ in lines]

    fig._ani = animation.FuncAnimation(fig, update, interval=int(REFRESH_S * 1000),
                                      blit=False, cache_frame_data=False)

    def on_close(_):
        stop.set()

    fig.canvas.mpl_connect("close_event", on_close)

    try:
        plt.show()
    except Exception:
        pass
    finally:
        stop.set()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args      = sys.argv[1:]
    no_plot   = "--no-plot" in args
    csv_args  = [a for a in args if not a.startswith("--")]

    path_ref  = [pathlib.Path(csv_args[0]) if csv_args else None]
    stop      = threading.Event()

    print("GPU Profiler — Live Monitor  (Ctrl+C or close plot to exit)\n")

    term_thread = threading.Thread(target=run_terminal, args=(path_ref, stop), daemon=True)
    term_thread.start()

    if no_plot:
        try:
            term_thread.join()
        except KeyboardInterrupt:
            pass
    else:
        try:
            run_plot(path_ref, stop)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n  Plot unavailable ({e}). Running terminal-only.\n")
            try:
                term_thread.join()
            except KeyboardInterrupt:
                pass

    stop.set()
    print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
