#!/usr/bin/env python3
"""
Analyze telemetry CSVs produced by gpu_profiler.
Usage: python scripts/analyze.py data/run_*.csv
"""
import sys
import pathlib
import pandas as pd


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_s"] = df["timestamp_ms"] / 1000.0
    df["power_w"]     = df["power_mw"]    / 1000.0
    df["used_mem_gb"] = df["used_memory_bytes"] / 1024**3
    return df


def summarize(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    cols = {
        "power_w":          "Power (W)",
        "temp_c":           "Temp  (°C)",
        "clock_sm_mhz":     "SM Clock (MHz)",
        "clock_mem_mhz":    "Mem Clock (MHz)",
        "utilization_gpu":  "SM Util (%)",       # nvmlDeviceGetUtilizationRates().gpu = SM active %
        "utilization_mem":  "Mem Ctrl Util (%)",
        "used_mem_gb":      "VRAM Used (GB)",
    }
    stats = df[list(cols)].describe().loc[["mean", "min", "max", "std"]]
    stats.columns = list(cols.values())
    print(stats.to_string())

    # SM utilization summary line
    sm = df["utilization_gpu"]
    time_above_99 = (sm >= 99).sum() * df["timestamp_s"].diff().median()
    print(f"\nSM utilization  mean={sm.mean():.1f}%  min={sm.min()}%  max={sm.max()}%  "
          f"time≥99%={time_above_99:.1f}s")

    # Throttle detection: SM clock drops ≥9% from its peak
    peak_clock = df["clock_sm_mhz"].max()
    throttle_threshold = peak_clock * 0.91
    throttled = df[df["clock_sm_mhz"] < throttle_threshold]
    if throttled.empty:
        print(f"\nNo throttling detected (peak SM clock {peak_clock} MHz).")
    else:
        first_t = throttled["timestamp_s"].iloc[0]
        print(f"\nThrottling detected at t={first_t:.1f}s "
              f"(clock dropped below {throttle_threshold:.0f} MHz = 91% of {peak_clock} MHz peak)")

    # Temperature alert
    max_temp = df["temp_c"].max()
    if max_temp >= 75:
        t_over = df[df["temp_c"] >= 75]["timestamp_s"].iloc[0]
        print(f"Temperature exceeded 75 °C at t={t_over:.1f}s (peak {max_temp} °C)")


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(pathlib.Path("data").glob("run_*.csv"))
        if not paths:
            sys.exit("No CSV files found. Pass paths as arguments or run from project root.")

    for p in paths:
        df = load(str(p))
        summarize(df, pathlib.Path(p).name)


if __name__ == "__main__":
    main()
