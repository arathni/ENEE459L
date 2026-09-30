from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    times = []
    bench.workload.synchronize()

    for i in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()

        times.append((end - start) / 1_000_000.0)

    return times


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown("samples", "too few samples(less than 4)")

    median = statistics.median(samples[len(samples) // 2:])

    if median <= 0:
        return unknown("samples", "median <= 0")

    threshold = median * (1 + WARMUP_TOL)
    count = 0

    for sample in samples:
        if sample <= threshold:
            break
        count += 1

    return measured(
        count,
        f"leading prefix above (1 + {WARMUP_TOL}) x median of the run's second half",
        settled_rate_ms=median,
        threshold_ms=threshold,
        tolerance=WARMUP_TOL,
        retained=len(samples) - count,
    )



def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    s = sorted(samples)
    n = len(s)

    out = {}
    out["n"] = n
    out["mean"] = round(statistics.fmean(s), 4)

    if n > 2:
        out["std"] = round(statistics.stdev(s), 4)
    else:
        out["std"] = 0.0

    out["min"] = round(s[0], 4)
    out["max"] = round(s[-1], 4)

    for q in [.5, .95, .99]:
        h = (n - 1) * q
        i = int(h)

        if i + 1 < n:
            value = s[i] + (h - i) * (s[i + 1] - s[i])
        else:
            value = s[i]

        out[f"p{int(q * 100)}"] = round(value, 4)

    return out

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    if len(samples) < MIN_SAMPLES_FOR_MODALITY:
        return unknown("samples", "not enough samples (less than 20)")

    s = sorted(samples)
    trim = int(len(s) * .05)
    trimmed = s[trim:-trim]
    gaps = []

    for i in range(len(trimmed) - 1):
        gaps.append(trimmed[i + 1] - trimmed[i])

    median_gap = statistics.median(gaps)

    if median_gap <= 0:
        return unknown("samples", "timer resolution is too coarse")

    widest_gap = max(gaps)
    ratio = widest_gap / median_gap
    split = trim + gaps.index(widest_gap) + 1
    left_count = len(s[:split])
    right_count = len(s[split:])

    return measured(
        ratio >= MULTIMODAL_GAP_RATIO
        and left_count / len(s) >= MIN_MODE_FRACTION
        and right_count / len(s) >= MIN_MODE_FRACTION,
        f"widest trimmed gap >= {MULTIMODAL_GAP_RATIO}x the median gap, with >= {MIN_MODE_FRACTION:.0%} of samples on each side",
        gap_ratio=round(ratio, 2),
        widest_gap_ms=round(widest_gap, 4),
        typical_gap_ms=round(median_gap, 5),
        modes=[
            {
                "n": left_count,
                "share": left_count / len(s),
                "median_ms": round(statistics.median(s[:split]), 4),
            },
            {
                "n": right_count,
                "share": right_count / len(s),
                "median_ms": round(statistics.median(s[split:]), 4),
            },
        ],
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    result = bench.runner(["nvpmodel", "-q"])

    if result.returncode != 0:
        return unknown("nvpmodel -q", result.stdout)

    lines = result.stdout.splitlines()

    for i in range(len(lines)):
        if "NV Power Mode:" in lines[i]:
            name = lines[i].split("NV Power Mode:", 1)[1].strip()
            mode = int(lines[i + 1].strip())
            break

    minimum = read_text(bench.telemetry, CPUFREQ_MIN)
    maximum = read_text(bench.telemetry, CPUFREQ_MAX)
    out = measured(name, "nvpmodel -q", mode_index=mode)

    if minimum is None or maximum is None:
        out["jetson_clocks"] = None
    else:
        out["jetson_clocks"] = int(minimum) == int(maximum)

    return out



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    temperatures = []

    for zone in (bench.telemetry / THERMAL_ZONES).glob("thermal_zone*"):
        try:
            raw = read_text(zone, "temp")
        except TypeError:
            continue

        if raw is None:
            continue

        temperature = int(raw)

        if temperature <= -1000:
            continue

        temperatures.append(temperature / 1000.0)

    out = {}

    if temperatures:
        out["temperature_c"] = measured(max(temperatures), f"{THERMAL_ZONES}/*/temp")
    else:
        out["temperature_c"] = unknown(f"{THERMAL_ZONES}/*/temp", "no readable temperatures")

    power = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)

    if power is None:
        out["power_mw"] = unknown(" | ".join(POWER_RAIL_CANDIDATES), "none of the documented INA3221 rail paths could be read")
    else:
        out["power_mw"] = measured(int(power[1]), power[0])

    load = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)

    if load is None:
        out["gpu_utilization_percent"] = unknown(" | ".join(GPU_LOAD_CANDIDATES), "none of the documented GPU load paths could be read")
    else:
        out["gpu_utilization_percent"] = measured(int(load[1]) / 10.0, load[0])

    return out

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
