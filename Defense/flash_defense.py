#!/usr/bin/env python3

from __future__ import annotations

import json
import platform
import re
import subprocess
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd


ROOT = "."
OUT_DIR = "defense_outputs"

TEST_FOLDERS = ["TEST1_SPATIAL", "TEST2_HEIGHT", "TEST3_POWER"]
EXPOSURES = ["short", "medium", "long"]
EXPOSURE_TIMES = {"short": 0.25, "medium": 1.0, "long": 4.0}
CLEAN_GAINS = {"short": 0.25, "medium": 1.0, "long": 4.0}

CORRUPTION_STRENGTHS = [4, 8, 16, 32, 64, 128, 256]
DETECTOR_THRESHOLD = 0.25
FLASH_GAIN = 4.0
BIAS_MODE = "additive_plus_floor"
BIAS_STRENGTH = 4.0
MASK_KERNEL_SIZE = 41
SENSOR_MAX_SCALE = 1.5
TONE_PERCENTILE = 99.7
TONE_KEY = 0.18
GAMMA = 1.0
SEVERE_THRESHOLD = -50.0
NEAR_BLACK_THRESHOLD = -80.0
SAVE_IMAGES = False
SAVE_WORST_N = 12


def safe_id(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_") or "item"


def read_exr(path: Path) -> np.ndarray:
    img = np.squeeze(iio.imread(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img[..., :3].astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(np.maximum(img, 0.0))


def luma_rgb(img: np.ndarray) -> np.ndarray:
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def mean_luma(img: np.ndarray) -> float:
    return float(np.mean(luma_rgb(img)))


def scale_flash_to_peak_ratio(flash: np.ndarray, ambient: np.ndarray, peak_ratio: float) -> np.ndarray:
    flash_y = luma_rgb(flash)
    amb_y = luma_rgb(ambient)
    flash_ref = float(np.percentile(flash_y, 99.9))
    amb_ref = float(np.percentile(amb_y, 99.9))
    if flash_ref <= 1e-9:
        return np.zeros_like(flash, dtype=np.float32)
    scale = (peak_ratio * max(amb_ref, 1e-9)) / flash_ref
    return np.ascontiguousarray(flash * scale)


def make_clean_exposures(ambient: np.ndarray, sensor_max: float) -> dict[str, np.ndarray]:
    out = {}
    for exp in EXPOSURES:
        raw = ambient * CLEAN_GAINS[exp]
        out[exp] = np.clip(raw / sensor_max, 0.0, 1.0).astype(np.float32)
    return out


def make_attack_exposures(
    clean_exps: dict[str, np.ndarray],
    ambient: np.ndarray,
    flash_scaled: np.ndarray,
    sensor_max: float,
    flash_gain: float,
    bias_mode: str,
    bias_strength: float,
) -> dict[str, np.ndarray]:
    out = {
        "short": clean_exps["short"].copy(),
        "medium": clean_exps["medium"].copy(),
        "long": None,
    }

    attack_raw = ambient * CLEAN_GAINS["long"] + flash_scaled * flash_gain

    if bias_mode == "additive_plus_floor":
        amb_ref = float(np.percentile(luma_rgb(ambient), 99.9))
        attack_raw = attack_raw + (bias_strength * amb_ref)

    elif bias_mode == "saturating_mask":
        fy = luma_rgb(flash_scaled)
        if np.max(fy) > 1e-9:
            mask = (fy > np.percentile(fy, 99.0)).astype(np.uint8)
            kernel = np.ones((MASK_KERNEL_SIZE, MASK_KERNEL_SIZE), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1).astype(bool)
            amb_ref = float(np.percentile(luma_rgb(ambient), 99.9))
            attack_raw[mask] = attack_raw[mask] + bias_strength * amb_ref

    out["long"] = np.clip(attack_raw / sensor_max, 0.0, 1.0).astype(np.float32)
    return out


def fuse_hdr(
    exps: dict[str, np.ndarray],
    use_exposures: list[str],
    tone_percentile: float,
    tone_key: float,
    gamma: float,
) -> np.ndarray:
    weights = {"short": 0.20, "medium": 0.30, "long": 0.50}
    num = None
    den = 0.0

    for exp in use_exposures:
        rad = exps[exp] / max(EXPOSURE_TIMES[exp], 1e-6)
        w = weights[exp]
        num = rad * w if num is None else num + rad * w
        den += w

    hdr = num / max(den, 1e-6)
    y = luma_rgb(hdr)
    ref = float(np.percentile(y, tone_percentile))
    scale = tone_key / max(ref, 1e-9)

    mapped = (scale * hdr) / (1.0 + scale * hdr)
    mapped = np.clip(mapped, 0.0, 1.0)

    if gamma != 1.0:
        mapped = np.power(mapped, 1.0 / gamma)

    return mapped.astype(np.float32)


def decide_rejection(
    clean_exps: dict[str, np.ndarray],
    attack_exps: dict[str, np.ndarray],
    detector_threshold: float,
) -> tuple[str, dict[str, float]]:
    eps = 1e-9
    ratios = {}
    residuals = {}

    for exp in EXPOSURES:
        c = mean_luma(clean_exps[exp]) + eps
        a = mean_luma(attack_exps[exp]) + eps
        ratios[exp] = a / c
        residuals[exp] = abs(float(np.log(ratios[exp])))

    rejected = max(residuals, key=residuals.get)
    if residuals[rejected] < detector_threshold:
        rejected = "none"

    debug = {}

    for exp in EXPOSURES:
        debug[f"{exp}_ratio"] = ratios[exp]
        debug[f"{exp}_log_abs_residual"] = residuals[exp]
        debug[f"{exp}_mean_clean"] = mean_luma(clean_exps[exp])
        debug[f"{exp}_mean_attack"] = mean_luma(attack_exps[exp])

    debug["max_log_abs_residual"] = max(residuals.values())
    return rejected, debug


def delta_percent(test: float, ref: float) -> float:
    if abs(ref) < 1e-12:
        return np.nan
    return 100.0 * (test - ref) / ref


def summarize_delta(vals, severe_threshold: float, near_black_threshold: float) -> dict[str, float]:
    s = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()

    if len(s) == 0:
        return {
            "n": 0,
            "mean_delta_percent": np.nan,
            "median_delta_percent": np.nan,
            "minimum_delta_percent": np.nan,
            "severe_rate_percent": np.nan,
            "near_black_rate_percent": np.nan,
        }

    return {
        "n": int(len(s)),
        "mean_delta_percent": float(s.mean()),
        "median_delta_percent": float(s.median()),
        "minimum_delta_percent": float(s.min()),
        "severe_rate_percent": float((s <= severe_threshold).mean() * 100.0),
        "near_black_rate_percent": float((s <= near_black_threshold).mean() * 100.0),
    }


def p95(vals) -> float:
    s = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(np.percentile(s, 95))


def save_rgb(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img8, cv2.COLOR_RGB2BGR))


def make_panel(clean: np.ndarray, undefended: np.ndarray, defended: np.ndarray, label: str) -> np.ndarray:
    panel = np.concatenate([clean, undefended, defended], axis=1)
    panel8 = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
    panel8 = cv2.cvtColor(panel8, cv2.COLOR_RGB2BGR)

    cv2.putText(
        panel8,
        "clean | undefended FLASH | defended",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        panel8,
        label,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    return panel8


def run() -> None:
    root = Path(ROOT).resolve()
    outdir = Path(OUT_DIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    clean_fp_rows = []

    for test_name in TEST_FOLDERS:
        test_dir = root / test_name
        ambient_path = test_dir / "amb_night.exr"
        flash_dir = test_dir / "flash_sweep"

        if not test_dir.exists():
            continue
        if not ambient_path.exists():
            continue
        if not flash_dir.exists():
            continue

        ambient = read_exr(ambient_path)
        sensor_max = float(np.max(ambient) * SENSOR_MAX_SCALE)
        clean_exps = make_clean_exposures(ambient, sensor_max)

        clean_img = fuse_hdr(
            clean_exps,
            EXPOSURES,
            tone_percentile=TONE_PERCENTILE,
            tone_key=TONE_KEY,
            gamma=GAMMA,
        )

        clean_mean = mean_luma(clean_img)

        clean_reject, clean_debug = decide_rejection(
            clean_exps,
            clean_exps,
            DETECTOR_THRESHOLD,
        )

        clean_fp_rows.append({
            "test": test_name,
            "condition": "night",
            "clean_false_rejection": 1 if clean_reject != "none" else 0,
            "clean_rejected_exposure": clean_reject,
            **clean_debug,
        })

        flash_files = sorted(flash_dir.glob("*.exr"))

        for flash_path in flash_files:
            flash_raw = read_exr(flash_path)

            for strength in CORRUPTION_STRENGTHS:
                scaled_flash = scale_flash_to_peak_ratio(flash_raw, ambient, strength)

                attack_exps = make_attack_exposures(
                    clean_exps=clean_exps,
                    ambient=ambient,
                    flash_scaled=scaled_flash,
                    sensor_max=sensor_max,
                    flash_gain=FLASH_GAIN,
                    bias_mode=BIAS_MODE,
                    bias_strength=BIAS_STRENGTH,
                )

                t0 = time.perf_counter()
                rejected, debug = decide_rejection(
                    clean_exps,
                    attack_exps,
                    DETECTOR_THRESHOLD,
                )
                t1 = time.perf_counter()

                undefended_img = fuse_hdr(
                    attack_exps,
                    EXPOSURES,
                    tone_percentile=TONE_PERCENTILE,
                    tone_key=TONE_KEY,
                    gamma=GAMMA,
                )
                t2 = time.perf_counter()

                used_exposures = [e for e in EXPOSURES if e != rejected] if rejected != "none" else EXPOSURES

                defended_img = fuse_hdr(
                    attack_exps,
                    used_exposures,
                    tone_percentile=TONE_PERCENTILE,
                    tone_key=TONE_KEY,
                    gamma=GAMMA,
                )
                t3 = time.perf_counter()

                undef_mean = mean_luma(undefended_img)
                def_mean = mean_luma(defended_img)

                before = delta_percent(undef_mean, clean_mean)
                after = delta_percent(def_mean, clean_mean)

                detail_rows.append({
                    "test": test_name,
                    "condition": "night",
                    "flash_file": flash_path.name,
                    "corruption_strength": strength,
                    "bias_mode": BIAS_MODE,
                    "bias_strength": BIAS_STRENGTH,
                    "rejected_exposure": rejected,
                    "used_exposures": "+".join(used_exposures),
                    "corrupted_exposure_rejected": rejected == "long",
                    "clean_mean_luma": clean_mean,
                    "undefended_mean_luma": undef_mean,
                    "defended_mean_luma": def_mean,
                    "delta_before_percent": before,
                    "delta_after_percent": after,
                    "improvement_pp": after - before,
                    "attack_success_severe_before": before <= SEVERE_THRESHOLD,
                    "attack_success_severe_after": after <= SEVERE_THRESHOLD,
                    "attack_success_near_black_before": before <= NEAR_BLACK_THRESHOLD,
                    "attack_success_near_black_after": after <= NEAR_BLACK_THRESHOLD,
                    "detection_ms": (t1 - t0) * 1000.0,
                    "undefended_fusion_ms": (t2 - t1) * 1000.0,
                    "defended_fusion_ms": (t3 - t2) * 1000.0,
                    "total_defense_ms": ((t1 - t0) + (t3 - t2)) * 1000.0,
                    **debug,
                })

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(outdir / "defense_per_case.csv", index=False)

    clean_fp = pd.DataFrame(clean_fp_rows)
    clean_fp.to_csv(outdir / "defense_clean_false_rejections.csv", index=False)

    if len(detail) == 0:
        print("No cases processed.")
        return

    summary_rows = []

    for (test, strength), g in detail.groupby(["test", "corruption_strength"]):
        before = summarize_delta(g["delta_before_percent"], SEVERE_THRESHOLD, NEAR_BLACK_THRESHOLD)
        after = summarize_delta(g["delta_after_percent"], SEVERE_THRESHOLD, NEAR_BLACK_THRESHOLD)

        summary_rows.append({
            "test": test,
            "condition": "night",
            "corruption_strength": strength,
            "cases": len(g),
            "rejected_cases": int((g["rejected_exposure"] != "none").sum()),
            "corrupted_exposure_rejected_cases": int(g["corrupted_exposure_rejected"].sum()),
            "rejection_rate_percent": 100.0 * (g["rejected_exposure"] != "none").mean(),
            "corrupted_exposure_rejection_rate_percent": 100.0 * g["corrupted_exposure_rejected"].mean(),
            "severe_before_percent": before["severe_rate_percent"],
            "severe_after_percent": after["severe_rate_percent"],
            "severe_reduction_pp": before["severe_rate_percent"] - after["severe_rate_percent"],
            "near_black_before_percent": before["near_black_rate_percent"],
            "near_black_after_percent": after["near_black_rate_percent"],
            "near_black_reduction_pp": before["near_black_rate_percent"] - after["near_black_rate_percent"],
            "median_delta_before_percent": before["median_delta_percent"],
            "median_delta_after_percent": after["median_delta_percent"],
            "minimum_delta_before_percent": before["minimum_delta_percent"],
            "minimum_delta_after_percent": after["minimum_delta_percent"],
        })

    by_test_strength = pd.DataFrame(summary_rows)
    by_test_strength.to_csv(outdir / "defense_summary_by_test_strength.csv", index=False)

    strength_rows = []

    for strength, g in detail.groupby("corruption_strength"):
        before = summarize_delta(g["delta_before_percent"], SEVERE_THRESHOLD, NEAR_BLACK_THRESHOLD)
        after = summarize_delta(g["delta_after_percent"], SEVERE_THRESHOLD, NEAR_BLACK_THRESHOLD)

        strength_rows.append({
            "condition": "night",
            "corruption_strength": strength,
            "cases": len(g),
            "rejected_cases": int((g["rejected_exposure"] != "none").sum()),
            "corrupted_exposure_rejected_cases": int(g["corrupted_exposure_rejected"].sum()),
            "rejection_rate_percent": 100.0 * (g["rejected_exposure"] != "none").mean(),
            "corrupted_exposure_rejection_rate_percent": 100.0 * g["corrupted_exposure_rejected"].mean(),
            "severe_before_percent": before["severe_rate_percent"],
            "severe_after_percent": after["severe_rate_percent"],
            "severe_reduction_pp": before["severe_rate_percent"] - after["severe_rate_percent"],
            "near_black_before_percent": before["near_black_rate_percent"],
            "near_black_after_percent": after["near_black_rate_percent"],
            "near_black_reduction_pp": before["near_black_rate_percent"] - after["near_black_rate_percent"],
            "median_delta_before_percent": before["median_delta_percent"],
            "median_delta_after_percent": after["median_delta_percent"],
            "minimum_delta_before_percent": before["minimum_delta_percent"],
            "minimum_delta_after_percent": after["minimum_delta_percent"],
        })

    by_strength = pd.DataFrame(strength_rows)
    by_strength.to_csv(outdir / "defense_summary_by_strength.csv", index=False)

    before = summarize_delta(detail["delta_before_percent"], SEVERE_THRESHOLD, NEAR_BLACK_THRESHOLD)
    after = summarize_delta(detail["delta_after_percent"], SEVERE_THRESHOLD, NEAR_BLACK_THRESHOLD)

    overall = pd.DataFrame([{
        "condition": "night",
        "cases": len(detail),
        "rejected_cases": int((detail["rejected_exposure"] != "none").sum()),
        "corrupted_exposure_rejected_cases": int(detail["corrupted_exposure_rejected"].sum()),
        "rejection_rate_percent": 100.0 * (detail["rejected_exposure"] != "none").mean(),
        "corrupted_exposure_rejection_rate_percent": 100.0 * detail["corrupted_exposure_rejected"].mean(),
        "severe_before_percent": before["severe_rate_percent"],
        "severe_after_percent": after["severe_rate_percent"],
        "severe_reduction_pp": before["severe_rate_percent"] - after["severe_rate_percent"],
        "near_black_before_percent": before["near_black_rate_percent"],
        "near_black_after_percent": after["near_black_rate_percent"],
        "near_black_reduction_pp": before["near_black_rate_percent"] - after["near_black_rate_percent"],
        "median_delta_before_percent": before["median_delta_percent"],
        "median_delta_after_percent": after["median_delta_percent"],
        "minimum_delta_before_percent": before["minimum_delta_percent"],
        "minimum_delta_after_percent": after["minimum_delta_percent"],
    }])

    overall.to_csv(outdir / "defense_summary_overall.csv", index=False)

    counts = detail["rejected_exposure"].value_counts(dropna=False).reset_index()
    counts.columns = ["rejected_exposure", "count"]
    counts.to_csv(outdir / "defense_rejection_counts.csv", index=False)

    latency = pd.DataFrame([{
        "num_cases": len(detail),
        "median_detection_ms": float(detail["detection_ms"].median()),
        "p95_detection_ms": p95(detail["detection_ms"]),
        "max_detection_ms": float(detail["detection_ms"].max()),
        "median_defended_fusion_ms": float(detail["defended_fusion_ms"].median()),
        "p95_defended_fusion_ms": p95(detail["defended_fusion_ms"]),
        "max_defended_fusion_ms": float(detail["defended_fusion_ms"].max()),
        "median_total_defense_ms": float(detail["total_defense_ms"].median()),
        "p95_total_defense_ms": p95(detail["total_defense_ms"]),
        "max_total_defense_ms": float(detail["total_defense_ms"].max()),
    }])

    latency.to_csv(outdir / "defense_latency_summary.csv", index=False)

    if SAVE_IMAGES:
        imgdir = outdir / "defense_examples"
        imgdir.mkdir(parents=True, exist_ok=True)

        worst = detail.sort_values("delta_before_percent", ascending=True).head(SAVE_WORST_N)
        example_rows = []

        for _, row in worst.iterrows():
            test_dir = root / row["test"]
            ambient = read_exr(test_dir / "amb_night.exr")
            sensor_max = float(np.max(ambient) * SENSOR_MAX_SCALE)
            clean_exps = make_clean_exposures(ambient, sensor_max)

            clean_img = fuse_hdr(
                clean_exps,
                EXPOSURES,
                tone_percentile=TONE_PERCENTILE,
                tone_key=TONE_KEY,
                gamma=GAMMA,
            )

            flash_raw = read_exr(test_dir / "flash_sweep" / row["flash_file"])
            scaled_flash = scale_flash_to_peak_ratio(flash_raw, ambient, float(row["corruption_strength"]))

            attack_exps = make_attack_exposures(
                clean_exps=clean_exps,
                ambient=ambient,
                flash_scaled=scaled_flash,
                sensor_max=sensor_max,
                flash_gain=FLASH_GAIN,
                bias_mode=BIAS_MODE,
                bias_strength=BIAS_STRENGTH,
            )

            undefended_img = fuse_hdr(
                attack_exps,
                EXPOSURES,
                tone_percentile=TONE_PERCENTILE,
                tone_key=TONE_KEY,
                gamma=GAMMA,
            )

            rejected = row["rejected_exposure"]
            used = [e for e in EXPOSURES if e != rejected] if rejected != "none" else EXPOSURES

            defended_img = fuse_hdr(
                attack_exps,
                used,
                tone_percentile=TONE_PERCENTILE,
                tone_key=TONE_KEY,
                gamma=GAMMA,
            )

            base = (
                f"{safe_id(row['test'])}_night_"
                f"strength{safe_id(row['corruption_strength'])}_"
                f"{safe_id(Path(row['flash_file']).stem)}"
            )

            save_rgb(imgdir / f"{base}_clean.png", clean_img)
            save_rgb(imgdir / f"{base}_undefended.png", undefended_img)
            save_rgb(imgdir / f"{base}_defended.png", defended_img)

            label = (
                f"before={row['delta_before_percent']:.1f}%, "
                f"after={row['delta_after_percent']:.1f}%, "
                f"reject={row['rejected_exposure']}"
            )

            panel = make_panel(clean_img, undefended_img, defended_img, label)
            panel_path = imgdir / f"{base}_panel.png"
            cv2.imwrite(str(panel_path), panel)

            example_rows.append({
                "test": row["test"],
                "flash_file": row["flash_file"],
                "corruption_strength": row["corruption_strength"],
                "rejected_exposure": row["rejected_exposure"],
                "delta_before_percent": row["delta_before_percent"],
                "delta_after_percent": row["delta_after_percent"],
                "panel": str(panel_path),
            })

        pd.DataFrame(example_rows).to_csv(outdir / "defense_example_manifest.csv", index=False)

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tests": TEST_FOLDERS,
        "ambient": "night",
        "corruption_strengths": CORRUPTION_STRENGTHS,
        "detector_threshold": DETECTOR_THRESHOLD,
        "flash_gain": FLASH_GAIN,
        "bias_mode": BIAS_MODE,
        "bias_strength": BIAS_STRENGTH,
        "sensor_max_scale": SENSOR_MAX_SCALE,
        "tone_percentile": TONE_PERCENTILE,
        "tone_key": TONE_KEY,
        "gamma": GAMMA,
        "severe_threshold": SEVERE_THRESHOLD,
        "near_black_threshold": NEAR_BLACK_THRESHOLD,
    }

    try:
        env["lscpu"] = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        env["lscpu"] = "unavailable"

    with open(outdir / "defense_runtime_environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    run()
