import csv
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np


ROOT = Path.home() / "Desktop" / "NEWSIMULATION_THREESETS"
TEST_FOLDER = "TEST1_SPATIAL"

EXPOSURE_TIMES = np.array([0.25, 1.0, 4.0], dtype=np.float32)
EXPOSURE_GAINS = {
    "short": 0.25,
    "medium": 1.0,
    "long": 4.0,
}
FLASH_GAINS = {
    "short": 0.0,
    "medium": 0.15,
    "long": 0.85,
}
SENSOR_MAX_SCALE = 1.5
TONEMAP_GAMMA = 1.5


def read_exr(path: Path) -> np.ndarray:
    img = np.squeeze(iio.imread(path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img[..., :3].astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(np.maximum(img, 0.0))


def mean_gray(img: np.ndarray) -> float:
    return float(np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)))


def process_isp(ambient: np.ndarray, flash: np.ndarray, sensor_max: float) -> np.ndarray:
    exp_short = ambient * EXPOSURE_GAINS["short"] + flash * FLASH_GAINS["short"]
    exp_medium = ambient * EXPOSURE_GAINS["medium"] + flash * FLASH_GAINS["medium"]
    exp_long = ambient * EXPOSURE_GAINS["long"] + flash * FLASH_GAINS["long"]

    img_short = np.clip((exp_short / sensor_max) * 255.0, 0, 255).astype(np.uint8)
    img_medium = np.clip((exp_medium / sensor_max) * 255.0, 0, 255).astype(np.uint8)
    img_long = np.clip((exp_long / sensor_max) * 255.0, 0, 255).astype(np.uint8)

    merge = cv2.createMergeDebevec()
    hdr = merge.process([img_short, img_medium, img_long], EXPOSURE_TIMES)

    tonemap = cv2.createTonemapReinhard(TONEMAP_GAMMA)
    output = tonemap.process(hdr)

    return np.clip(np.nan_to_num(output) * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    base_dir = ROOT / TEST_FOLDER
    flash_dir = base_dir / "flash_sweep"

    amb_day = read_exr(base_dir / "amb_day.exr")
    amb_night = read_exr(base_dir / "amb_night.exr")

    day_sensor_max = float(np.max(amb_day) * SENSOR_MAX_SCALE)
    night_sensor_max = float(np.max(amb_night) * SENSOR_MAX_SCALE)

    zero_day = np.zeros_like(amb_day, dtype=np.float32)
    zero_night = np.zeros_like(amb_night, dtype=np.float32)

    base_day_luma = mean_gray(process_isp(amb_day, zero_day, day_sensor_max))
    base_night_luma = mean_gray(process_isp(amb_night, zero_night, night_sensor_max))

    results_path = base_dir / "results.csv"

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Day_Score", "Night_Score"])

        for flash_path in sorted(flash_dir.glob("*.exr")):
            flash = read_exr(flash_path)

            day_img = process_isp(amb_day, flash, day_sensor_max)
            day_score = 1.0 - (mean_gray(day_img) / base_day_luma)
            cv2.imwrite(
                str(flash_dir / f"day_final_{flash_path.stem}.png"),
                cv2.cvtColor(day_img, cv2.COLOR_RGB2BGR),
            )

            night_img = process_isp(amb_night, flash, night_sensor_max)
            night_score = 1.0 - (mean_gray(night_img) / base_night_luma)
            cv2.imwrite(
                str(flash_dir / f"night_final_{flash_path.stem}.png"),
                cv2.cvtColor(night_img, cv2.COLOR_RGB2BGR),
            )

            writer.writerow([
                flash_path.stem,
                round(day_score, 4),
                round(night_score, 4),
            ])


if __name__ == "__main__":
    main()
