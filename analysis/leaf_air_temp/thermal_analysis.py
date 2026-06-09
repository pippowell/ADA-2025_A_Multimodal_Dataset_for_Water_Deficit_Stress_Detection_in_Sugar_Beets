"""
thermal_analysis.py

Computes leaf temperature from thermal TIF images and compares
against air temperature sensor readings for the same time window.

For each valid plant/date/hour:
  - Reads up to 5 thermal TIFs (validated against a provided dataset quality check JSON record, if provided and if desired)
  - Takes the coolest 50% of pixels per TIF (largely the plants themselves) and averages them
  - Averages that across all valid TIFs to get one average leaf temperature reading
  - Pulls the median air temperature from the 10-minute window matching the bed (the robot takes about 10 minutes to scan each bed):
      W1 (bed 1, control) → first 10 min past the hour
      W2 (bed 2, water deficit) → second 10 min past the hour
      W3 (bed 3, water deficit) → third 10 min past the hour
  - Records diff = air_temp - leaf_temp for each sensor

Output is a JSON file keyed by plant ID, each containing a list of readings.

Developed with assistance from Claude (Anthropic) via Claude Code.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


COOL_FRACTION = 0.50  # use the coolest this fraction of pixels

_DEFAULT_QC_JSON = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "06_26_quality_check.json"
)


def bed_number(plant_id: str) -> int:
    """Extract bed number from plant ID, e.g. 'W2_R7' -> 2."""
    return int(plant_id[1])


def air_window_minutes(bed: int) -> tuple[int, int]:
    """Return (start_minute, end_minute) past the hour for the given bed."""
    start = (bed - 1) * 10
    return start, start + 10


def leaf_temp_from_tif(path: Path) -> float:
    """Return median temperature of the coolest COOL_FRACTION pixels in a TIF."""
    data = tifffile.imread(path).astype(np.float32).ravel()
    threshold = np.percentile(data, COOL_FRACTION * 100)
    return float(np.median(data[data <= threshold]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute leaf-to-air temperature differences from thermal TIFs."
    )
    parser.add_argument(
        "--data-dir", default=r"D:\staged\thermal",
        help="Root directory containing per-plant thermal folders.",
    )
    parser.add_argument(
        "--quality-control-json", default=_DEFAULT_QC_JSON,
        help="Path to dataset quality check JSON report.",
    )
    parser.add_argument(
        "--air-parquet", default="data/air_readings.parquet",
        help="Path to air_readings.parquet.",
    )
    parser.add_argument(
        "--output", default="thermal_analysis.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--start-date", default="2025_09_01",
        help="Earliest date folder to include (inclusive), format YYYY_MM_DD.",
    )
    parser.add_argument(
        "--end-date", default="2025_09_14",
        help="Latest date folder to include (inclusive), format YYYY_MM_DD.",
    )
    parser.add_argument(
        "--ignore-review", action="store_true",
        help="Skip dataset quality check JSON validation and use all TIFs found on disk.",
    )
    args = parser.parse_args()

    print("Loading dataset quality check JSON...")
    with open(args.quality_control_json, encoding="utf-8") as f:
        progress = json.load(f)

    if args.ignore_review:
        print("NOTE: --ignore-review set, skipping validation filter.")

    print("Loading air readings parquet...")
    air_df = pd.read_parquet(args.air_parquet)

    data_dir = Path(args.data_dir)
    results: dict[str, list[dict]] = {}
    total_readings = 0

    plant_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    print(f"Found {len(plant_dirs)} plant directories.\n")

    for plant_dir in plant_dirs:
        plant_id = plant_dir.name

        if not args.ignore_review and plant_id not in progress:
            print(f"  [{plant_id}] not in progress JSON — skipping")
            continue

        # Unzip extraction often produces a nested folder with the same name
        inner = plant_dir / plant_id
        scan_root = inner if inner.is_dir() else plant_dir

        bed = bed_number(plant_id)
        win_start_min, win_end_min = air_window_minutes(bed)
        plant_results: list[dict] = []

        date_dirs = sorted(d for d in scan_root.iterdir() if d.is_dir())
        for date_dir in date_dirs:
            date_str = date_dir.name

            if date_str < args.start_date or date_str > args.end_date:
                continue

            if not args.ignore_review and date_str not in progress.get(plant_id, {}):
                continue

            for hour_dir in sorted(d for d in date_dir.iterdir() if d.is_dir()):
                hour_str = hour_dir.name

                if int(hour_str) != 15:
                    continue

                if not args.ignore_review and hour_str not in progress.get(plant_id, {}).get(date_str, {}):
                    continue

                images_entry = progress.get(plant_id, {}).get(date_str, {}).get(hour_str, {}).get("images", {})

                # Collect valid TIF paths — all on disk when ignoring review,
                # otherwise only those that passed the JSON validation check
                valid_tifs: list[Path] = []
                for i in range(5):
                    key = f"thermal_{i:03d}"
                    tif_path = hour_dir / f"{key}.tif"
                    if not tif_path.exists():
                        continue
                    if not args.ignore_review:
                        rec = images_entry.get(key)
                        if rec is None or rec.get("error"):
                            continue
                    valid_tifs.append(tif_path)

                if not valid_tifs:
                    continue

                # Per-TIF: median of coolest COOL_FRACTION pixels
                tif_temps = [leaf_temp_from_tif(p) for p in valid_tifs]
                leaf_temp = float(np.median(tif_temps))

                # Locate the matching 10-minute air temperature window in Berlin time
                date_iso = date_str.replace("_", "-")
                hour_start = pd.Timestamp(
                    f"{date_iso} {hour_str}:00:00", tz="Europe/Berlin"
                )
                win_start = hour_start + pd.Timedelta(minutes=win_start_min)
                win_end   = hour_start + pd.Timedelta(minutes=win_end_min)

                window = air_df[
                    (air_df["berlin_timestamp"] >= win_start) &
                    (air_df["berlin_timestamp"] <  win_end)
                ]

                def sensor_median(sensor_num: str) -> float | None:
                    val = window[window["sensor_number"] == sensor_num]["air_temperature_value"].median()
                    return None if pd.isna(val) else round(float(val), 4)

                s1 = sensor_median("1")
                s2 = sensor_median("2")

                reading: dict = {
                    "date": date_str,
                    "hour": hour_str,
                    "leaf_temp_c": round(leaf_temp, 4),
                    "valid_tif_count": len(valid_tifs),
                    "air_temp_sensor1_c": s1,
                    "air_temp_sensor2_c": s2,
                    "diff_sensor1": round(s1 - leaf_temp, 4) if s1 is not None else None,
                    "diff_sensor2": round(s2 - leaf_temp, 4) if s2 is not None else None,
                }
                plant_results.append(reading)

        if plant_results:
            results[plant_id] = plant_results
            total_readings += len(plant_results)
            print(f"  [{plant_id}]  {len(plant_results)} readings")
        else:
            print(f"  [{plant_id}]  no valid readings in date range")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {total_readings} readings across {len(results)} plants -> {args.output}")


if __name__ == "__main__":
    main()
