"""
Interactive unzipper for ADA-2025 staged dataset.
Extracts selected modality/plant zip files from provided data directory into
{data_dir}/{modality}/{plant}/ directories.

Developed with assistance from Claude (Anthropic) via Claude Code.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is required: pip install tqdm")
    sys.exit(1)

MODALITIES = ["thermal", "hsi", "rgb", "depth"]
BEDS = ["W1", "W2", "W3"]


# ---------------------------------------------------------------------------
# Discovery & validation
# ---------------------------------------------------------------------------

def discover_plants(staged_dir: Path) -> list[str]:
    """Scan zip filenames to find all plant IDs present in the data folder."""
    bed_prefixes = tuple(f"{bed}_" for bed in BEDS)
    plant_ids: set[str] = set()
    for mod in MODALITIES:
        mod_dir = staged_dir / mod
        if not mod_dir.is_dir():
            continue
        for zf in mod_dir.glob("*.zip"):
            for prefix in bed_prefixes:
                if zf.stem.startswith(prefix):
                    plant_ids.add(zf.stem[len(prefix):])
                    break
    return sorted(plant_ids)


def check_files(staged_dir: Path, plants: list[str]) -> list[str]:
    """
    Return paths of expected zip files that are missing.
    Note - this only checks that all plant IDs found during discovery have data in all four modalities, not ex. whether a certain plant ID is missing
    """
    missing = []
    for mod in MODALITIES:
        for bed in BEDS:
            for plant in plants:
                p = staged_dir / mod / f"{bed}_{plant}.zip"
                if not p.exists():
                    missing.append(str(p))
    return missing


# ---------------------------------------------------------------------------
# Interactive multi-select UI
# ---------------------------------------------------------------------------

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def multiselect(title: str, options: list[str]) -> list[str]:
    """
    Numpad multi-select: press numbers to toggle, 0 = select all, Enter = confirm.
    """
    selected: set[int] = set()
    n = len(options)
    col_w = max(len(o) for o in options) + 10
    cols = max(1, min(3, 80 // col_w))
    message = ""

    while True:
        _clear()
        print(f"\n  {title}")
        print("  " + "=" * 62)

        for i, opt in enumerate(options):
            num = i + 1
            mark = "x" if num in selected else " "
            cell = f"[{mark}] {num}. {opt}"
            end = "\n" if (i + 1) % cols == 0 or (i + 1) == n else ""
            print(f"  {cell:<{col_w}}", end=end)

        if n % cols != 0:
            print()

        print()
        if message:
            print(f"  {message}")
        print(f"  Selected: {len(selected)}/{n}  |  0 = all, 1-{n} = toggle, Enter = confirm")
        inp = input("  > ").strip()
        message = ""

        if inp == "":
            if not selected:
                message = "Select at least one option."
            else:
                break
        else:
            for token in inp.replace(",", " ").split():
                try:
                    num = int(token)
                    if num == 0:
                        selected = set(range(1, n + 1))
                        break
                    elif 1 <= num <= n:
                        selected.discard(num) if num in selected else selected.add(num)
                    else:
                        message = f"  {num} is out of range (1-{n})."
                except ValueError:
                    message = f"  '{token}' is not a valid number."

    return [options[i - 1] for i in sorted(selected)]


def select_plants(plants: list[str]) -> list[str]:
    """
    Ask whether to unzip all plants or specific ones by ID (e.g. A2, R7).
    """
    while True:
        _clear()
        print(f"\n  Step 3 of 3 — Select plants")
        print("  " + "=" * 62)
        print(f"  {len(plants)} plant ID(s) found: {', '.join(plants)}")
        print()
        print("  Enter to unzip all, or type plant IDs separated by commas:")
        inp = input("  > ").strip().upper()

        if inp == "":
            return list(plants)

        tokens = [t.strip() for t in inp.replace(",", " ").split() if t.strip()]
        valid = [t for t in tokens if t in plants]
        invalid = [t for t in tokens if t not in plants]

        if invalid:
            print(f"  Unknown ID(s): {', '.join(invalid)}  —  valid: {', '.join(plants)}")
            input("  Press Enter to try again...")
            continue

        if not valid:
            input("  No valid plants entered. Press Enter to try again...")
            continue

        return valid


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_zip(zip_path: Path, extract_dir: Path, label: str) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        with tqdm(
            total=len(members),
            desc=f"  {label}",
            unit="file",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            for member in members:
                zf.extract(member, extract_dir)
                pbar.update(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ADA-2025 dataset unzipper")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=r"D:\staged",
        help="Path to the staged data folder (default: D:\\staged)",
    )
    args = parser.parse_args()
    staged_dir = Path(args.data_dir)

    _clear()
    print("\n  ADA-2025 Dataset Unzipper")
    print("  " + "=" * 62)
    print(f"\n  Data folder: {staged_dir}")

    if not staged_dir.is_dir():
        print(f"\n  [ERROR] Directory not found: {staged_dir}")
        sys.exit(1)

    # --- Discover plants -------------------------------------------------
    plants = discover_plants(staged_dir)
    if not plants:
        print("\n  [ERROR] No plant zip files found in the data folder.")
        sys.exit(1)

    # --- Validate --------------------------------------------------------
    expected = len(MODALITIES) * len(BEDS) * len(plants)
    print(f"\n  Checking {expected} expected zip files ({len(plants)} plant(s) discovered) ...")
    missing = check_files(staged_dir, plants)

    if missing:
        print(f"\n  [WARNING] {len(missing)} file(s) missing:")
        for f in missing:
            print(f"    - {f}")
        ans = input("\n  Continue with available files? [y/N]: ").strip().lower()
        if ans != "y":
            sys.exit(0)
    else:
        print(f"  All {expected} zip files present.\n")
        input("  Press Enter to continue to selection ...")

    # --- Select modalities -----------------------------------------------
    sel_modalities = multiselect("Step 1 of 3 — Select modalities to unzip:", MODALITIES)

    # --- Select beds -----------------------------------------------------
    sel_beds = multiselect("Step 2 of 3 — Select beds to unzip:", BEDS)

    # --- Select plants ---------------------------------------------------
    sel_plants = select_plants(plants)

    # --- Confirmation ----------------------------------------------------
    _clear()
    total = len(sel_modalities) * len(sel_beds) * len(sel_plants)
    print(f"\n  Ready to unzip {total} archive(s).")
    print(f"  Modalities : {', '.join(sel_modalities)}")
    print(f"  Beds       : {', '.join(sel_beds)}")
    print(f"  Plants     : {', '.join(sel_plants)}")
    print(f"  Output     : {staged_dir}\\<modality>\\<bed>_<plant>\\")
    print()
    ans = input("  Proceed? [y/N]: ").strip().lower()
    if ans != "y":
        print("  Aborted.")
        sys.exit(0)

    # --- Extract ---------------------------------------------------------
    print()
    errors: list[tuple[str, str]] = []

    for mod in sel_modalities:
        print(f"\n  [{mod}]")
        for bed in sel_beds:
            for plant in sel_plants:
                label = f"{bed}_{plant}"
                zip_path = staged_dir / mod / f"{label}.zip"
                extract_dir = staged_dir / mod / label
                if not zip_path.exists():
                    print(f"    [SKIP] {zip_path.name} — not found")
                    continue
                try:
                    extract_zip(zip_path, extract_dir, f"{mod}/{label}")
                except Exception as exc:
                    errors.append((str(zip_path), str(exc)))
                    print(f"    [ERROR] {zip_path.name}: {exc}")

    # --- Summary ---------------------------------------------------------
    print("\n  " + "=" * 62)
    if errors:
        print(f"  Finished with {len(errors)} error(s):")
        for path, err in errors:
            print(f"    {path}: {err}")
    else:
        print(f"  Done — {total - len(errors)} archive(s) extracted successfully.")
    print()


if __name__ == "__main__":
    main()
