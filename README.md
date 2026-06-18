# ADA 2025 - A Multimodal Dataset for Dry Stress Detection in Sugar Beets

This is the main repository for all code related to the ADA 2025 dataset.

## Contents of the Dataset
A comprehensive multimodal dataset of greenhouse-grown sugar beets. Data was collected automatically between August 4, 2025 and September 22, 2025. The dataset follows the plants from seeding to a predetermined growth stage and contains hyperspectral, thermal, RGB, and depth imagery of the plants during this period, with dry stress induced in 2/3 of the plants on September 3, 2025. Environmental data and metadata are also provided.

The main dataset card with further links to the individual subcollections for each modality can be found at the following link - PLACEHOLDER. 

## Contents of this Repo
In the `analysis` folder, all code used to create the exemplary results in the ADA 2025 conference paper can be found, along with relevant notes for replicating our findings.

### Environment Setup

The `environment/` folder contains files for recreating the Python environment used for all analyses.

**Conda (recommended):**
```
conda env create -f environment/environment.yml
conda activate ada2025
```

**pip:**
```
pip install -r environment/requirements.txt
```

### Utility Scripts

**`utils/unzip_data.py`** — Interactive tool for selectively extracting the staged dataset archives. Run this first to unpack the modality/bed/plant combinations you need before running any analysis.

**`utils/sensor_summary.py`** — Prints a summary table of record counts, sensor counts, sampling frequency, and date range for each sensor parquet file and the PAR logger spreadsheet. Useful for verifying downloaded data files are complete. Run from the repository root:

```
python utils/sensor_summary.py
```

### Analysis Pipeline

All analyses use only the 3 pm (15h) timestamp. Note that all relevant data files must be present in the `data` folder for the analyses to run. See the section "Preparing the `data` Folder" for precise instructions. 

**Step 1 — `analysis/leaf_air_temp/thermal_analysis.py`**
Processes thermal TIF images to compute leaf-to-air temperature differences. Reads images from the staged thermal directory and matches them against the air temperature sensor parquet. Run from the `analysis/leaf_air_temp/` directory (or supply explicit `--output` path) so the JSON lands next to the plotting script.

```
cd analysis/leaf_air_temp
python thermal_analysis.py \
    --data-dir D:\staged\thermal \
    --air-parquet ../../data/air_readings.parquet \
    --output thermal_analysis.json \
    --start-date 2025_09_01 \
    --end-date 2025_09_14
```

**Step 2 (optional) — `analysis/leaf_air_temp/plot_thermal.py`**
Plots the leaf-to-air temperature differential per bed using the JSON produced in Step 1. Run from `analysis/leaf_air_temp/`.

```
cd analysis/leaf_air_temp
python plot_thermal.py --input thermal_analysis.json --output thermal_plot.png
```

**Step 3 — `analysis/combined/run_stats.py`**
Runs the three core statistical analyses (thermal LMM, hyperspectral LMM, cross-modal Spearman correlations) and writes intermediate CSVs to `analysis/combined/processed/`. Reads from `data/`, `analysis/leaf_air_temp/thermal_analysis.json`, and the hyperspectral Excel file.

```
python analysis/combined/run_stats.py
```

**Step 4 — `analysis/combined/make_figures.py`**
Generates the three publication figures from the processed CSVs produced in Step 3. Figures are saved to `analysis/figures/`.

```
python analysis/combined/make_figures.py
```

**Step 5 - Plant Height Estimation Pipeline (`analysis/plant_height/run_plant_height_pipeline.py`)**
Builds RGB+depth point clouds, computes vegetation-filtered depth metrics, generates W1/W2 trajectory plots, and fits a mixed-effects model with random intercept per plant

```
python analysis/plant_height/run_plant_height_pipeline.py --data-root D:/staged
```

Defaults (including pipeline/model date windows and thresholds) are defined in: `analysis/plant_height/config.py`. More information on the height estimation pipeline is provided in the respective [README](analysis/plant_height/README.md).


### Paper Figure Scripts

The `paper/` folder contains two standalone scripts used to produce the figures in the ADA 2025 paper. These are independent of the main analysis pipeline and can be run from the repository root using the same `ada2025` environment.

**`paper/make_results_figure.py`** — Generates the four-panel results figure (plant height, NDVI, SIPI, leaf-to-air temperature) for W1 vs W2 over the Sept 1–14 window. Reads from `analysis/leaf_air_temp/thermal_analysis.json`, the hyperspectral Excel file in `data/`, and a plant height CSV at `analysis/plant_height/processed/depth_results_w1_w2_values.csv`. Outputs `paper/results_multimodal.pdf` and `paper/results_multimodal.png`.

```
python paper/make_results_figure.py
```

**`paper/make_example_figure.py`** — Generates the four-panel modality showcase figure (RGB, Thermal, NDVI, Depth) used as the dataset example image. Reads images directly from the staged dataset at `D:/staged` — adjust the `STAGED` path constant at the top of the script if your data lives elsewhere. Outputs `analysis/figures/example_modalities.png`.

```
python paper/make_example_figure.py
```

## Preparing the `data` Folder

The analysis scripts expect certain files to be present in the `data/` folder before running. **Most of these files are not included in this repository** and must be obtained or generated separately.

**Parquet files** — The sensor and environmental readings (e.g., `air_readings.parquet`) are not bundled with the repository. Download these from the dataset collection using the dataset card link above.

**Hyperspectral Excel file** — The hyperspectral summary data is provided as a `.xlsx` file generated by the code in `analysis/hyperspectral`. The default filename is expected by the analysis scripts — do not rename it.

**Quality check JSON** — This file is included with the repository and requires no additional setup.

Before running any analysis, confirm that your `data/` folder contains all three of the following:

- The parquet file(s) (e.g., `air_readings.parquet`)
- The hyperspectral Excel file (`results_w5_mean_15_sa.xlsx`)
- The quality check JSON file (included with this repo)
