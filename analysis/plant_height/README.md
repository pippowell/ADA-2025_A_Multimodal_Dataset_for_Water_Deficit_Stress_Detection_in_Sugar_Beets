# Plant Height Pipeline (W1/W2)

This folder contains scripts for a full plant-height estimation pipeline using depth+RGB data.

Scope:
- beds `W1` and `W2` only
- 15h scans by default
- fixed frame index `007` by default

Pipeline stages:
1. `step_01_build_pointclouds.py` — aligns depth and RGB and writes PLY files with `exg`
2. `step_02_compute_metrics.py` — computes per-cloud height statistics (`z_p25`, `z_mean`, etc.)
3. `step_03_plot_depth_results.py` — plots W1/W2 trajectories and exports plotted values CSV
4. `step_04_fit_lmm.py` — fits mixed model with random intercept per plant (`(1 | plant_id)`)

Orchestrator:
- `run_plant_height_pipeline.py` runs all stages in sequence.

## Typical run

```bash
python analysis/plant_height/run_plant_height_pipeline.py \
  --data-root D:/staged
```

## Required input structure

The plant-height pipeline expects the modality-first staged layout created by the
dataset unzip utility described in the main repository README (`utils/unzip_data.py`).

After using that unzip workflow, the data root should look like:

```text
data_root/
├── depth/
│   ├── W1_A2/
│   │   └── .../<date>/<hour>/...
│   └── W2_.../
└── rgb/
    ├── W1_A2/
    │   └── .../<date>/<hour>/...
    └── W2_.../
```

Set `--data-root` to this directory (the same root used as output target for
`utils/unzip_data.py`).

## Configuration

Default pipeline/model parameters are defined in:

`analysis/plant_height/config.py`

Update values there if you want to change defaults globally.

## Key outputs

- Point clouds: `analysis/plant_height/processed/ply/*.ply`
- Metrics CSV: `analysis/plant_height/processed/depth_analysis_metrics.csv`
- Plot values CSV: `analysis/plant_height/processed/depth_results_w1_w2_values.csv`
- Plot figure: `analysis/plant_height/figures/depth_results_w1_w2.png`
- Model input: `analysis/plant_height/models/mixedlm_model_input.csv`
- Model summary: `analysis/plant_height/models/mixedlm_height_estimate_summary.txt`
- Coefficients: `analysis/plant_height/models/mixedlm_height_estimate_coefficients.csv`
- Model trajectories: `analysis/plant_height/models/mixedlm_height_estimate_trajectories.png`

## Model definition

- Height transform: `height_estimate = sensor_height - z_p25`
- Raw depth `z` values are measured downward from the sensor. Subtracting from
  `sensor_height` flips this to upward plant height.
- Quantile equivalence: `Q0.75(height) = sensor_height - Q0.25(depth)`, so this
  transform corresponds to a 75th-percentile height estimate.
- Mixed model: `height_estimate ~ day * treatment + (1 | plant_id)`
