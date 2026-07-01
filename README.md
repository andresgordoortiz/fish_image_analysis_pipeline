# SPIM 4D Image Processing Pipeline

A Nextflow pipeline for lightsheet (SPIM) microscopy data: deconvolution, segmentation with Cellpose, cell tracking with **ultrack**, and merging into 4D stacks you can open in the `ultrack_viewer` GUI.

**Authors:** Andrés Gordo & Guilherme Ventura · **Institute:** IMP Vienna

This README is the step-by-step guide. There are two parts:

1. **Run the pipeline on the HPC** (SLURM cluster, all the heavy work)
2. **Visualise the results with the ultrack_viewer** (on the HIVE workstation)

---

## Part 1 · Run the pipeline on the HPC

### 1.1 Get access to the cluster

1. Ask IT for a **CLIP HPC cluster** account. They can check internal docs at
   [biocenterat.sharepoint.com/.../HPC-Documentation](https://biocenterat.sharepoint.com/sites/Portal/SitePages/Infrastructure%20Services/Information%20Technology/HPC-Documentation.aspx).
2. From your laptop, SSH into the login node:

   ```bash
   ssh cbe.vbc.ac.at
   # enter your credentials
   ```

3. **Always work inside `/scratch-cbe/users/<your-user>/`**. Anything on the main server fills it up and slows everyone down. Create your folder if it does not exist:

   ```bash
   mkdir -p /scratch-cbe/users/$USER
   cd       /scratch-cbe/users/$USER
   ```

### 1.2 Clone the repo

```bash
git clone https://github.com/andresgordoortiz/spim_preprocessing.git
cd spim_preprocessing
```

### 1.3 Prepare your data

The pipeline accepts **either** of these two inputs (set `input.directory` in `config.json`):

- A **folder of per-timepoint TIFFs** named `t0001_*.tif`, `t0002_*.tif`, ...
- A **single hyperstack file** (`.czi`, `.tif`, `.tiff` with multiple Z/T)

If your data is large, keep the original copy somewhere safe and copy/symlink just the timepoint TIFFs into your scratch folder to save space.

### 1.4 Edit `config.json`

Open `config.json` in your favourite editor (`nano`, `vim`, ...). The minimum fields you need to set are:

```json
{
  "input":  { "directory": "/scratch-cbe/users/me/data/my_experiment/" },
  "output": { "directory": "/scratch-cbe/users/me/results/my_experiment/" },

  "seqera_tower": {
    "enabled": true,
    "access_token": "paste-your-token-here"
  }
}
```

Toggles to control what runs (set to `true` / `false`):

| Section | What it does |
| --- | --- |
| `preprocessing.enabled` | Modular preprocessing pipeline (shading + Z correction + deconv + …) |
| `preprocessing.method` | `"deconvolution"` (GPU Richardson-Lucy) or `"selfnet"` |
| `roi_cropping.enabled`  | Crop every timepoint to a Fiji `.roi` rectangle |
| `segmentation.enabled`  | Cellpose 3D segmentation |
| `tracking.enabled`      | ultrack cell tracking (needs segmentation on) |
| `benchmark.enabled`     | Per-timepoint timing/memory report |

#### Preprocessing sub-stages (all individually toggleable)

Each step in the modular preprocessing pipeline has its own `enabled` flag and its own parameters under `preprocessing`:

```json
"preprocessing": {
  "enabled": true,
  "method": "deconvolution",

  "downscale_xy":           { "factor": 0.33 },
  "shading_correction":     { "enabled": true, "sigma_xy": 96.0, "max_amplification": 2.0 },
  "z_intensity_correction": { "enabled": true, "method": "p95", "smooth_window": 11 },
  "isotropic_reslice":      { "enabled": true },
  "deconvolution_3d":       { "enabled": true, "psf_path": "/path/psf.tif", "niter": 3, "padding": 64 },
  "deconvolution_xz":       { "enabled": true, "niter": 3, "padding": 64 },
  "background_subtraction": { "enabled": true, "resolution_xy": 10, "resolution_xz": 10, "noise_lvl": 2 },
  "gaussian_smooth":        { "enabled": true, "sigma": 1.2 },
  "clahe":                  { "enabled": true, "clip_limit": 0.01, "kernel_size": 64, "p_low": 0.5, "p_high": 99.5 },
  "percentile_normalization": { "enabled": true, "percentile_low": 10.0, "percentile_high": 99.9 },
  "final_cast":             { "min_v": 0, "max_v": 65535, "dtype": "uint16" },

  "save_intermediates": true
}
```

To turn off a stage (e.g. CLAHE for a quick test), set `preprocessing.clahe.enabled: false`. The pipeline skips that stage entirely and skips its intermediate save.

#### `--simulate` mode — parameter sweep + Cellpose benchmark

For tuning preprocessing parameters on a single image without spinning up the full Nextflow pipeline:

```bash
# 1. Write a sweep file describing the experiments to try
cat > sweep.json <<'EOF'
{
  "base_config": "./config.json",
  "input": ["./subset/t0001.tif"],
  "output_dir": "./simulation_results/",
  "save_intermediates": false,
  "experiments": [
    { "name": "no_clahe",  "overrides": { "clahe":  { "enabled": false } } },
    { "name": "clahe_005", "overrides": { "clahe":  { "clip_limit": 0.005 } } },
    { "name": "clahe_020", "overrides": { "clahe":  { "clip_limit": 0.020 } } },
    { "name": "no_smooth", "overrides": { "gaussian_smooth": { "enabled": false } } },
    { "name": "tight_norm","overrides": { "percentile_normalization": { "percentile_low": 20, "percentile_high": 99.5 } } }
  ]
}
EOF

# 2. Run the sweep — each experiment is preprocessed, segmented with Cellpose,
#    and nuclei are counted.
python3 spim_pipeline_fixed.py --simulate --sweep_file sweep.json

# 3. Compare results
cat simulation_results/summary.md
```

Per experiment you get:

```
simulation_results/<exp_name>/
├── t0001_33.tif            # preprocessed image (uint16)
├── t0001_33_mask.tif       # Cellpose segmentation mask (uint16)
└── t0001_33_metadata.json  # config used + per-stage timings + intensity stats
```

Plus a `summary.csv` and `summary.md` ranking experiments by runtime, mean intensity, and **detected nuclei count** (from Cellpose). Open each `_33.tif` in napari/Fiji for visual inspection.

**Path tips:**

- Use **plain spaces** in paths, never backslash-escape them (e.g. `"my data"` not `"my\ data"`).
- Absolute paths are safest: `/scratch-cbe/users/me/data/...`
- Relative paths are resolved against the repo directory (e.g. `./data/`).
- For tracking, make sure `voxel_size` is correct (in micrometres). Try not to use the automatic detection, as the metadata is oftentimes wrong.

**Picking which timepoints to process.** Three fields under `input` control this; they compose (later filters apply on top of earlier ones):

| Field | Behaviour |
| --- | --- |
| `input.timepoints` | Process ONLY these timepoints. Each entry can be a 1-based numeric index (`5`), the same as a string (`"5"`), or a filename stem (`"t0005"`) or full filename (`"t0005.tif"`). `null` = no filter. |
| `input.max_timepoints` | After the `timepoints` filter, keep only the first N. `null` = no limit. |

Examples:

```json
"input": {
  "directory": "/scratch-cbe/users/me/data/my_experiment/",

  "_comment_timepoints": "Just t0001 and t0005 (e.g. two timepoints to debug).",
  "timepoints": [1, 5],

  "_comment_max_timepoints": "Limit to first 10 timepoints from the filtered list.",
  "max_timepoints": 10
}
```

```json
"input": {
  "directory": "/scratch-cbe/users/me/data/my_experiment/",
  "timepoints": ["t0001_Channel 2.tif", "t0005_Channel 2.tif", "t0010_Channel 2.tif"],
  "max_timepoints": null
}
```

If a requested timepoint isn't in the input directory, the run fails fast with a list of available timepoints.

**Seqera token (optional but recommended).** Create a free account at [tower.nf](https://tower.nf), go to **Settings → Your tokens**, generate a token, and paste it into `seqera_tower.access_token`. This lets you watch the run live in the browser under **Runs**.

### 1.5 Submit the pipeline

```bash
sbatch submit_pipeline.sh config.json
```

That is it. The script:

- Loads `nextflow`, `java`, `build-env`
- Points Singularity/Apptainer at the pre-cached container (`/groups/pinheiro/user/andres.gordo/containers_licences/`)
- Exports your Seqera token if you set one
- Runs `nextflow run ./spim_pipeline.nf --config_json config.json`

If `config.json` or your input path contains spaces, just keep them as plain spaces — `submit_pipeline.sh` handles sanitisation.

### 1.6 Monitor the run

| Where | What to look at |
| --- | --- |
| `squeue --me`                     | Whether your job is queued / running |
| `tower.nf` → **Runs**             | Live progress, per-task logs, timeline |
| `<output_dir>/pipeline_<date>.log` | Full Nextflow log (with `tee`) |
| `<output_dir>/reports/`           | HTML report, timeline, trace |

If something fails, the pipeline auto-retries up to 3 times (see `nextflow.config`). To restart from where it stopped, just re-run:

```bash
sbatch submit_pipeline.sh config.json
```

Set `system.resume = false` in `config.json` if you want a clean re-run.

### 1.7 Output structure

After the run completes, your `output.directory` will look like:

```
my_experiment/
├── 00_split_input/        # per-timepoint TIFFs (if you gave a hyperstack)
├── 00_cropped/            # only if roi_cropping.enabled
├── 00b_isotropic/         # only if preprocessing.isotropic_reslice and preprocessing off
├── 01_preprocessed/       # the isotropic, normalised volumes
│   └── *_processed.tif
├── 02_segmented/          # Cellpose label volumes
├── 02_segmented_downscaled/   # only if segmentation.downscale_labels < 1
├── 03_tracking/           # only if tracking.enabled
│   └── results/
│       ├── tracks.csv
│       └── segments.zarr
├── benchmark/             # only if benchmark.enabled
├── metadata/              # voxel size + image metadata
├── reports/               # Nextflow HTML / trace / timeline
├── logs/                  # per-step logs
└── pipeline_<date>.log    # the full Nextflow log
```

### 1.8 Move the results to the main server (when done)

Once you are happy with the run, copy the output out of scratch and into the main server storage so the team can access it:

```bash
rsync -avh --progress \
  /scratch-cbe/users/$USER/results/my_experiment/ \
  /groups/pinheiro/user/$USER/my_experiment/
```

`/groups/pinheiro/user/` is the main server; it is backed up and shared. **Don't run pipeline jobs from there** — that is what scratch is for. Only move finished results back.

---

## Part 2 · Visualise the results with `ultrack_viewer` (on the HIVE)

`ultrack_viewer.py` is a napari-based GUI for browsing the preprocessed volume, segmentation labels, and tracks side-by-side.

### 2.1 Open a PowerShell session on the HIVE

1. Remote-desktop / connect to the **HIVE** workstation (the Windows-based one).
2. Open **PowerShell**.

### 2.2 Create the conda environment (once)

The viewer needs a small conda env with napari, dask, zarr, etc. The env file is shipped in the repo:

```powershell
# go to the repo you cloned earlier (or copy the yml to the HIVE)
cd path\to\spim_preprocessing

# create the env with mamba (faster than conda)
mamba env create -f ultrack_viewer_env.yml

# activate it
mamba activate ultrack-viewer
```

### 2.3 Go to your results folder

- If the results are on the **HIVE local disk**: just `cd` into the output folder.
- If the results are still on the **main server (`/groups/pinheiro/...`)**, the HIVE sees it as the **`V:`** drive. Mount it first:

  ```powershell
  V:
  cd V:\path\to\my_experiment
  ```

### 2.4 Launch the viewer

The viewer accepts three layers that are all optional but work best together:

| Layer   | Flag           | Typical file |
| ------- | -------------- | ------------ |
| Tracks  | `--tracks`     | `03_tracking/results/tracks.csv` |
| Labels  | `--segments`   | `03_tracking/results/segments.zarr` |
| Volume  | `--processed`  | `01_preprocessed/<name>_processed.tif` |

A typical launch:

```powershell
mamba activate ultrack-viewer

python ultrack_viewer.py `
    --tracks    03_tracking\results\tracks.csv `
    --segments  03_tracking\results\segments.zarr `
    --processed 01_preprocessed\4D_hyperstack_processed.tif `
    --preload `
    --load-downsample 2
```

Flag cheatsheet:

- `--preload` — load the whole `segments.zarr` into RAM and use a GPU-friendly display volume. This makes scrubbing across time **much** smoother. Only use it if the HIVE has enough RAM for your volume.
- `--load_downsample 2` — keep the volume in RAM at half resolution (8× less RAM). **Recommended** for big datasets; the analysis still works at half-res.
- `--downsample 2` — display-only downsample (full-res is still in RAM). Use this if you only want a faster on-screen render.

> Paths with spaces: wrap them in double quotes, e.g. `--tracks "03_tracking\results\my tracks.csv"`.

Once napari opens, use the layer panel to toggle the processed volume, the segmentation labels, and the tracks on/off. The time slider scrubs through timepoints.

---

## Troubleshooting

**`sbatch` job stays in `PD` forever.** The cluster is busy. `squeue --me` will show the reason. If you need GPU, queue `g` is the bottleneck.

**Pipeline fails immediately with "Input path does not exist".** Check the path in `config.json`. Use `ls` from the login node to confirm. Remember: no backslash-escaped spaces in JSON strings.

**Container not found error.** The pre-pulled container lives at `/groups/pinheiro/user/andres.gordo/containers_licences/`. If it is missing, ask Andrés to repopulate it, or run `./setup_container.sh` from the login node (needs internet, ~30 min).

**Tracking fails but everything else works.** Make sure the Gurobi license is in `/groups/pinheiro/user/andres.gordo/containers_licences/gurobi.lic` and that `segmentation.enabled = true` (tracking consumes the segmentation labels).

**Viewer is sluggish / crashes on load.** Drop `--preload`, set `--load_downsample 4`, or set `--downsample 4` for a low-res display.

**Want to re-use finished tasks.** `system.resume = true` (default) makes Nextflow skip already-done work. Just `sbatch submit_pipeline.sh config.json` again.

---

## Contact

Andrés Gordo — andres.ortiz@imp.ac.at
