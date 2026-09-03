# SPIM 4D Image Processing Pipeline

A Nextflow pipeline for lightsheet (SPIM) microscopy data: modular preprocessing (XY shading + Z intensity correction + isotropic resampling), segmentation with Cellpose, cell tracking with **ultrack**, and merging into 4D stacks you can open in the `ultrack_viewer` GUI.

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
| `preprocessing.enabled` | Modular preprocessing (planar + depth correction + isotropic resampling). All CPU. |
| `roi_cropping.enabled`  | Crop every timepoint to a Fiji `.roi` rectangle |
| `downscaling.enabled`   | XY downscale applied BEFORE segmentation (uses `downscaling.factor`) |
| `raw_export.enabled`    | Export the raw input sliced-isotropic + downscaled for `ultrack_viewer.py --processed` (independent of preprocessing) |
| `segmentation.enabled`  | Cellpose 3D segmentation |
| `tracking.enabled`      | ultrack cell tracking (needs segmentation on) |
| `benchmark.enabled`     | Per-timepoint timing/memory report |

**Path tips:**

- Use **plain spaces** in paths, never backslash-escape them (e.g. `"my data"` not `"my\ data"`).
- Absolute paths are safest: `/scratch-cbe/users/me/data/...`
- Relative paths are resolved against the repo directory (e.g. `./data/`).
- For tracking, make sure `voxel_size` is correct (in micrometres). Try not to use the automatic detection, as the metadata is oftentimes wrong.

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
├── 00c_downscaled/        # only if downscaling.enabled and preprocessing off
├── 01_preprocessed/       # the isotropic, normalised volumes
│   └── *_processed.tif
├── 01b_raw_isotropic/     # only if raw_export.enabled — RAW signal, sliced isotropic + downscaled
│   └── *_raw_iso_Channel*.tif        (use this with --processed in the viewer to overlay tracks on raw)
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

### 1.9 Overlay tracks on the **RAW** signal (`raw_export`)

The preprocessed chain (`01_preprocessed/`) applies shading correction
(planar) + Z intensity correction (depth) + isotropic resampling.
Each step is a small, independent NumPy/SciPy script under `bin/`
(ported from the AIAF-32 modular scripts); see
[Preprocessing details](#preprocessing-details) below.
That's great for segmentation, but when you want to **interpret** a
track — confirming whether a cell actually ingressed, judging whether a
jump is a real movement or a segmentation artefact — you want to see
the tracks on the **raw** signal, not on the processed one.
depth flattening — just the
same `XY cubic rescale + optional isotropic Z resample` used by
`DOWNSCALE_XY`, applied to the RAW preprocessing chain (no shading correction, no CLAHE, no
deconvolution — just the same `XY cubic rescale + optional
isotropic Z resample` used by `DOWNSCALE_XY`, applied to the RAW
input).

```json
{
  "raw_export": {
    "enabled": true,
    "factor": 0.33,
    "isotropic_reslice": true
  }
}
```

| Field | What it does |
| --- | --- |
| `enabled`        | Turns the export on (`true`) / off (`false`). Off by default — you opt in. |
| `factor`         | XY scale factor (<1.0 = downscale). Pick a value appropriate for napari display (typically `0.25`–`0.5` for a multi-GB acquisition). |
| `isotropic_reslice` | Resample Z so the voxel size matches the downscaled XY pixel size. Required for correct registration with the tracks (tracks are emitted in isotropic coordinates). |

Output goes to `01b_raw_isotropic/<name>_raw_iso_Channel*.tif`
(per-timepoint ImageJ TIFFs matching the same voxel geometry as the
preprocessed chain) plus the merged `4D_hyperstack_raw_iso.tif` when
`output.skip_merge = false`.

> The preprocessed chain is **not** affected — it still operates on
> the original-resolution raw input. The raw export is a purely
> additive overlay for visualisation.

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
| Volume  | `--processed`  | `01_preprocessed/<name>_processed.tif` (or `01b_raw_isotropic/4D_hyperstack_raw_iso.tif` if you enabled `raw_export`) |

A typical launch on the preprocessed volume:

```powershell
mamba activate ultrack-viewer

python ultrack_viewer.py `
    --tracks    03_tracking\results\tracks.csv `
    --segments  03_tracking\results\segments.zarr `
    --processed 01_preprocessed\4D_hyperstack_processed.tif `
    --preload `
    --load-downsample 2
```

To overlay the tracks on the **raw** signal instead (so you can see
the unprocessed intensity under the nuclei), swap the `--processed`
path to the raw export:

```powershell
python ultrack_viewer.py `
    --tracks    03_tracking\results\tracks.csv `
    --segments  03_tracking\results\segments.zarr `
    --processed 01b_raw_isotropic\4D_hyperstack_raw_iso.tif `
    --preload `
    --load-downsample 2
```

Both volumes share the same voxel geometry (isotropic + same
downscaling), so the cross-section viewer and the sphere overlay
work identically against either layer.

Flag cheatsheet:

- `--preload` — load the whole `segments.zarr` into RAM and use a GPU-friendly display volume. This makes scrubbing across time **much** smoother. Only use it if the HIVE has enough RAM for your volume.
- `--load_downsample 2` — keep the volume in RAM at half resolution (8× less RAM). **Recommended** for big datasets; the analysis still works at half-res.
- `--downsample 2` — display-only downsample (full-res is still in RAM). Use this if you only want a faster on-screen render.

> Paths with spaces: wrap them in double quotes, e.g. `--tracks "03_tracking\results\my tracks.csv"`.

Once napari opens, use the layer panel to toggle the processed volume, the segmentation labels, and the tracks on/off. The time slider scrubs through timepoints.

---

## Preprocessing details

The preprocessing chain is intentionally **modular**: each correction is its
own small Python script under `bin/`, its own Nextflow process, and its own
SLURM resource profile. This makes every step independently tunable,
testable, and re-runnable with `nextflow run -resume` after a parameter
change.

```
SPLIT_INPUT_FILE (optional, for hyperstack inputs)
       │
       ▼
  CROP_WITH_ROI (optional)
       │
       ▼
PLANAR_CORRECTION ──► DEPTH_CORRECTION ──► ISOTROPIC ──► Cellpose → ultrack
```

| Step | Script | What it does | Default parameters |
| --- | --- | --- | --- |
| Planar (XY) shading | `bin/planar_intensity_correction.py` | Estimates a smooth flat-field from the mean-Z projection and divides every slice by it. | `sigma_xy = 64` |
| Depth (Z) intensity | `bin/depth_intensity_correction.py` | Rescales each Z slice so a robust per-slice statistic is constant along Z, with moving-average smoothing. | `mode = p99`, `smooth_window = 9`, `gain_clip = [0.25, 4.0]` |
| Isotropic resample | `bin/isotropic_resample.py` | Resamples Z so the voxel size matches the smallest XY pixel size. | `target_um = 0.374`, `order = 3` (cubic) |

All three are pure NumPy + SciPy, run on CPU, and chain via Nextflow. The
math is ported from the AIAF-32 modular scripts and adapted to read/write
plain TIFFs with ImageJ metadata (voxel sizes round-trip through every
step, so downstream Cellpose / ultrack / viewer see the corrected geometry
automatically).

To tune any step, edit its block in `config.json`:

```json
{
  "preprocessing": {
    "enabled": true,
    "planar":    { "sigma_xy": 64.0 },
    "depth":     { "mode": "p99", "smooth_window": 9, "gain_min": 0.25, "gain_max": 4.0 },
    "isotropic": { "target_um": 0.374, "order": 3 }
  }
}
```

To disable preprocessing entirely (e.g. you already have preprocessed
TIFFs from another tool) set `preprocessing.enabled = false` and point
`preprocessing.preprocessed_dir` at the folder containing the timepoint
TIFFs.

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
