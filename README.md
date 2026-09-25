# SPIM 4D Image Processing Pipeline

A Nextflow pipeline for lightsheet (SPIM) microscopy: modular preprocessing (XY shading + Z intensity + isotropic resampling), Cellpose segmentation, **ultrack** cell tracking, and 4D stacks that load into `ultrack_viewer`.

**Authors:** Andrés Gordo & Guilherme Ventura · **Institute:** IMP Vienna

This README has two parts:

1. Run the pipeline on the SLURM **HPC**.
2. Visualise results with `ultrack_viewer` on the **HIVE** workstation.

---

## Quickstart

```bash
# 1. Login: ssh cbe.vbc.ac.at  →  mkdir -p /scratch-cbe/users/$USER && cd /scratch-cbe/users/$USER
git clone https://github.com/andresgordoortiz/spim_preprocessing.git
cd spim_preprocessing

# 2. Edit config.json — at minimum set input.directory, output.directory.
#    Containers, Gurobi licence, Seqera token: see § 1.5, § 1.6, § 1.4.

# 3. Submit
sbatch submit_pipeline.sh config.json
```

Then monitor in your browser at **tower.nf → Runs** (after pasting a free token into `seqera_tower.access_token`).

---

## Part 1 · Run on the HPC

### 1.1 Cluster access

```bash
ssh cbe.vbc.ac.at
mkdir -p /scratch-cbe/users/$USER && cd /scratch-cbe/users/$USER   # always work in scratch
```

### 1.2 Prepare data

| Input | Example filenames | `input.directory` |
| --- | --- | --- |
| Folder of per-timepoint TIFFs | `t0000_Channel 1.tif`, `t0001_*.tif`, … | the folder |
| Zeiss hyperstack | `movie.czi` | the `.czi` or its parent folder |
| 4D/5D OME-TIFF / ImageJ hyperstack | `full_stack.tif` | the `.tif` itself |
| Imaris or 5D HDF5 | `dataset.ims` / `DataSet.h5` | the file itself |
| Folder of per-timepoint / per-channel HDF5 (Bio-Formats split) | `prefix--C00--T00000.h5`, `prefix--C00--T00001.h5`, … | the folder; pick channel with `input.channel` |

A `.ims` / single `.h5` is split into per-timepoint TIFFs by `SPLIT_INPUT_FILE`; voxel sizes are auto-detected from `PhysicalSize{X,Y,Z}` metadata. A folder of `--C##--T#####.h5` files bypasses `SPLIT_INPUT_FILE` entirely — each `.h5` already maps to one `(timepoint, channel)` pair. Channels are 1-indexed in `config.json` (`channel: 1` → `C00`).

### 1.3 Edit `config.json`

Minimum fields:

```json
{
  "input":  { "directory": "/scratch-cbe/users/me/data/" },
  "output": { "directory": "/scratch-cbe/users/me/results/" },
  "seqera_tower": { "enabled": true, "access_token": "paste-your-token" }
}
```

Toggles (`true` / `false`):

| Section | What it runs |
| --- | --- |
| `preprocessing.enabled` | Planar + depth + isotropic resampling (CPU) |
| `roi_cropping.enabled`  | Crop each timepoint to a Fiji `.roi` |
| `downscaling.enabled`   | XY downscale before segmentation (`downscaling.factor`) |
| `raw_export.enabled`    | Export raw input sliced-isotropic + downscaled for viewer overlay (`raw_export`) |
| `segmentation.enabled`  | Cellpose 3D |
| `tracking.enabled`      | ultrack (needs `segmentation.enabled = true`) |
| `benchmark.enabled`     | Per-timepoint timing/memory report |

**Path tips:** plain spaces (no `\"` escapes), absolute paths are safest, relative paths resolve against the repo dir. For tracking, set `voxel_size.{x,y,z}_um` explicitly (auto-detect is unreliable on some file formats).

**Seqera token:** free account at [tower.nf](https://tower.nf) → **Settings → Your tokens**.

### 1.4 Container images

Every process runs inside Apptainer. The pipeline needs three pre-pulled images on a shared filesystem (compute nodes have no internet):

| Container | Used by | Default |
| --- | --- | --- |
| `system.container_image` | All main processes | `library://andresgordoortiz/spim_imp/python_packages_spim:sha256.6ef173bb…` |
| `system.fiji_container_image` | `CROP_WITH_ROI` | `docker://fiji/fiji:20220415` |
| `system.ultrack_container` | `PREP_ULTRACK` + `ULTRACK_*` | `docker://qbiotumber/ultrack:latest` |

The defaults point at the original maintainer's shared folder (`/groups/pinheiro/user/andres.gordo/containers_licences/`), which becomes inaccessible when they leave. Pre-pull your own copies and override:

```bash
mkdir -p /groups/<your-area>/<your-user>/containers && cd $_
apptainer pull --name spim_pipeline.sif library://andresgordoortiz/spim_imp/python_packages_spim:sha256.6ef173bb45b113a36deae4315200cd8f311de2d7108b4b73e8f17a12cffe7559
apptainer pull --name fiji.sif   docker://fiji/fiji:20220415                    # only if CROP_WITH_ROI
apptainer pull --name ultrack.sif docker://qbiotumber/ultrack:latest            # only if tracking
```

Then set the absolute paths in `config.json` under `system.{container_image,fiji_container_image,ultrack_container}`. `submit_pipeline.sh` verifies each file exists before launching.

Override order (highest priority first): `config.json` `system.*` key → `$SPIM_{PIPELINE,FIJI,ULTRACK}_CONTAINER` env var → hardcoded fallback in `nextflow.config`.

> **Fiji note.** The default Fiji URI is `docker://`; on air-gapped clusters pre-pull the `.sif` and override with the absolute path.

### 1.5 Gurobi licence (required for tracking)

`ULTRACK_SOLVE` uses Gurobi; without a valid licence it fails and only tracking is skipped.

**Get a free WLS Academic licence** (renews every 90 days):

1. Create a Gurobi account with your **institutional email** (academic only — gmail/outlook/yahoo are rejected).
2. Sign in → **Licenses → Request** → **ACADEMIC → WLS → GENERATE NOW!**.
3. Download `gurobi.lic`. Set a calendar reminder to renew before expiry.

**Install on the cluster:**

```bash
mkdir -p /groups/<your-area>/<your-user>/gurobi
cp ~/Downloads/gurobi.lic $_/
chmod 644 $_/gurobi.lic
```

**Point the pipeline at it** — set `system.gurobi_license_path` in `config.json`, or export `$GUROBI_LICENSE_PATH`, or fall back to the path in `nextflow.config`. The path must be absolute and readable from compute nodes (`/groups/`, `/scratch-cbe/`, … work out of the box).

If `submit_pipeline.sh` can't find the file it prints a `WARNING` and continues — segmentation + 4D merging still run, but `ULTRACK_SOLVE` will fail.

### 1.6 Submit

```bash
sbatch submit_pipeline.sh config.json
```

The script loads `nextflow` + `java`, points Apptainer at the pre-cached images, exports your Seqera token, and runs `nextflow run ./spim_pipeline.nf --config_json config.json`. Paths with spaces are handled automatically.

### 1.7 Monitor

| Where | What to look at |
| --- | --- |
| `squeue --me`              | Queue / run status |
| `tower.nf` → **Runs**      | Live per-task progress, logs, timeline |
| `<output_dir>/pipeline_<date>.log` | Full Nextflow log |
| `<output_dir>/reports/`    | HTML report, timeline, trace |

The pipeline auto-retries failed tasks up to 3 times. To restart from where it stopped, just re-run `sbatch submit_pipeline.sh config.json` (resume is on by default). For a clean re-run, set `system.resume = false`.

### 1.8 Output

```
my_experiment/
├── 00_split_input/        # per-timepoint TIFFs (only for hyperstack inputs)
├── 00_cropped/            # only if roi_cropping.enabled
├── 00b_isotropic/         # only if preprocessing off + isotropic_reslice on
├── 00c_downscaled/        # only if downscaling on + preprocessing off
├── 01_preprocessed/       # isotropic, shading-corrected + depth-flattened
│   └── *_processed.tif
├── 01b_raw_isotropic/     # only if raw_export.enabled — raw signal, sliced to match preprocessed geometry
├── 02_segmented/          # Cellpose labels
├── 02_segmented_downscaled/   # only if segmentation.downscale_labels < 1
├── 03_tracking/           # only if tracking.enabled
│   └── results/{tracks.csv, segments.zarr}
├── benchmark/             # only if benchmark.enabled
├── metadata/              # voxel size + image metadata
├── reports/               # Nextflow HTML / trace / timeline
├── logs/                  # per-step logs
└── pipeline_<date>.log
```

When the run is done, move it to the main server for sharing:

```bash
rsync -avh --progress /scratch-cbe/users/$USER/results/my_experiment/ /groups/pinheiro/user/$USER/
```

`/groups/pinheiro/user/` is backed up and shared — keep it for finished results, never run jobs from there.

### 1.9 Overlay tracks on RAW signal (`raw_export`)

Shading correction, Z flattening, and isotropic resampling make segmentation easier but hide the real intensity when interpreting tracks. `raw_export` runs `XY cubic rescale + optional isotropic Z resample` on the **raw** input (no shading correction) so you can load it as a viewer overlay:

```json
{ "raw_export": { "enabled": true, "factor": 0.33, "isotropic_reslice": true } }
```

| Field | Effect |
| --- | --- |
| `enabled`            | Turn the export on/off |
| `factor`             | XY downscale; pick by RAM budget (typically `0.25`–`0.5`) |
| `isotropic_reslice`  | Match XY pixel size to Z so tracks register in viewer coordinates |

Output: `01b_raw_isotropic/<name>_raw_iso_Channel*.tif` (and the merged `4D_hyperstack_raw_iso.tif` when `output.skip_merge = false`). The preprocessed chain is unaffected — `raw_export` is purely additive.

---

## Part 2 · Visualise results with `ultrack_viewer` (HIVE)

`ultrack_viewer.py` is a napari GUI for browsing the processed volume, segmentation labels, and tracks side-by-side.

### 2.1 Create the conda env (once)

Open PowerShell on the HIVE and run:

```powershell
cd path\to\spim_preprocessing
mamba env create -f ultrack_viewer_env.yml
mamba activate ultrack-viewer
```

### 2.2 Reach your results

- HIVE local disk → `cd <path>`.
- Still on `/groups/pinheiro/...` → the HIVE sees it as `V:`:

  ```powershell
  V: ; cd V:\path\to\my_experiment
  ```

### 2.3 Launch the viewer

| Layer   | Flag           | File |
| ------- | -------------- | --- |
| Tracks  | `--tracks`     | `03_tracking/results/tracks.csv` |
| Labels  | `--segments`   | `03_tracking/results/segments.zarr` |
| Volume  | `--processed`  | `01_preprocessed/4D_hyperstack_processed.tif` (or `01b_raw_isotropic/4D_hyperstack_raw_iso.tif` if `raw_export` is on) |

Typical launch:

```powershell
mamba activate ultrack-viewer
python ultrack_viewer.py `
    --tracks    03_tracking\results\tracks.csv `
    --segments  03_tracking\results\segments.zarr `
    --processed 01_preprocessed\4D_hyperstack_processed.tif `
    --preload `
    --load-downsample 2
```

Both volumes share the same voxel geometry, so `--processed` works against either file. Useful flags:

- `--preload` — whole `segments.zarr` in RAM; much smoother scrubbing. Only if HIVE has enough RAM.
- `--load_downsample 2` — keep volume in RAM at half res (8× less RAM, recommended for big datasets).
- `--downsample 2` — display-only downsample (full-res in RAM).

> Paths with spaces: wrap in double quotes, e.g. `--tracks "03_tracking\results\my tracks.csv"`.

---

## Preprocessing details

The chain is **modular** — each correction is its own script under `bin/`, its own Nextflow process, and its own SLURM profile. Every step is independently tunable and re-runnable via `nextflow run -resume`.

```
SPLIT_INPUT_FILE (optional, for hyperstack inputs)
       │
       ▼
  CROP_WITH_ROI (optional)
       │
       ▼
PLANAR_CORRECTION ──► DEPTH_CORRECTION ──► ISOTROPIC ──► DOWNSCALE_XY (optional)
                                                                │
                                                                ▼
                                                        CELLPOSE_SEGMENT
                                                                │
                                                                ▼
                                                       MERGE_HYPERSTACKS
                                                                │
                                                                ▼
                                                PREP_ULTRACK → ULTRACK_SEGMENT
                                                                │
                                                                ▼
                                                ULTRACK_LINK → ULTRACK_SOLVE → ULTRACK_EXPORT
```

| Step | Script | What it does | Default |
| --- | --- | --- | --- |
| Planar (XY) shading | `bin/planar_intensity_correction.py` | Estimates flat-field from mean-Z, divides every slice by it | `sigma_xy = 64` |
| Depth (Z) intensity | `bin/depth_intensity_correction.py` | Rescales each Z slice so a robust per-slice statistic is constant in Z | `mode = p99`, `smooth_window = 9`, `gain_clip = [0.25, 4.0]` |
| Isotropic resample  | `bin/isotropic_resample.py` | Resamples Z to match the smallest XY pixel size | `target_um = 0.374`, `order = 3` (cubic) |

All three are pure NumPy + SciPy on CPU. The math is ported **verbatim** from AIAF-32; `tests/test_aiaf32_equivalence.py` verifies bit-identical output. ImageJ metadata round-trips through every step so Cellpose / ultrack / the viewer see the corrected geometry automatically.

```bash
/usr/local/bin/python3 tests/test_aiaf32_equivalence.py   # max abs diff = 0
```

To tune any step, edit its block:

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

To skip preprocessing entirely, set `preprocessing.enabled = false` and point `preprocessing.preprocessed_dir` at the folder of timepoint TIFFs you already have.

---

## Troubleshooting

**`sbatch` job stays `PD` forever.** Cluster is busy — check `squeue --me`. If you need GPU, queue `g` is the bottleneck.

**"Input path does not exist".** Verify the path in `config.json` with `ls` from the login node. No backslash-escaped spaces in JSON strings.

**Container not found.** The `.img`/`.sif` is missing from `system.container_image` (§ 1.4) or the default location was deleted. Pre-pull your own copy (§ 1.4) and update `config.json`. Maintainers can repopulate the shared folder with `./setup_container.sh` from the login node (needs internet, ~30 min).

**Tracking fails, everything else works.** Gurobi licence is missing, expired (90-day WLS Academic), or `system.gurobi_license_path` is wrong — see § 1.5. Also confirm `segmentation.enabled = true` (tracking consumes those labels).

**Viewer sluggish / crashes on load.** Drop `--preload`, raise `--load_downsample` / `--downsample` to `4`.

**Re-use finished work.** `system.resume = true` (default) skips already-done tasks. Just re-run `sbatch submit_pipeline.sh config.json`.

---

## Contact

Andrés Gordo — andres.ortiz@imp.ac.at
