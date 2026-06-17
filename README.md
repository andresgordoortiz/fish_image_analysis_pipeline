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

#### Picking which timepoints to process

Use `input.timepoints` to process specific timepoints instead of all of them (or the first N). Three forms are accepted:

```json
"input": {
  "timepoints": 42,                       // one specific timepoint
  "timepoints": [1, 5, 10, 20],          // a list of timepoints
  "timepoints": "1,5,10-12,42-44"        // comma + inclusive ranges
}
```

Out-of-range timepoints are silently skipped (they don't exist, so there's nothing to process). The output naming, downstream stages and intermediate folders are unaffected — only the timepoints that run change.

To take "the first N" of a continuous run, just use a range — `"timepoints": "1-5"` is the equivalent of the old `max_timepoints: 5` for any data that starts at t0001. Leave `timepoints` `null` (default) to process every timepoint.

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
| `preprocessing.enabled` | Shading + Z correction + deconvolution (or Self-Net) |
| `preprocessing.aydin.enabled` | Optional **aydin** self-supervised denoising (runs before deconv). Recommended for noisy samples (medaka embryos) — see [Aydin denoising](#aydin-self-supervised-denoising-optional) below. |
| `roi_cropping.enabled`  | Crop every timepoint to a Fiji `.roi` rectangle |
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
├── 00a_denoised/          # only if preprocessing.aydin.enabled
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
    --load_downsample 2
```

Flag cheatsheet:

- `--preload` — load the whole `segments.zarr` into RAM and use a GPU-friendly display volume. This makes scrubbing across time **much** smoother. Only use it if the HIVE has enough RAM for your volume.
- `--load_downsample 2` — keep the volume in RAM at half resolution (8× less RAM). **Recommended** for big datasets; the analysis still works at half-res.
- `--downsample 2` — display-only downsample (full-res is still in RAM). Use this if you only want a faster on-screen render.

> Paths with spaces: wrap them in double quotes, e.g. `--tracks "03_tracking\results\my tracks.csv"`.

Once napari opens, use the layer panel to toggle the processed volume, the segmentation labels, and the tracks on/off. The time slider scrubs through timepoints.

---

## Aydin self-supervised denoising (optional)

The pipeline can run [aydin](https://github.com/royerlab/aydin) (Royer's lab) as an **optional** pre-processing step that runs *before* deconvolution / Self-Net. Aydin is self-supervised — it needs **no clean ground truth** — and exposes a curated menu of classical (NLM, BMnD, bilateral, …) and ML (Noise2Self-FGR, Noise2Self-CNN) denoisers that auto-tune their parameters.

This is the recommended way to deal with **noisy, cloudy embryos** (e.g. medaka) where you want to suppress background without blurring small, dim nuclei.

### When to turn it on

* **Yes** — your nuclei are dim or the background is heterogeneous (autofluorescence, scattering, cloudy cytoplasm). Medaka embryos typically benefit.
* **No** — your data is already clean (low-noise acquisitions, good signal-to-noise). Aydin is a no-op on clean data but it will still take time to auto-tune, so leave it off when you don't need it.

### Configuration

Set in `config.json` under `preprocessing.aydin`:

```json
"aydin": {
  "enabled":  true,
  "method":   "nlm",      // see algorithms below
  "tile_shape": "",       // e.g. "128,256,256" if a single slab doesn't fit in RAM
  "max_memory_mb": 0      // 0 = no cap; otherwise aydin auto-picks a tile shape
}
```

The output of the aydin step is written to `00a_denoised/t*_denoised.tif` and then fed into the regular deconvolution / Self-Net / Cellpose stages as if it were the raw input — the rest of the pipeline does not change.

### Recommended algorithms (medaka lightsheet)

| `method` | What it is | When to use it |
| --- | --- | --- |
| `nlm` **(default)** | Non-local means, 3D, edge-preserving | **First choice for medaka.** No hallucination, preserves small/dim nuclei. |
| `bmnd` | n-D generalisation of BM3D | Sharper than NLM, ~2–3× slower. Use after NLM if you want crisper nuclei. |
| `bilateral` | Edge-preserving bilateral filter | Cheaper than NLM, less sharp. |
| `gaussianmedian` | Mixed Gaussian / median | Cheap baseline. |
| `noise2self-fgr` | ML — CatBoost/lightGBM regressor | If classical under-denoises. CPU only, no hallucination problem (per aydin README). |
| `noise2self-cnn` | ML — PyTorch CNN | **Avoid for medaka.** Aydin's own README flags this as hallucination-prone for small features. We expose it for completeness, not as a recommendation. |

> The full list of algorithms is at the top of [spim_aydin_preprocess.py](spim_aydin_preprocess.py) (`AYDIN_METHODS`).

### Memory

Aydin's working set is roughly **8× the input volume** (input + denoiser scratch + output + integration buffers). For a `(500, 1024, 1024)` uint16 timepoint (~1 GB raw) the recommended allocation is **32 GB**. If the full volume doesn't fit, set `tile_shape` (e.g. `"128,256,256"`) or `max_memory_mb` and aydin will tile the volume automatically. Per-timepoint failures (exit 137) almost always mean the tile is still too big — halve the tile and retry.

### Why aydin, not Noise2Self?

Noise2Self ([czbiohub-sf/noise2self](https://github.com/czbiohub-sf/noise2self)) is the academic paper implementation — 98% Jupyter notebooks, no 3D support, no CLI, you supply the architecture and data loader. Aydin is the productionised version of the same idea from the same lab: n-dimensional arrays, 3D/volumetric first-class, CPU and GPU paths, a curated denoiser menu, and a working pipeline-friendly Python API. We chose aydin because it drops in; bolting Noise2Self onto this Nextflow pipeline would be weeks of engineering.

---

## Troubleshooting

**`sbatch` job stays in `PD` forever.** The cluster is busy. `squeue --me` will show the reason. If you need GPU, queue `g` is the bottleneck.

**Pipeline fails immediately with "Input path does not exist".** Check the path in `config.json`. Use `ls` from the login node to confirm. Remember: no backslash-escaped spaces in JSON strings.

**Container not found error.** The pre-pulled container lives at `/groups/pinheiro/user/andres.gordo/containers_licences/`. If it is missing, ask Andrés to repopulate it, or run `./setup_container.sh` from the login node (needs internet, ~30 min).

**Tracking fails but everything else works.** Make sure the Gurobi license is in `/groups/pinheiro/user/andres.gordo/containers_licences/gurobi.lic` and that `segmentation.enabled = true` (tracking consumes the segmentation labels).

**Viewer is sluggish / crashes on load.** Drop `--preload`, set `--load_downsample 4`, or set `--downsample 4` for a low-res display.

**Want to re-use finished tasks.** `system.resume = true` (default) makes Nextflow skip already-done work. Just `sbatch submit_pipeline.sh config.json` again.

**Aydin step fails with `ImportError: aydin`.** The container was built before aydin was added. Rebuild it: `sbatch submit_pipeline.sh config.json` won't trigger a rebuild — you need to rerun the build workflow or `docker build` the image locally with the updated `microscopy_env.yml`.

**Aydin step fails with exit 137 (OOM).** The auto-picked tile is too large for the SLURM allocation. Either bump `AYDIN_DENOISE` memory in `nextflow.config`, or set `preprocessing.aydin.tile_shape` (e.g. `"128,256,256"`) in `config.json` to force a smaller tile.

**Aydin step produces visibly blobby / "hallucinated" nuclei.** You almost certainly set `method: "noise2self-cnn"`. Switch to `nlm` (default) or `bmnd`. The aydin README itself warns that the CNN variant is hallucination-prone for small features.

---

## Contact

Andrés Gordo — andres.ortiz@imp.ac.at
