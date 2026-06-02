# SPIM 4D Image Processing Pipeline

A Nextflow pipeline for processing lightsheet microscopy (SPIM) data: deconvolution, segmentation with Cellpose, and merging into BigDataViewer-compatible 4D stacks.

**Authors:** Andrés Gordo & Guilherme Ventura  
**Institute:** IMP Vienna

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/andresgordoortiz/spim_preprocessing.git
cd spim_preprocessing

# 2. Edit the configuration file
nano config.json    # Set your input/output paths and parameters

# 3. Download the container (run once from login node - takes ~30 min)
./setup_container.sh

# 4. Submit to the cluster
sbatch submit_pipeline.sh
```

That's it! The pipeline handles everything else.

---

## What This Pipeline Does

1. **ROI Cropping** (optional) — Crop all timepoints using an ImageJ ROI file
2. **Preprocessing** — Shading correction, Z-intensity normalization, deconvolution (or Self-Net deblurring)
3. **Segmentation** — 3D cell segmentation with Cellpose
4. **Merging** — Combine all timepoints into a single 4D stack (TIFF or BigDataViewer HDF5)

---

## Configuration

All settings are in `config.json`. Here's what you need to change:

### Essential Settings

```json
{
  "input": {
    "directory": "/path/to/your/tiff/files/",
    "channel": 2
  },
  "output": {
    "directory": "./results/",
    "format": "bdv"
  }
}
```

Your input files should be named like: `t0001_Channel 1.tif`, `t0002_Channel 1.tif`, etc.

### Voxel Size

Either auto-detect from image metadata or set manually:

```json
{
  "voxel_size": {
    "auto_detect": false,
    "x_um": 0.347,
    "y_um": 0.347,
    "z_um": 2.0
  }
}
```

### ROI Cropping (Optional)

To crop all timepoints to a region of interest:

```json
{
  "roi_cropping": {
    "enabled": true,
    "roi_path": "./my_crop_region.roi"
  }
}
```

Create the `.roi` file in ImageJ/Fiji by drawing a rectangle on one of your images and saving it (`Edit > Selection > Save...`).

### Preprocessing Method (Optional)

The pipeline can sharpen / isotropize each volume with one of two interchangeable methods, selected by `preprocessing.method`:

- `"deconvolution"` (default) — GPU Richardson-Lucy deconvolution using `preprocessing.psf_path`.
- `"selfnet"` — deep-learning Self-Net deblurring / isotropic reconstruction using a trained `deblur_net` model.

Both produce the same isotropic, intensity-normalized output, so the rest of the pipeline (segmentation, tracking, merging) is unchanged. Self-Net runs **in place of deconvolution**, after shading and Z-intensity correction, and the rest of the preprocessing chain (WBNS, CLAHE, final normalization) still runs around it. To use Self-Net:

```json
{
  "preprocessing": {
    "method": "selfnet",
    "selfnet": {
      "model_path": "/path/to/deblur_net.pkl",
      "model_path_xz": null,
      "model_path_yz": null,
      "ngf": 64,
      "n_blocks": 6,
      "norm": "instance",
      "batch_size": 8,
      "net_min_v": 0,
      "net_max_v": 65535,
      "net_percentile_low": 30,
      "net_percentile_high": 99.999,
      "net_thres_scale": 1.5,
      "no_net_normalization": false
    }
  }
}
```

- `model_path` — trained Self-Net state-dict (`.pkl`). Required when `method` is `"selfnet"`. Use the `deblur_net_*.pkl` checkpoint (not the `netG_*` / `netD_*` files, which are training-only).
- `model_path_xz` / `model_path_yz` — optional per-view models; leave `null` to reuse `model_path` for both reslice directions.
- `net_min_v` / `net_max_v` — intensity range used to normalize the network input (defaults cover full 16-bit range).
- `net_percentile_low` / `net_percentile_high` / `net_thres_scale` — training-matched percentile normalization applied to the Self-Net input so the model sees its expected intensity distribution. Set `no_net_normalization: true` to skip it.

Self-Net also runs on a GPU node; leaving `method` unset keeps the default deconvolution behavior.

### Seqera Tower (Optional)

Track your pipeline runs online at [tower.nf](https://tower.nf):

1. Create an account and get your access token
2. Add it to the config:

```json
{
  "seqera_tower": {
    "enabled": true,
    "access_token": "your-token-here"
  }
}
```

---

## Output Structure

```
results/
├── 00_cropped/           # ROI-cropped images (if enabled)
├── 01_preprocessed/      # Deconvolved images per timepoint
├── 02_segmented/         # Cellpose segmentation masks
├── 03_hyperstack/        # Final 4D stack (HDF5+XML or TIFF)
├── logs/                 # Processing logs
├── metadata/             # Voxel size and image metadata
└── reports/              # QC report and Nextflow reports
```

The final output in `03_hyperstack/` can be opened directly in BigDataViewer or Fiji.

---

## Container

The pipeline uses a pre-built Apptainer/Singularity container hosted on Sylabs Cloud (~5GB).

**Important:** Run the setup script from the login node before submitting:

```bash
./setup_container.sh
```

This downloads the container to your results directory. Compute nodes typically can't access external networks, so this step must be done from the login node where you have internet access.

The container will be cached and reused for subsequent runs.

---

## Cluster Requirements

- SLURM scheduler
- Nextflow 23.04+
- Singularity/Apptainer
- GPU nodes (for deconvolution and Cellpose)

The pipeline is configured for the IMP cluster but can be adapted to other SLURM environments by editing `nextflow.config`.

---

## Troubleshooting

**Pipeline fails immediately:**
- Check that your input directory exists and contains TIFF files
- Verify the filename pattern matches `t####_Channel #.tif`

**Out of memory:**
- Edit `nextflow.config` to increase memory for specific processes

**Singularity pull timeout / connection reset:**
- Make sure you ran `./setup_container.sh` from the login node first
- Compute nodes typically cannot download from the internet
- The container must be pre-cached before submitting the job

**Resume after failure:**
- The pipeline automatically resumes from where it left off
- Just run `sbatch submit_pipeline.sh` again

---

## Advanced: Running Locally

For testing on a local machine:

```bash
nextflow run spim_pipeline.nf \
    --config_json config.json \
    -profile local
```

---

## Citation

If you use this pipeline, please cite:

> Gordo A., Ventura G. (2026). SPIM 4D Image Processing Pipeline. IMP Vienna.

---

## Contact

- **Andrés Gordo** — andres.ortiz@imp.ac.at
