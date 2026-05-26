#!/usr/bin/env bash
# SPIM Pipeline - Container Setup Script
# Run this ONCE from the login node before submitting the pipeline

set -euo pipefail

CONFIG_JSON="${1:-./config.json}"

if [ ! -f "$CONFIG_JSON" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_JSON"
    exit 1
fi

# Extract output directory from config
OUTPUT_DIR=$(sed -n '/"output"/,/}/p' "$CONFIG_JSON" | grep '"directory"' | sed 's/.*: *"\([^"]*\)".*/\1/')
mkdir -p "$OUTPUT_DIR"

# Setup cache directories - we always populate the shared long-term folder so
# every pipeline run on the cluster reuses the same pre-pulled images.
CONTAINERS_DIR="/groups/pinheiro/user/andres.gordo/containers_licences"
export NXF_SINGULARITY_CACHEDIR="$CONTAINERS_DIR"
export APPTAINER_CACHEDIR="$CONTAINERS_DIR/.cache"
export APPTAINER_TMPDIR="$CONTAINERS_DIR/.tmp"
mkdir -p "$NXF_SINGULARITY_CACHEDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Container details
CONTAINER_URI="library://andresgordoortiz/spim_imp/python_packages_spim:sha256.6ef173bb45b113a36deae4315200cd8f311de2d7108b4b73e8f17a12cffe7559"
CONTAINER_SIF="$NXF_SINGULARITY_CACHEDIR/andresgordoortiz-spim_imp-python_packages_spim-sha256.6ef173bb45b113a36deae4315200cd8f311de2d7108b4b73e8f17a12cffe7559.img"

if [ -f "$CONTAINER_SIF" ]; then
    echo "Container already exists: $CONTAINER_SIF"
    echo "To force re-download, delete the file and run this script again."
    exit 0
fi

echo "================================================"
echo "  SPIM Pipeline - Container Download"
echo "================================================"
echo "This will download the Apptainer/Singularity container (~5GB)"
echo "from Sylabs Cloud. This may take 30-45 minutes."
echo ""
echo "Output: $CONTAINER_SIF"
echo "================================================"
echo ""

# Load apptainer module
module load apptainer/1.3.4 2>/dev/null || module load singularity 2>/dev/null || {
    echo "ERROR: Could not load apptainer or singularity module"
    exit 1
}

# Pull the container
echo "Downloading main pipeline container..."
apptainer pull --force "$CONTAINER_SIF" "$CONTAINER_URI"

# Also pull the Fiji container (used for label downscaling)
FIJI_DOCKER_URI="docker://fiji/fiji:20220415"
echo ""
echo "Downloading Fiji container (for label downscaling)..."
apptainer pull --force --dir "$NXF_SINGULARITY_CACHEDIR" "$FIJI_DOCKER_URI" || {
    echo "WARNING: Fiji container pull failed. Label downscaling will not be available."
    echo "You can retry later or skip this if downscale_labels is set to 1.0"
}

echo ""
echo "================================================"
echo "  Container download complete!"
echo "================================================"
echo "You can now submit the pipeline with:"
echo "  sbatch submit_pipeline.sh"
echo "================================================"
