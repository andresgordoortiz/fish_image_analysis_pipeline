#!/usr/bin/env bash
#SBATCH --no-requeue
#SBATCH --mem 8GB
#SBATCH -p c
#SBATCH --qos c_medium
#SBATCH --time 2-00:00:00

# SPIM Pipeline Submission Script
# All configuration is in config.json - edit that file, not this one!

set -uo pipefail

# Handle job cancellation gracefully
_term() {
    echo "Caught signal - terminating pipeline..."
    kill -s SIGTERM -$pid 2>/dev/null || kill -s SIGTERM $pid
    wait $pid
}
trap _term TERM INT

# Load required modules (module load can return non-zero on HPC systems even on success)
echo "Loading modules..."
module load build-env/f2022 2>&1 || true
module load nextflow/25.04.7 2>&1 || true
module load java/21 2>&1 || true

# Verify critical commands are available
if ! command -v nextflow &>/dev/null; then
    echo "ERROR: nextflow not found after module load. Check 'module avail nextflow'."
    exit 1
fi
echo "Modules loaded: nextflow $(nextflow -version 2>&1 | grep -oP 'version \K\S+' || echo '?')"

# Re-enable strict error handling for the rest of the script
set -e

# Configuration file - REQUIRED as first positional argument.
# Any additional args are forwarded verbatim to `nextflow run`.
if [ $# -lt 1 ] || [[ "$1" == -* ]]; then
    echo "ERROR: missing required config.json argument"
    echo ""
    echo "Usage: sbatch submit_pipeline.sh <path/to/config.json> [extra nextflow args...]"
    echo "   or: ./submit_pipeline.sh   <path/to/config.json> [extra nextflow args...]"
    exit 1
fi

CONFIG_JSON="$1"
shift 1

if [ ! -f "$CONFIG_JSON" ]; then
    echo "ERROR: config file not found: $CONFIG_JSON"
    exit 1
fi

# Resolve to an absolute path so that:
#   1. Nextflow (which may be launched from a different working directory)
#      can always find the file.
#   2. The path printed in logs is unambiguous.
# `realpath` is available on the HPC images we use; fall back to a python
# one-liner if not.
if command -v realpath &>/dev/null; then
    CONFIG_JSON="$(realpath "$CONFIG_JSON")"
else
    CONFIG_JSON="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$CONFIG_JSON")"
fi

EXTRA_NXF_ARGS="${@}"

# Extract paths from config using grep/sed (simple JSON parsing)
# We look for lines after "input" and "output" sections
INPUT_DIR=$(sed -n '/"input"/,/}/p' "$CONFIG_JSON" | grep '"directory"' | sed 's/.*: *"\([^"]*\)".*/\1/')
OUTPUT_DIR=$(sed -n '/"output"/,/}/p' "$CONFIG_JSON" | grep '"directory"' | sed 's/.*: *"\([^"]*\)".*/\1/')

# Sanitize paths: remove shell-style backslash escaping (e.g. "Position\ 5" -> "Position 5")
# and trailing slashes, in case users escape spaces in the JSON string
INPUT_DIR=$(echo "$INPUT_DIR" | sed 's/\\\\//g; s/\\//g' | sed 's:/*$::')
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | sed 's/\\\\//g; s/\\//g' | sed 's:/*$::')

# Validate input path exists. Accept either a directory of per-timepoint TIFFs
# OR a single .czi / hyperstack .tif/.tiff file (Nextflow will resolve the
# parent directory automatically).
if [ ! -e "$INPUT_DIR" ]; then
    echo "ERROR: Input path does not exist: $INPUT_DIR"
    echo "  (If the path contains spaces, use plain spaces in config.json, not backslash-escaped)"
    exit 1
fi
if [ ! -d "$INPUT_DIR" ] && [ ! -f "$INPUT_DIR" ]; then
    echo "ERROR: Input path is neither a directory nor a regular file: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Setup Seqera Tower (extract token from config — grep may return 1 if key absent)
TOWER_TOKEN=$(grep -o '"access_token"[[:space:]]*:[[:space:]]*"[^"]*"' "$CONFIG_JSON" 2>/dev/null | sed 's/.*: *"\([^"]*\)"/\1/' || true)
TOWER_FLAG=""
if [ -n "$TOWER_TOKEN" ] && [ "$TOWER_TOKEN" != "YOUR_SEQERA_TOWER_TOKEN_HERE" ]; then
    export TOWER_ACCESS_TOKEN="$TOWER_TOKEN"
    TOWER_FLAG="-with-tower"
    echo "Seqera Tower: enabled"
else
    echo "Seqera Tower: disabled"
fi

# Setup caching directories - must set BOTH singularity and apptainer variables
CACHE_DIR="$OUTPUT_DIR/singularity_images"
export NXF_SINGULARITY_CACHEDIR="$CACHE_DIR"
export NXF_APPTAINER_CACHEDIR="$CACHE_DIR"
export SINGULARITY_CACHEDIR="$CACHE_DIR"
export APPTAINER_CACHEDIR="$CACHE_DIR"
export SINGULARITY_TMPDIR="$CACHE_DIR/tmp"
export APPTAINER_TMPDIR="$CACHE_DIR/tmp"
export NXF_TEMP="$OUTPUT_DIR/.nextflow_temp"
export NXF_OPTS="-Xss4M"
export NXF_JVM_ARGS="-Xms2g -Xmx5g"
mkdir -p "$CACHE_DIR" "$CACHE_DIR/tmp" "$NXF_TEMP"

# Check that container was pre-pulled (compute nodes often can't access internet)
CONTAINER_BASENAME="andresgordoortiz-spim_imp-python_packages_spim-sha256.6ef173bb45b113a36deae4315200cd8f311de2d7108b4b73e8f17a12cffe7559.img"
CONTAINER_SIF="$CACHE_DIR/$CONTAINER_BASENAME"
if [ ! -f "$CONTAINER_SIF" ]; then
    echo ""
    echo "ERROR: Container image not found!"
    echo "  Expected: $CONTAINER_SIF"
    echo ""
    echo "Run the setup script from the login node first:"
    echo "  ./setup_container.sh"
    echo ""
    echo "Or copy from an existing results directory:"
    echo "  cp -L results_testing/singularity_images/$CONTAINER_BASENAME $CACHE_DIR/"
    echo ""
    exit 1
fi

echo "Container: $CONTAINER_SIF (cached)"

# Print summary
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/pipeline_${TIMESTAMP}.log"
N_FILES=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -name '*.tif' -o -name '*.tiff' \) 2>/dev/null | wc -l)

echo ""
echo "================================================"
echo "  SPIM Pipeline - IMP Vienna"
echo "================================================"
echo "Config       : $CONFIG_JSON"
echo "Input        : $INPUT_DIR"
echo "Output       : $OUTPUT_DIR"
echo "TIFF files   : $N_FILES"
echo "Log          : $LOG_FILE"
echo "================================================"
echo ""

# Parse resume flag from config (default: true)
RESUME_FLAG="-resume"
RESUME_VAL=$(python3 -c "import json; print(json.load(open('$CONFIG_JSON')).get('system',{}).get('resume',True))" 2>/dev/null || echo "True")
if [ "$RESUME_VAL" = "False" ]; then
    RESUME_FLAG=""
    echo "Resume: disabled (config)"
else
    echo "Resume: enabled"
fi

# Ensure the reports directory exists
mkdir -p "$OUTPUT_DIR/reports"

# Run the pipeline (Nextflow reads all params from config.json)
nextflow run ./spim_pipeline.nf \
    --config_json "$CONFIG_JSON" \
    -c ./nextflow.config \
    $RESUME_FLAG \
    -with-report "$OUTPUT_DIR/reports/report_${TIMESTAMP}.html" \
    -with-timeline "$OUTPUT_DIR/reports/timeline_${TIMESTAMP}.html" \
    -with-trace "$OUTPUT_DIR/reports/trace_${TIMESTAMP}.txt" \
    $TOWER_FLAG \
    $EXTRA_NXF_ARGS \
    2>&1 | tee "$LOG_FILE" &

pid=$!
wait $pid
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Pipeline completed successfully!"
    echo ""
    echo "Results:"
    echo "  - Preprocessed : $OUTPUT_DIR/01_preprocessed/"
    echo "  - Segmented    : $OUTPUT_DIR/02_segmented/"
    echo "  - Hyperstacks  : in 01_preprocessed/ and 02_segmented/ (if merge enabled)"
    echo "  - Tracking     : $OUTPUT_DIR/03_tracking/ (if tracking enabled)"
    echo "  - Reports      : $OUTPUT_DIR/reports/"
else
    echo "✗ Pipeline failed (exit code: $EXIT_CODE)"
    echo "Check log: $LOG_FILE"
fi

exit $EXIT_CODE
