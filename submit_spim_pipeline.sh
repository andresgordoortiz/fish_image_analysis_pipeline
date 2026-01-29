#!/usr/bin/env bash
#SBATCH --no-requeue
#SBATCH --mem 30G
#SBATCH -p c
#SBATCH --qos c_medium
#SBATCH --time 2-00:00:00

# Configure bash
set -e          # exit immediately on error
set -u          # exit immediately if using undefined variables
set -o pipefail # ensure bash pipelines return non-zero status if any of their command fails

# Setup trap function to be run when canceling the pipeline job
_term() {
    echo "Caught SIGTERM signal!"
    kill -s SIGTERM -$pid 2>/dev/null || kill -s SIGTERM $pid
    wait $pid
}
trap _term TERM INT

# ============================================================================
# LOAD MODULES FIRST
# ============================================================================
module load build-env/f2022
module load nextflow/25.04.7
module load java/21

# ============================================================================
# CONFIGURE TOWER *AFTER* LOADING MODULES
# ============================================================================
export TOWER_ACCESS_TOKEN="eyJ0aWQiOiAxMzM2Nn0uZWRlMTAxYmIzZGE4ZDNjMzJjM2M1MmZkZThhNjBhZGI2M2EyNjE4Mg=="

# Optional: Specify Tower workspace (if you have multiple workspaces)
# export TOWER_WORKSPACE_ID="your_workspace_id"

# ============================================================================
# NEXTFLOW JVM SETTINGS
# ============================================================================
export NXF_OPTS="-Xss4M"
export NXF_JVM_ARGS="-Xms2g -Xmx5g"

# ============================================================================
# SLURM CONFIGURATION (for Nextflow's internal use)
# ============================================================================
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running as SLURM job ID: $SLURM_JOB_ID"
    echo "SLURM_CONF is set to: $SLURM_CONF"
else
    echo "Warning: Not running as a SLURM job"
fi

export SLURM_CONF=${SLURM_CONF:-/etc/slurm/slurm.conf}
export SLURM_EXPORT_ENV=ALL

# DO NOT set NXF_EXECUTOR - let Nextflow config handle this
# DO NOT set NXF_CLUSTER_SEED - this can interfere with Tower

# Debug output
echo "Environment variables for Tower:"
env | grep -E 'TOWER|NXF' | grep -v TOKEN

# ============================================================================
# USER CONFIGURATION
# ============================================================================
INPUT_DIR="./data"
OUTPUT_DIR="./spim_pipeline_output"
CONFIG_JSON="./config_medaka.json"
CHANNEL=2
PROFILE="standard"
RESUME=""
PIPELINE_SCRIPT="./spim_pipeline.nf"

# ============================================================================
# VALIDATION
# ============================================================================
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$CONFIG_JSON" ]; then
    echo "ERROR: Configuration file does not exist: $CONFIG_JSON"
    exit 1
fi

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "ERROR: Pipeline script does not exist: $PIPELINE_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# SETUP ENVIRONMENT
# ============================================================================
export NXF_TEMP="$OUTPUT_DIR/.nextflow_temp"
mkdir -p "$NXF_TEMP"

# ============================================================================
# LOGGING
# ============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/pipeline_submission_${TIMESTAMP}.log"

echo "============================================================================"
echo "SPIM Pipeline Submission with Tower"
echo "============================================================================"
echo "Timestamp       : $TIMESTAMP"
echo "Input directory : $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Configuration   : $CONFIG_JSON"
echo "Profile         : $PROFILE"
echo "Resume          : ${RESUME:-false}"
echo "Log file        : $LOG_FILE"
echo "Tower enabled   : YES"
echo "============================================================================"

N_FILES=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.tif" -o -name "*.tiff" \) | wc -l)
echo "Found $N_FILES TIFF files to process"
echo "============================================================================"
echo ""

# ============================================================================
# EXECUTE PIPELINE
# ============================================================================
nextflow run "$PIPELINE_SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config_json "$CONFIG_JSON" \
    --channel "$CHANNEL" \
    -profile "$PROFILE" \
    -c "./nextflow.config" \
    $RESUME \
    -with-report "$OUTPUT_DIR/reports/nextflow_report_${TIMESTAMP}.html" \
    -with-timeline "$OUTPUT_DIR/reports/nextflow_timeline_${TIMESTAMP}.html" \
    -with-trace "$OUTPUT_DIR/reports/nextflow_trace_${TIMESTAMP}.txt" \
    -with-dag "$OUTPUT_DIR/reports/nextflow_dag_${TIMESTAMP}.html" \
    -with-tower \
    2>&1 | tee "$LOG_FILE" &

# Store the PID for the trap function
pid=$!
wait $pid
EXIT_CODE=$?

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "============================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline completed successfully!"
    echo ""
    echo "Results available in:"
    echo "  - Preprocessed: $OUTPUT_DIR/01_preprocessed/"
    echo "  - Segmented   : $OUTPUT_DIR/02_segmented/"
    echo "  - Hyperstack  : $OUTPUT_DIR/03_hyperstack/"
    echo "  - Reports     : $OUTPUT_DIR/reports/"
else
    echo "Pipeline failed with exit code: $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
fi
echo "============================================================================"

exit $EXIT_CODE