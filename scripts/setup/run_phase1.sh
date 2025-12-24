#!/bin/bash
# run_phase1.sh - Complete Phase 1: Infrastructure Setup & Baseline Benchmarks
#
# This script runs all Phase 1 tasks in sequence:
# 1. Verify GPU
# 2. Setup conda environment
# 3. Clone repositories
# 4. Download models
# 5. Run baseline benchmark
# 6. Run streaming benchmark

set -e  # Exit on any error

echo "=============================================================="
echo "PHASE 1: INFRASTRUCTURE SETUP & BASELINE BENCHMARKS"
echo "=============================================================="
echo ""
echo "This script will:"
echo "  1. Verify GPU setup"
echo "  2. Create conda environment with all dependencies"
echo "  3. Clone required repositories"
echo "  4. Download models from HuggingFace (~11GB)"
echo "  5. Run baseline benchmarks"
echo "  6. Run streaming benchmarks"
echo ""
echo "Estimated time: 30-60 minutes (mostly model download)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Step 1: Verify GPU
echo "=============================================================="
echo "STEP 1: VERIFYING GPU"
echo "=============================================================="
bash "$SCRIPT_DIR/verify_gpu.sh"

echo ""
read -p "GPU verification complete. Press Enter to continue..."

# Step 2: Setup environment
echo ""
echo "=============================================================="
echo "STEP 2: SETTING UP ENVIRONMENT"
echo "=============================================================="
bash "$SCRIPT_DIR/setup_environment.sh"

echo ""
echo "IMPORTANT: Activate the environment before continuing!"
echo ""
echo "Run this command in your terminal:"
echo "  conda activate chatterbox"
echo ""
read -p "After activating, press Enter to continue..."

# Verify environment is active
if [[ "$CONDA_DEFAULT_ENV" != "chatterbox" ]]; then
    echo "WARNING: 'chatterbox' environment may not be active."
    echo "Current environment: $CONDA_DEFAULT_ENV"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please run: conda activate chatterbox"
        exit 1
    fi
fi

# Step 3: Clone repositories
echo ""
echo "=============================================================="
echo "STEP 3: CLONING REPOSITORIES"
echo "=============================================================="
bash "$SCRIPT_DIR/clone_repos.sh"

echo ""
read -p "Repositories cloned. Press Enter to continue..."

# Step 4: Download models
echo ""
echo "=============================================================="
echo "STEP 4: DOWNLOADING MODELS"
echo "=============================================================="
echo "This will download ~11GB of models. This may take a while..."
echo ""
python "$SCRIPT_DIR/download_models.py"

echo ""
read -p "Models downloaded. Press Enter to run benchmarks..."

# Step 5: Run baseline benchmark
echo ""
echo "=============================================================="
echo "STEP 5: RUNNING BASELINE BENCHMARK"
echo "=============================================================="
echo "This will take ~10-15 minutes..."
echo ""
python "$PROJECT_ROOT/scripts/benchmark/baseline.py"

echo ""
read -p "Baseline complete. Press Enter to run streaming benchmark..."

# Step 6: Run streaming benchmark
echo ""
echo "=============================================================="
echo "STEP 6: RUNNING STREAMING BENCHMARK"
echo "=============================================================="
echo "This will take ~10-15 minutes..."
echo ""
python "$PROJECT_ROOT/scripts/benchmark/streaming.py"

# Summary
echo ""
echo "=============================================================="
echo "PHASE 1 COMPLETE!"
echo "=============================================================="
echo ""
echo "Results saved in: $PROJECT_ROOT/benchmarks/results/"
echo ""
ls -la "$PROJECT_ROOT/benchmarks/results/"
echo ""
echo "Next steps:"
echo "  1. Review benchmark results"
echo "  2. Proceed to Phase 2: ONNX/TensorRT optimization"
echo ""
