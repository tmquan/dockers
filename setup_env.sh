#!/bin/bash

# Setup script for HuggingFace Model Deployment environment
# This script sets up the conda environment and verifies prerequisites

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_NAME="deploy"

echo "=============================================================================="
echo "HuggingFace Model Deployment - Environment Setup"
echo "=============================================================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found!"
    echo ""
    echo "Please install Miniconda or Anaconda:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Exiting without changes."
        exit 0
    fi
fi

# Create conda environment
echo "📦 Creating conda environment: ${ENV_NAME}"
echo ""
cd "${SCRIPT_DIR}"
conda env create -f environment.yml

echo ""
echo "✅ Conda environment created successfully!"
echo ""

# Activate environment and verify
echo "🔍 Verifying installation..."
echo ""

# Source conda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Check Python version
echo "Python version: $(python --version)"

# Check key packages
echo ""
echo "Checking installed packages:"
packages=("docker" "transformers" "pandas" "requests")
for pkg in "${packages[@]}"; do
    if python -c "import ${pkg}" 2>/dev/null; then
        version=$(python -c "import ${pkg}; print(${pkg}.__version__)")
        echo "  ✅ ${pkg}: ${version}"
    else
        echo "  ❌ ${pkg}: NOT FOUND"
    fi
done

# Check genai-perf
if command -v genai-perf &> /dev/null; then
    echo "  ✅ genai-perf: $(genai-perf --version 2>&1 | head -n1)"
else
    echo "  ⚠️  genai-perf: NOT FOUND (may need manual installation)"
fi

echo ""
echo "=============================================================================="
echo "Docker Prerequisites Check"
echo "=============================================================================="
echo ""

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker: $(docker --version)"
    
    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        echo "✅ Docker daemon is running"
        
        # Check NVIDIA runtime
        if docker info | grep -i nvidia &> /dev/null; then
            echo "✅ NVIDIA Docker runtime is available"
        else
            echo "⚠️  NVIDIA Docker runtime NOT found"
            echo "   Install with: sudo apt-get install nvidia-docker2"
        fi
    else
        echo "⚠️  Docker daemon is not running"
    fi
else
    echo "❌ Docker not found"
    echo "   Install from: https://docs.docker.com/get-docker/"
fi

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "⚠️  nvidia-smi not found (GPU may not be available)"
fi

echo ""
echo "=============================================================================="
echo "Setup Complete!"
echo "=============================================================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To test the deployment scripts:"
echo "  python docker_hf.py --help"
echo "  python measure_perf.py --help"
echo ""
echo "To run the complete benchmark workflow:"
echo "  ./run_benchmark.sh"
echo ""
echo "For more information, see README.md"
echo ""

