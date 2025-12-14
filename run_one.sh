#!/bin/bash

# Single test benchmark script for quick testing
# This script deploys and benchmarks a single method/engine combination
# Usage: ./run_one.sh [METHOD] [ENGINE] [OPTIONS]

set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# ============================================================================
# Default Configuration
# ============================================================================
METHOD="${1:-hf}"                                      # Default: hf
ENGINE="${2:-vllm}"                                    # Default: vllm
MODEL="${3:-Qwen/Qwen3-30B-A3B-Thinking-2507}"         # Default model
OUTPUT_DIR="artifacts"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Benchmark parameters
INPUT_SEQUENCE_LENGTH=32768   # 32k tokens context (max model length)
ACTUAL_INPUT_LEN=30000        # Actual input tokens (leave buffer for variation)
OUTPUT_SEQUENCE_LENGTH=3000   # Fixed output
CONCURRENCY=40                # Concurrent requests
REQUEST_COUNT=1000            # Total requests
WARMUP_REQUEST_COUNT=100      # Warmup requests
DEFAULT_GPU_MEMORY=0.9

# Port configuration
if [ "${METHOD}" = "triton" ]; then
    PORT=9000  # OpenAI frontend port for Triton
else
    PORT=8000  # Default port for HF
fi

# Allow port override
PORT="${4:-$PORT}"

MODEL_SANITIZED=$(echo "${MODEL}" | sed 's/\//-/g' | tr '[:upper:]' '[:lower:]')
CONTAINER_NAME="${MODEL_SANITIZED}-${METHOD}-${ENGINE}"

echo "=============================================================================="
echo "Single Test Benchmark"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Method: ${METHOD}"
echo "  Engine: ${ENGINE}"
echo "  Model: ${MODEL}"
echo "  Container: ${CONTAINER_NAME}"
echo "  Port: ${PORT}"
echo "  Input: SYNTHETIC (${ACTUAL_INPUT_LEN} tokens)"
echo "  Output: ${OUTPUT_SEQUENCE_LENGTH} tokens"
echo "  Concurrency: ${CONCURRENCY}"
echo "  Request Count: ${REQUEST_COUNT}"
echo ""

# ============================================================================
# Prerequisite Checks
# ============================================================================
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

if ! docker info | grep -i nvidia &> /dev/null; then
    echo "❌ NVIDIA Docker runtime not found. Please install nvidia-docker2."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found."
    exit 1
fi

if [ ! -f "docker.py" ]; then
    echo "❌ docker.py not found in current directory."
    exit 1
fi

if [ ! -f "measure.py" ]; then
    echo "❌ measure.py not found in current directory."
    exit 1
fi

# Check for genai-perf
if ! command -v genai-perf &> /dev/null; then
    echo "⚠️  genai-perf not found locally. Will use Docker image."
    
    if [ -n "$NGC_API_KEY" ]; then
        echo "   Logging into NVIDIA Container Registry..."
        echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin > /dev/null 2>&1 || true
    elif [ -n "$NVIDIA_API_KEY" ]; then
        echo "   Logging into NVIDIA Container Registry..."
        echo "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin > /dev/null 2>&1 || true
    fi
else
    echo "✅ genai-perf found locally"
fi

echo "✅ Prerequisites check passed"
echo ""

# Check HF_TOKEN
if [ -z "${HF_TOKEN}" ]; then
    echo "⚠️  Warning: HF_TOKEN not set. This may be required for private models."
    echo "   Set it with: export HF_TOKEN=your_token_here"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Cleanup existing container
# ============================================================================
echo "🧹 Cleaning up existing containers..."
python3 docker.py stop --container-name "${CONTAINER_NAME}" 2>/dev/null || true
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
echo "   ✅ Cleanup complete"
echo ""

# ============================================================================
# Start container
# ============================================================================
echo "🚀 Starting ${METHOD} container with ${ENGINE} engine..."

START_ARGS=(
    "--method" "${METHOD}"
    "--model" "${MODEL}"
    "--engine" "${ENGINE}"
    "--port" "${PORT}"
    "--max-model-len" "${INPUT_SEQUENCE_LENGTH}"
    "--gpu-memory" "${DEFAULT_GPU_MEMORY}"
    "--container-name" "${CONTAINER_NAME}"
)

# Add method-specific arguments
if [ "${METHOD}" = "triton" ]; then
    START_ARGS+=("--openai-frontend" "--openai-port" "${PORT}")
fi

if ! python3 docker.py start "${START_ARGS[@]}"; then
    echo "❌ Failed to start container"
    exit 1
fi

# ============================================================================
# Wait for service to be ready
# ============================================================================
echo ""
echo "⏳ Waiting for service to be ready..."
ENDPOINT="http://localhost:${PORT}"
MAX_WAIT=600  # 10 minutes
WAIT_TIME=0

# Determine health endpoint
if [ "${METHOD}" = "triton" ]; then
    HEALTH_ENDPOINT="/health/ready"
else
    HEALTH_ENDPOINT="/v1/models"
fi

echo "   Checking: ${ENDPOINT}${HEALTH_ENDPOINT}"

while [ ${WAIT_TIME} -lt ${MAX_WAIT} ]; do
    if curl -s -o /dev/null -w "%{http_code}" "${ENDPOINT}${HEALTH_ENDPOINT}" --max-time 2 | grep -q "200"; then
        echo "✅ Service is ready!"
        break
    fi
    echo "   Still starting... (${WAIT_TIME}s elapsed)"
    sleep 10
    WAIT_TIME=$((WAIT_TIME + 10))
done

if [ ${WAIT_TIME} -ge ${MAX_WAIT} ]; then
    echo "❌ Timeout waiting for service"
    echo ""
    echo "📋 Container logs (last 50 lines):"
    python3 docker.py logs --container-name "${CONTAINER_NAME}" | tail -50
    echo ""
    python3 docker.py stop --container-name "${CONTAINER_NAME}"
    exit 1
fi

# ============================================================================
# Run benchmark
# ============================================================================
echo ""
echo "📊 Running benchmark..."
echo "   This will take several minutes..."
echo ""

OUTPUT_FILE="${OUTPUT_DIR}/benchmark_${MODEL_SANITIZED}_${METHOD}_${ENGINE}_${TIMESTAMP}.csv"

MEASURE_ARGS=(
    "--method" "${METHOD}"
    "--model" "${MODEL}"
    "--engine" "${ENGINE}"
    "--endpoint" "${ENDPOINT}"
    "--input-tokens-mean" "${ACTUAL_INPUT_LEN}"
    "--output-tokens-mean" "${OUTPUT_SEQUENCE_LENGTH}"
    "--concurrency" "${CONCURRENCY}"
    "--request-count" "${REQUEST_COUNT}"
    "--warmup-request-count" "${WARMUP_REQUEST_COUNT}"
    "--output-file" "${OUTPUT_FILE}"
    "--streaming"
)

BENCHMARK_SUCCESS=false

if python3 measure.py "${MEASURE_ARGS[@]}"; then
    echo ""
    echo "✅ Benchmark completed successfully!"
    BENCHMARK_SUCCESS=true
else
    echo ""
    echo "❌ Benchmark failed!"
    echo ""
    echo "📋 Container logs (last 50 lines):"
    python3 docker.py logs --container-name "${CONTAINER_NAME}" | tail -50
fi

# ============================================================================
# Cleanup
# ============================================================================
echo ""
echo "🛑 Stopping container..."
python3 docker.py stop --container-name "${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================================================="
if [ "$BENCHMARK_SUCCESS" = true ]; then
    echo "🎉 Test Complete: ${METHOD}/${ENGINE}"
else
    echo "❌ Test Failed: ${METHOD}/${ENGINE}"
fi
echo "=============================================================================="
echo ""

if [ "$BENCHMARK_SUCCESS" = true ]; then
    echo "✅ Results saved to: ${OUTPUT_FILE}"
    echo ""
    echo "📊 View results with:"
    echo "   cat ${OUTPUT_FILE}"
    echo ""
    echo "   Or analyze with Python:"
    echo "   import pandas as pd"
    echo "   df = pd.read_csv('${OUTPUT_FILE}')"
    echo "   print(df.T)"
    echo ""
else
    echo "❌ No results saved due to benchmark failure"
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "   1. Check container logs: python3 docker.py logs --container-name ${CONTAINER_NAME}"
    echo "   2. Check if model is accessible (HF_TOKEN may be required)"
    echo "   3. Verify GPU memory is sufficient"
    echo "   4. Check port ${PORT} is not already in use"
    echo ""
    exit 1
fi

echo "✅ Done!"
echo ""

