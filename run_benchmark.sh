#!/bin/bash

# Complete workflow script for deploying and benchmarking Qwen3-30B-A3B-Thinking-2507
# Usage: ./run_benchmark.sh

set -e

# Configuration
METHOD="hf"  # Deployment method: hf (HuggingFace), nim (NVIDIA NIM), unim (Multi-LLM NIM)
MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
# Available engines:
#   - vllm: Fast and flexible (recommended for most use cases)
#   - sglang: Optimized for structured generation  
#   - trtllm: NVIDIA TensorRT-LLM (best performance)
ENGINES=("trtllm" "vllm" "sglang")
BASE_PORT=8000
INPUT_FILE="input.jsonl"
OUTPUT_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Sanitize model name for filename
MODEL_NAME=$(echo "${MODEL}" | sed 's/\//-/g' | tr '[:upper:]' '[:lower:]')
COMBINED_OUTPUT="benchmark_results_${METHOD}_${MODEL_NAME}_${TIMESTAMP}.csv"
CONCURRENCY=40
REQUEST_COUNT=100

echo "=============================================================================="
echo "Qwen3-30B-A3B-Thinking-2507 Benchmark Workflow"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Method: ${METHOD^^}"  # Display in uppercase
echo "  Model: ${MODEL}"
echo "  Engines: ${ENGINES[*]}"
echo "  Input: ${INPUT_FILE}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Combined Output: ${COMBINED_OUTPUT}"
echo "  Concurrency: ${CONCURRENCY}"
echo "  Request Count: ${REQUEST_COUNT}"
echo ""

# Stop all existing containers first for a clean start
echo "🧹 Cleaning up existing containers (method: ${METHOD})..."
EXISTING_CONTAINERS=$(docker ps -a --filter "name=${METHOD}-" --format "{{.Names}}" 2>/dev/null)
if [ ! -z "${EXISTING_CONTAINERS}" ]; then
    echo "   Found existing containers:"
    echo "${EXISTING_CONTAINERS}" | while read container; do
        echo "   - ${container}"
    done
    echo "   Stopping and removing..."
    docker ps -a --filter "name=${METHOD}-" --format "{{.Names}}" | xargs -r docker stop 2>/dev/null
    docker ps -a --filter "name=${METHOD}-" --format "{{.Names}}" | xargs -r docker rm 2>/dev/null
    echo "   ✅ Cleanup complete"
else
    echo "   No existing containers found"
fi
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

if ! docker info | grep -i nvidia &> /dev/null; then
    echo "❌ NVIDIA Docker runtime not found. Please install nvidia-docker2."
    exit 1
fi

if ! command -v genai-perf &> /dev/null; then
    echo "⚠️  genai-perf not found locally. Will use Docker image: nvcr.io/nvidia/eval-factory/genai-perf:25.11"
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker or genai-perf (pip install genai-perf)."
        exit 1
    fi
else
    echo "✅ genai-perf found locally"
fi

if [ ! -f "${INPUT_FILE}" ]; then
    echo "❌ Input file not found: ${INPUT_FILE}"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Check if HF_TOKEN is set
if [ -z "${HF_TOKEN}" ]; then
    echo "⚠️  Warning: HF_TOKEN not set. This may be required for private models."
    echo "   Set it with: export HF_TOKEN=your_token_here"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Track individual result files for combining later
declare -a RESULT_FILES

# Deploy and benchmark each engine
for i in "${!ENGINES[@]}"; do
    ENGINE="${ENGINES[$i]}"
    PORT=$((BASE_PORT + i))
    # Container name format: method-model-engine (e.g., hf-qwen3-30b-a3b-thinking-2507-vllm)
    MODEL_SANITIZED=$(echo "${MODEL}" | sed 's/\//-/g' | tr '[:upper:]' '[:lower:]')
    CONTAINER_NAME="${METHOD}-${MODEL_SANITIZED}-${ENGINE}"
    
    echo ""
    echo "=============================================================================="
    echo "Testing Engine: ${ENGINE} (Port: ${PORT})"
    echo "=============================================================================="
    echo ""
    
    # Stop any existing container
    echo "🧹 Cleaning up existing containers..."
    python docker_hf.py stop --container-name "${CONTAINER_NAME}" 2>/dev/null || true
    
    # Start container
    echo "🚀 Starting ${ENGINE} container..."
    python docker_hf.py start \
        --model "${MODEL}" \
        --engine "${ENGINE}" \
        --port "${PORT}" \
        --max-model-len 131072 \
        --gpu-memory 0.8 \
        --container-name "${CONTAINER_NAME}"
    
    # Wait for container to be ready
    echo "⏳ Waiting for model to load..."
    ENDPOINT="http://localhost:${PORT}"
    MAX_WAIT=600  # 10 minutes
    WAIT_TIME=0
    
    # Use appropriate health check endpoint based on engine
    if [ "${ENGINE}" = "trtllm" ]; then
        HEALTH_ENDPOINT="/health"
        MODEL_ENDPOINT="/health"
    else
        HEALTH_ENDPOINT="/v1/models"
        MODEL_ENDPOINT="/v1/models"
    fi
    
    echo "   Checking server health at: ${ENDPOINT}${HEALTH_ENDPOINT}"
    echo "   Checking model availability at: ${ENDPOINT}${MODEL_ENDPOINT}"
    
    while [ ${WAIT_TIME} -lt ${MAX_WAIT} ]; do
        # First check if server is healthy
        SERVER_READY=false
        if curl -s -o /dev/null -w "%{http_code}" "${ENDPOINT}${HEALTH_ENDPOINT}" --max-time 2 | grep -q "200"; then
            SERVER_READY=true
        fi
        
        # Then check if model endpoint is accessible
        MODEL_READY=false
        if curl -s -o /dev/null -w "%{http_code}" "${ENDPOINT}${MODEL_ENDPOINT}" --max-time 2 | grep -q "200"; then
            MODEL_READY=true
        fi
        
        if [ "$SERVER_READY" = true ] && [ "$MODEL_READY" = true ]; then
            echo "✅ Server is healthy and model endpoint is ready!"
            break
        elif [ "$SERVER_READY" = true ]; then
            echo "   Server is healthy but model not ready yet... (${WAIT_TIME}s elapsed)"
        else
            echo "   Server still starting... (${WAIT_TIME}s elapsed)"
        fi
        
        sleep 10
        WAIT_TIME=$((WAIT_TIME + 10))
    done
    
    if [ ${WAIT_TIME} -ge ${MAX_WAIT} ]; then
        echo "❌ Timeout waiting for model to load"
        python docker_hf.py logs --container-name "${CONTAINER_NAME}"
        python docker_hf.py stop --container-name "${CONTAINER_NAME}"
        continue
    fi
    
    # Run benchmark (genai-perf will handle warmup with --warmup-request-count flag)
    echo ""
    echo "📊 Running performance benchmark..."
    echo "   Note: genai-perf will run 5 warmup requests before measurement"
    
    # Create unique output filename for this engine with method prefix
    ENGINE_OUTPUT="${OUTPUT_DIR}/benchmark_${METHOD}_${MODEL_SANITIZED}_${ENGINE}_${TIMESTAMP}.csv"
    
    python measure_perf.py \
        --model "${MODEL}" \
        --backend "${ENGINE}" \
        --endpoint "${ENDPOINT}" \
        --input "${INPUT_FILE}" \
        --output "${ENGINE_OUTPUT}" \
        --method "${METHOD}" \
        --concurrency "${CONCURRENCY}" \
        --request-count "${REQUEST_COUNT}"
    
    # Track the result file if benchmark succeeded
    if [ -f "${ENGINE_OUTPUT}" ]; then
        RESULT_FILES+=("${ENGINE_OUTPUT}")
        echo "   ✅ Results saved to: ${ENGINE_OUTPUT}"
    fi
    
    # Stop container
    echo ""
    echo "🛑 Stopping ${ENGINE} container..."
    python docker_hf.py stop --container-name "${CONTAINER_NAME}"
    
    echo ""
    echo "✅ ${ENGINE} benchmark complete"
done

echo ""
echo "=============================================================================="
echo "🎉 All benchmarks complete!"
echo "=============================================================================="
echo ""

# Combine all individual result files into one
if [ ${#RESULT_FILES[@]} -gt 0 ]; then
    echo "📊 Combining results from ${#RESULT_FILES[@]} engine(s)..."
    
    # Copy first file with headers
    cp "${RESULT_FILES[0]}" "${COMBINED_OUTPUT}"
    
    # Append data from other files (skip headers)
    for ((i=1; i<${#RESULT_FILES[@]}; i++)); do
        tail -n +2 "${RESULT_FILES[$i]}" >> "${COMBINED_OUTPUT}"
    done
    
    echo "   ✅ Combined results saved to: ${COMBINED_OUTPUT}"
    echo ""
    echo "Individual results:"
    for file in "${RESULT_FILES[@]}"; do
        echo "   - ${file}"
    done
else
    echo "⚠️  No results to combine"
fi

echo ""

if [ -f "${COMBINED_OUTPUT}" ]; then
    echo "📊 Summary:"
    echo ""
    # Display results table
    if command -v column &> /dev/null; then
        cat "${COMBINED_OUTPUT}" | column -t -s ','
    else
        cat "${COMBINED_OUTPUT}"
    fi
fi

echo ""
echo "✅ Done!"

