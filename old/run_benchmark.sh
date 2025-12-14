#!/bin/bash

# Complete workflow script for deploying and benchmarking with GenAI-Perf
# This uses synthetic input generation for extremely long context (128k tokens)
# Usage: ./run_benchmark.sh

set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
METHOD="hf"  # Deployment method: hf (HuggingFace)
MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
# Available engines:
#   - vllm: Fast and flexible (recommended for most use cases)
#   - sglang: Optimized for structured generation  
#   - trtllm: NVIDIA TensorRT-LLM (best performance)
# ENGINES=("python")
ENGINES=("vllm" "sglang" "trtllm")
BASE_PORT=8000
OUTPUT_DIR="artifacts"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Sanitize model name for filename
MODEL_NAME=$(echo "${MODEL}" | sed 's/\//-/g' | tr '[:upper:]' '[:lower:]')

# Synthetic input generation parameters (no input file needed!)
INPUT_SEQUENCE_LENGTH=32768   # 32k tokens context
INPUT_SEQUENCE_STDDEV=0       # No variation in input length
OUTPUT_SEQUENCE_LENGTH=200    # Fixed output length
CONCURRENCY=40                # Concurrent requests
REQUEST_COUNT=1000            # Number of requests to send (count-based benchmark)
WARMUP_REQUEST_COUNT=100      # Number of warmup requests before measurement
MEASUREMENT_INTERVAL=10000    # Measurement interval in milliseconds (10 seconds)
DEFAULT_GPU_MEMORY=0.9
echo "=============================================================================="
echo "GenAI-Perf Benchmark with Synthetic Input (32k tokens)"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Method: ${METHOD^^}"  # Display in uppercase
echo "  Model: ${MODEL}"
echo "  Engines: ${ENGINES[*]}"
echo "  Input: SYNTHETIC (${INPUT_SEQUENCE_LENGTH} tokens, stddev: ${INPUT_SEQUENCE_STDDEV})"
echo "  Output Length: ${OUTPUT_SEQUENCE_LENGTH} tokens"
echo "  Output Directory: ${OUTPUT_DIR}/"
echo "  Concurrency: ${CONCURRENCY}"
echo "  Request Count: ${REQUEST_COUNT} (will send exactly this many requests)"
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
    
    # Login to NVIDIA Container Registry if NGC_API_KEY or NVIDIA_API_KEY is set
    if [ -n "$NGC_API_KEY" ]; then
        echo "   Logging into NVIDIA Container Registry with NGC_API_KEY..."
        echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin > /dev/null 2>&1 || true
    elif [ -n "$NVIDIA_API_KEY" ]; then
        echo "   Logging into NVIDIA Container Registry with NVIDIA_API_KEY..."
        echo "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin > /dev/null 2>&1 || true
    fi
    
    GENAI_PERF_CMD="docker run --rm --net=host --gpus=all -v $PWD:/workdir -w /workdir nvcr.io/nvidia/eval-factory/genai-perf:25.11 genai-perf"
    echo "   Using Docker-based genai-perf (version 25.11)"
else
    echo "✅ genai-perf found locally"
    GENAI_PERF_CMD="genai-perf"
    
    # Check genai-perf version to determine argument compatibility
    GENAI_PERF_VERSION=$(genai-perf --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    echo "   Version: ${GENAI_PERF_VERSION:-unknown}"
    
    # Check if it supports --service-kind (required for our benchmark)
    if ! genai-perf --help 2>&1 | grep -q "service-kind"; then
        echo "   ❌ Local genai-perf does NOT support --service-kind argument"
        echo "   ⚠️  Falling back to Docker-based genai-perf..."
        
        if [ -n "$NGC_API_KEY" ]; then
            echo "   Logging into NVIDIA Container Registry with NGC_API_KEY..."
            echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin > /dev/null 2>&1 || true
        elif [ -n "$NVIDIA_API_KEY" ]; then
            echo "   Logging into NVIDIA Container Registry with NVIDIA_API_KEY..."
            echo "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin > /dev/null 2>&1 || true
        fi
        
        GENAI_PERF_CMD="docker run --rm --net=host --gpus=all -v $PWD:/workdir -w /workdir nvcr.io/nvidia/eval-factory/genai-perf:25.11 genai-perf"
        echo "   Using Docker-based genai-perf (nvcr.io/nvidia/eval-factory/genai-perf:25.11)"
    else
        echo "   ✅ Supports --service-kind argument"
    fi
fi

echo "✅ Prerequisites check passed (using synthetic input generation)"
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

# Track individual result profiles
declare -a RESULT_PROFILES

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
    
    # All engines use 32k context
    ACTUAL_MAX_LEN=32768
    ACTUAL_INPUT_LEN=30000
    
    # Start container
    echo "🚀 Starting ${ENGINE} container..."
    python docker_hf.py start \
        --model "${MODEL}" \
        --engine "${ENGINE}" \
        --port "${PORT}" \
        --max-model-len ${ACTUAL_MAX_LEN} \
        --gpu-memory ${DEFAULT_GPU_MEMORY} \
        --container-name "${CONTAINER_NAME}"
    
    # Wait for container to be ready
    echo "⏳ Waiting for model to load..."
    ENDPOINT="http://localhost:${PORT}"
    MAX_WAIT=600  # 10 minutes
    WAIT_TIME=0
    
    # Use appropriate health check endpoint based on engine
    HEALTH_ENDPOINT="/v1/models"
    MODEL_ENDPOINT="/v1/models"
    
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
    
    # Run benchmark with GenAI-Perf and synthetic input
    echo ""
    echo "📊 Running GenAI-Perf benchmark with synthetic input..."
    echo "   Input: ${ACTUAL_INPUT_LEN} tokens (32k context)"
    echo "   Output: ${OUTPUT_SEQUENCE_LENGTH} tokens"
    echo "   Concurrency: ${CONCURRENCY}"
    echo "   Request Count: ${REQUEST_COUNT}"
    
    # Create unique profile name
    PROFILE_NAME="${METHOD}_${MODEL_SANITIZED}_${ENGINE}_ISL${INPUT_SEQUENCE_LENGTH}_OSL${OUTPUT_SEQUENCE_LENGTH}"
    
    # Try with --service-kind first (newer versions), fall back to --backend (older versions)
    echo "   Attempting to run GenAI-Perf..."
    
    if ${GENAI_PERF_CMD} profile \
        -m "${MODEL}" \
        -u localhost:${PORT} \
        --endpoint-type chat \
        --service-kind openai \
        --streaming \
        --warmup-request-count ${WARMUP_REQUEST_COUNT} \
        --measurement-interval ${MEASUREMENT_INTERVAL} \
        --synthetic-input-tokens-mean ${ACTUAL_INPUT_LEN} \
        --synthetic-input-tokens-stddev ${INPUT_SEQUENCE_STDDEV} \
        --concurrency ${CONCURRENCY} \
        --output-tokens-mean ${OUTPUT_SEQUENCE_LENGTH} \
        --extra-inputs max_tokens:${OUTPUT_SEQUENCE_LENGTH} \
        --extra-inputs min_tokens:${OUTPUT_SEQUENCE_LENGTH} \
        --extra-inputs ignore_eos:true \
        --tokenizer ${MODEL} \
        --request-count ${REQUEST_COUNT} \
        --profile-export-file ${PROFILE_NAME}.json \
        2>&1 | tee genai_perf_output.log; then
        echo "   ✅ GenAI-Perf completed successfully"
    else
        # Check if it's a version issue
        if grep -q "unrecognized arguments: --service-kind" genai_perf_output.log 2>/dev/null; then
            echo "   ⚠️  Your genai-perf version doesn't support --service-kind"
            echo "   ⚠️  Please upgrade to the latest version:"
            echo "   "
            echo "      pip install --upgrade genai-perf"
            echo "   "
            echo "   Or use Docker-based genai-perf by removing local install:"
            echo "      pip uninstall genai-perf"
            echo "   "
            echo "   Skipping this engine..."
            rm -f genai_perf_output.log
            python docker_hf.py stop --container-name "${CONTAINER_NAME}"
            continue
        else
            echo "   ❌ GenAI-Perf failed with unknown error"
            cat genai_perf_output.log
            rm -f genai_perf_output.log
            python docker_hf.py stop --container-name "${CONTAINER_NAME}"
            continue
        fi
    fi
    
    rm -f genai_perf_output.log
    
    # Track result profile
    RESULT_PROFILES+=("${PROFILE_NAME}")
    echo "   ✅ Results saved to: ${OUTPUT_DIR}/"
    
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

# Summary of results
if [ ${#RESULT_PROFILES[@]} -gt 0 ]; then
    echo "📊 Benchmark results summary:"
    echo ""
    echo "Results are stored in ${OUTPUT_DIR}/ directory:"
    for profile in "${RESULT_PROFILES[@]}"; do
        echo "   - ${profile}"
    done
    echo ""
    echo "To analyze results, look for *_genai_perf.csv files in:"
    echo "   ${OUTPUT_DIR}/<model>-openai-chat-concurrency${CONCURRENCY}/"
    echo ""
    echo "Key metrics to check:"
    echo "   - Time to First Token (TTFT)"
    echo "   - Inter-Token Latency (ITL)"
    echo "   - Request Latency"
    echo "   - Throughput (tokens/sec)"
    echo ""
    echo "Example analysis with Python:"
    echo "   import pandas as pd"
    echo "   df = pd.read_csv('${OUTPUT_DIR}/<dir>/*_genai_perf.csv')"
    echo "   print(df)"
else
    echo "⚠️  No results generated"
fi

echo ""
echo "✅ Done!"
echo ""

