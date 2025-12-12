#!/bin/bash

# Complete workflow script for deploying and benchmarking models with Triton Server
# Usage: ./run_benchmark_with_triton.sh
#
# This script uses Triton's OpenAI-compatible frontend to expose /v1/chat/completions endpoints.
# The OpenAI frontend wraps Triton Server and provides OpenAI API compatibility.

set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
METHOD="triton"  # Deployment method: Triton Inference Server with OpenAI frontend
MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
# Available backends:
#   - trtllm: TensorRT-LLM backend for Triton (best performance, requires pre-built engines)
#   - vllm: vLLM backend for Triton (fast and flexible, works with HF models directly)
#   - python: Python backend for custom models (baseline)
BACKENDS=("vllm")
BASE_PORT=9000  # Base port for OpenAI frontend (9000, 9010, etc.)
OUTPUT_DIR="artifacts"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(echo "${MODEL}" | sed 's/\//-/g' | tr '[:upper:]' '[:lower:]')

# Synthetic input generation parameters
INPUT_SEQUENCE_LENGTH=32768   # 32k tokens context
INPUT_SEQUENCE_STDDEV=0       # No variation in input length
OUTPUT_SEQUENCE_LENGTH=200    # Fixed output length
CONCURRENCY=40                # Concurrent requests
REQUEST_COUNT=1000            # Number of requests to send
WARMUP_REQUEST_COUNT=100      # Number of warmup requests
DEFAULT_GPU_MEMORY=0.9

echo "=============================================================================="
echo "Triton Server Benchmark with Synthetic Input (32k tokens)"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Method: Triton Inference Server"
echo "  Model: ${MODEL}"
echo "  Backends: ${BACKENDS[*]}"
echo "  Input: SYNTHETIC (${INPUT_SEQUENCE_LENGTH} tokens, stddev: ${INPUT_SEQUENCE_STDDEV})"
echo "  Output Length: ${OUTPUT_SEQUENCE_LENGTH} tokens"
echo "  Output Directory: ${OUTPUT_DIR}/"
echo "  Concurrency: ${CONCURRENCY}"
echo "  Request Count: ${REQUEST_COUNT}"
echo "  Warmup Requests: ${WARMUP_REQUEST_COUNT}"
echo ""

# Stop all existing Triton containers first for a clean start
echo "🧹 Cleaning up existing Triton containers..."
EXISTING_CONTAINERS=$(docker ps -a --filter "name=hf-triton-" --format "{{.Names}}" 2>/dev/null)
if [ ! -z "${EXISTING_CONTAINERS}" ]; then
    echo "   Found existing containers:"
    echo "${EXISTING_CONTAINERS}" | while read container; do
        echo "   - ${container}"
    done
    echo "   Stopping and removing..."
    docker ps -a --filter "name=hf-triton-" --format "{{.Names}}" | xargs -r docker stop 2>/dev/null
    docker ps -a --filter "name=hf-triton-" --format "{{.Names}}" | xargs -r docker rm 2>/dev/null
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

# Track individual result files for combining later
declare -a RESULT_FILES

# Deploy and benchmark each backend
for i in "${!BACKENDS[@]}"; do
    BACKEND="${BACKENDS[$i]}"
    OPENAI_PORT=$((BASE_PORT + i * 10))
    TRITON_PORT=$((8000 + i * 10))
    GRPC_PORT=$((TRITON_PORT + 1))
    METRICS_PORT=$((TRITON_PORT + 2))
    
    MODEL_SANITIZED=$(echo "${MODEL}" | sed 's/\//-/g' | tr '[:upper:]' '[:lower:]')
    CONTAINER_NAME="hf-triton-${MODEL_SANITIZED}-${BACKEND}"
    
    # All backends use OpenAI frontend for benchmarking
    USE_OPENAI_FRONTEND=true
    ENDPOINT="http://localhost:${OPENAI_PORT}"
    HEALTH_ENDPOINT="/health/ready"
    
    echo ""
    echo "=============================================================================="
    echo "Testing Backend: ${BACKEND} (OpenAI port: ${OPENAI_PORT})"
    echo "=============================================================================="
    echo ""
    
    # Stop any existing container
    echo "🧹 Cleaning up existing containers..."
    python docker_hf_with_triton.py stop --container-name "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    
    # All backends use 32k context
    ACTUAL_MAX_LEN=32768
    ACTUAL_INPUT_LEN=30000
    
    # Start container with OpenAI frontend
    echo "🚀 Starting Triton container with ${BACKEND} backend + OpenAI frontend..."
    python docker_hf_with_triton.py start \
        --model "${MODEL}" \
        --backend "${BACKEND}" \
        --openai-frontend \
        --openai-port "${OPENAI_PORT}" \
        --max-model-len ${ACTUAL_MAX_LEN} \
        --gpu-memory ${DEFAULT_GPU_MEMORY} \
        --container-name "${CONTAINER_NAME}"
    
    # Wait for container to be ready
    if [ "${USE_OPENAI_FRONTEND}" = true ]; then
        echo "⏳ Waiting for Triton OpenAI frontend to be ready..."
    else
        echo "⏳ Waiting for Triton server to be ready..."
    fi
    
    MAX_WAIT=600  # 10 minutes
    WAIT_TIME=0
    
    echo "   Checking health at: ${ENDPOINT}${HEALTH_ENDPOINT}"
    
    while [ ${WAIT_TIME} -lt ${MAX_WAIT} ]; do
        # Check if server is ready
        SERVER_READY=false
        if curl -s -o /dev/null -w "%{http_code}" "${ENDPOINT}${HEALTH_ENDPOINT}" --max-time 2 | grep -q "200"; then
            SERVER_READY=true
        fi
        
        if [ "$SERVER_READY" = true ]; then
            if [ "${USE_OPENAI_FRONTEND}" = true ]; then
                echo "✅ OpenAI frontend is ready!"
            else
                echo "✅ Triton server is ready!"
            fi
            break
        else
            echo "   Server still starting... (${WAIT_TIME}s elapsed)"
        fi
        
        sleep 10
        WAIT_TIME=$((WAIT_TIME + 10))
    done
    
    if [ ${WAIT_TIME} -ge ${MAX_WAIT} ]; then
        if [ "${USE_OPENAI_FRONTEND}" = true ]; then
            echo "❌ Timeout waiting for OpenAI frontend to be ready"
        else
            echo "❌ Timeout waiting for Triton server to be ready"
        fi
        python docker_hf_with_triton.py logs --container-name "${CONTAINER_NAME}"
        python docker_hf_with_triton.py stop --container-name "${CONTAINER_NAME}"
        continue
    fi
    
    # Run benchmark
    echo ""
    echo "📊 Running performance benchmark..."
    if [ "${USE_OPENAI_FRONTEND}" = true ]; then
        echo "   Note: Using Triton Server with ${BACKEND} backend + OpenAI frontend"
        echo "   Endpoint: ${ENDPOINT}/v1/chat/completions"
    else
        echo "   Note: Using Triton Server with ${BACKEND} backend (native v2 protocol)"
        echo "   Endpoint: ${ENDPOINT}/v2/models/..."
    fi
    
    # Create unique output filename for this backend
    BACKEND_OUTPUT="${OUTPUT_DIR}/benchmark_${METHOD}_${MODEL_SANITIZED}_${BACKEND}_${TIMESTAMP}.csv"
    
    # For Triton OpenAI frontend, we need to use the sanitized model name
    # Triton sanitizes the model name in the repository: removes '/', replaces '-' and '.' with '_'
    # Example: "Qwen/Qwen3-30B-A3B-Thinking-2507" -> "qwen3_30b_a3b_thinking_2507"
    TRITON_MODEL_NAME=$(echo "${MODEL}" | sed 's/.*\///g' | sed 's/-/_/g' | sed 's/\./_/g' | tr '[:upper:]' '[:lower:]')
    
    echo "   Model name in Triton: ${TRITON_MODEL_NAME}"
    
    # Run benchmark with appropriate API mode
    if [ "${USE_OPENAI_FRONTEND}" = true ]; then
        # Use GenAI-Perf for OpenAI-compatible endpoints with synthetic input
        echo "   Running GenAI-Perf with OpenAI frontend and synthetic input..."
        echo "   Input: ${INPUT_SEQUENCE_LENGTH} tokens (extremely long context)"
        echo "   Output: ${OUTPUT_SEQUENCE_LENGTH} tokens"
        
        # Try with --service-kind first (newer versions)
        if ${GENAI_PERF_CMD} profile \
            -m "${TRITON_MODEL_NAME}" \
            --endpoint-type chat \
            --service-kind openai \
            --streaming \
            -u localhost:${OPENAI_PORT} \
            --synthetic-input-tokens-mean ${INPUT_SEQUENCE_LENGTH} \
            --synthetic-input-tokens-stddev ${INPUT_SEQUENCE_STDDEV} \
            --concurrency ${CONCURRENCY} \
            --output-tokens-mean ${OUTPUT_SEQUENCE_LENGTH} \
            --extra-inputs max_tokens:${OUTPUT_SEQUENCE_LENGTH} \
            --extra-inputs min_tokens:${OUTPUT_SEQUENCE_LENGTH} \
            --extra-inputs ignore_eos:true \
            --tokenizer "${MODEL}" \
            --measurement-interval ${MEASUREMENT_INTERVAL} \
            --profile-export-file ${METHOD}_${MODEL_SANITIZED}_${BACKEND}.json \
            -- \
            -v \
            --max-threads=256 2>&1 | tee genai_perf_output.log; then
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
                echo "   Skipping this backend..."
                rm -f genai_perf_output.log
                python docker_hf_with_triton.py stop --container-name "${CONTAINER_NAME}"
                docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
                continue
            else
                echo "   ❌ GenAI-Perf failed with unknown error"
                cat genai_perf_output.log
                rm -f genai_perf_output.log
                python docker_hf_with_triton.py stop --container-name "${CONTAINER_NAME}"
                docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
                continue
            fi
        fi
        
        rm -f genai_perf_output.log
    else
        # For Python backend with Triton v2 protocol (not commonly used with benchmarks)
        echo "   ⚠️  Python backend benchmarking not fully supported with synthetic input"
        echo "   Skipping ${BACKEND} backend..."
        continue
    fi
    
    # Check if results were generated
    if [ "${USE_OPENAI_FRONTEND}" = true ]; then
        if [ -d "${OUTPUT_DIR}" ]; then
            echo "   ✅ Results saved to: ${OUTPUT_DIR}/"
            RESULT_FILES+=("${METHOD}_${MODEL_SANITIZED}_${BACKEND}")
        else
            echo "   ⚠️  No artifacts directory found"
        fi
    fi
    
    # Stop container
    echo ""
    echo "🛑 Stopping Triton container..."
    python docker_hf_with_triton.py stop --container-name "${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    
    echo ""
    echo "✅ ${BACKEND} benchmark complete"
done

echo ""
echo "=============================================================================="
echo "🎉 All Triton benchmarks complete!"
echo "=============================================================================="
echo ""

# Summary of results
if [ ${#RESULT_FILES[@]} -gt 0 ]; then
    echo "📊 Benchmark results summary:"
    echo ""
    echo "Results are stored in ${OUTPUT_DIR}/ directory:"
    for profile in "${RESULT_FILES[@]}"; do
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
else
    echo "⚠️  No results generated"
fi

echo ""
echo "✅ Done!"
echo ""
echo "Full Triton Server deployment completed with:"
echo "  - Automatic model repository creation"
echo "  - Backend-specific config.pbtxt files"
echo "  - OpenAI-compatible frontend (exposes /v1/chat/completions)"
echo "  - Proper Triton Server configuration"
echo ""
echo "See TRITON_README.md for more details on the implementation."
echo ""

