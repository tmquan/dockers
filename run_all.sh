#!/bin/bash

# Unified benchmark script for all deployment methods and engines
# This script orchestrates model deployment and performance benchmarking
# Usage: ./run_all.sh

set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# ============================================================================
# Configuration
# ============================================================================
MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="artifacts"
MODEL_SANITIZED=$(echo "${MODEL}" | sed 's/\//-/g' | tr '[:upper:]' '[:lower:]')

# Benchmark parameters (synthetic input)
INPUT_SEQUENCE_LENGTH=32768   # 32k tokens context (max model length)
ACTUAL_INPUT_LEN=30000        # Actual input tokens (leave buffer for variation)
OUTPUT_SEQUENCE_LENGTH=3000   # Fixed output
CONCURRENCY=40                # Concurrent requests
REQUEST_COUNT=1000            # Total requests
WARMUP_REQUEST_COUNT=100      # Warmup requests
DEFAULT_GPU_MEMORY=0.9

# ============================================================================
# Deployment Configuration
# Methods: hf (HuggingFace), triton (Triton Server)
# Engines per method:
#   - hf: vllm, sglang, trtllm
#   - triton: vllm, python, trtllm
# ============================================================================

# Select deployment methods to test
METHODS=("hf" "triton")

# Define engines for each method
declare -A METHOD_ENGINES
METHOD_ENGINES["hf"]="vllm sglang trtllm"
METHOD_ENGINES["triton"]="vllm python"  # trtllm requires pre-built engines

# Port configuration
declare -A METHOD_BASE_PORT
METHOD_BASE_PORT["hf"]=8000
METHOD_BASE_PORT["triton"]=9000  # OpenAI frontend port

echo "=============================================================================="
echo "Unified Benchmark Suite"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Model: ${MODEL}"
echo "  Methods: ${METHODS[*]}"
echo "  Input: SYNTHETIC (${ACTUAL_INPUT_LEN} tokens)"
echo "  Output: ${OUTPUT_SEQUENCE_LENGTH} tokens"
echo "  Concurrency: ${CONCURRENCY}"
echo "  Request Count: ${REQUEST_COUNT}"
echo "  Output Directory: ${OUTPUT_DIR}/"
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

# Check if docker.py and measure.py exist
if [ ! -f "docker.py" ]; then
    echo "❌ docker.py not found in current directory."
    exit 1
fi

if [ ! -f "measure.py" ]; then
    echo "❌ measure.py not found in current directory."
    exit 1
fi

# Check for genai-perf (local or Docker)
if ! command -v genai-perf &> /dev/null; then
    echo "⚠️  genai-perf not found locally. Will use Docker image."
    
    # Login to NVIDIA Container Registry if credentials available
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

# ============================================================================
# Cleanup existing containers
# ============================================================================
echo "🧹 Cleaning up existing containers..."
CONTAINER_PATTERNS=("-hf-" "-triton-" "-nim-" "-unim-")

for pattern in "${CONTAINER_PATTERNS[@]}"; do
    EXISTING=$(docker ps -a --filter "name=${pattern}" --format "{{.Names}}" 2>/dev/null || true)
    if [ ! -z "${EXISTING}" ]; then
        echo "   Found containers with pattern '${pattern}':"
        echo "${EXISTING}" | while read container; do
            echo "   - ${container}"
        done
        echo "   Stopping and removing..."
        docker ps -a --filter "name=${pattern}" --format "{{.Names}}" | xargs -r docker stop 2>/dev/null || true
        docker ps -a --filter "name=${pattern}" --format "{{.Names}}" | xargs -r docker rm 2>/dev/null || true
    fi
done
echo "   ✅ Cleanup complete"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Track results
# ============================================================================
declare -a RESULT_PROFILES
declare -a FAILED_TESTS

# ============================================================================
# Main benchmark loop
# ============================================================================
TOTAL_TESTS=0
SUCCESSFUL_TESTS=0
FAILED_TESTS_COUNT=0

# Count total tests
for method in "${METHODS[@]}"; do
    engines=(${METHOD_ENGINES[$method]})
    TOTAL_TESTS=$((TOTAL_TESTS + ${#engines[@]}))
done

echo "📊 Running ${TOTAL_TESTS} benchmark tests..."
echo ""

TEST_NUM=0

for method in "${METHODS[@]}"; do
    engines=(${METHOD_ENGINES[$method]})
    BASE_PORT=${METHOD_BASE_PORT[$method]}
    
    for i in "${!engines[@]}"; do
        ENGINE="${engines[$i]}"
        PORT=$((BASE_PORT + i * 10))
        
        TEST_NUM=$((TEST_NUM + 1))
        
        echo ""
        echo "=============================================================================="
        echo "Test ${TEST_NUM}/${TOTAL_TESTS}: ${method}/${ENGINE}"
        echo "=============================================================================="
        echo ""
        
        # Generate container name
        CONTAINER_NAME="${MODEL_SANITIZED}-${method}-${ENGINE}"
        
        # Cleanup any existing container
        echo "🧹 Cleaning up..."
        python3 docker.py stop --container-name "${CONTAINER_NAME}" 2>/dev/null || true
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
        
        # Start container
        echo "🚀 Starting ${method} container with ${ENGINE} engine..."
        
        START_ARGS=(
            "--method" "${method}"
            "--model" "${MODEL}"
            "--engine" "${ENGINE}"
            "--port" "${PORT}"
            "--max-model-len" "${INPUT_SEQUENCE_LENGTH}"
            "--gpu-memory" "${DEFAULT_GPU_MEMORY}"
            "--container-name" "${CONTAINER_NAME}"
        )
        
        # Add method-specific arguments
        if [ "${method}" = "triton" ]; then
            START_ARGS+=("--openai-frontend" "--openai-port" "${PORT}")
        fi
        
        if ! python3 docker.py start "${START_ARGS[@]}"; then
            echo "❌ Failed to start container"
            FAILED_TESTS+=("${method}/${ENGINE}: Container start failed")
            FAILED_TESTS_COUNT=$((FAILED_TESTS_COUNT + 1))
            continue
        fi
        
        # Wait for container to be ready
        echo "⏳ Waiting for service to be ready..."
        ENDPOINT="http://localhost:${PORT}"
        MAX_WAIT=600  # 10 minutes
        WAIT_TIME=0
        
        # Determine health endpoint
        if [ "${method}" = "triton" ]; then
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
            python3 docker.py logs --container-name "${CONTAINER_NAME}" | tail -50
            python3 docker.py stop --container-name "${CONTAINER_NAME}"
            FAILED_TESTS+=("${method}/${ENGINE}: Service timeout")
            FAILED_TESTS_COUNT=$((FAILED_TESTS_COUNT + 1))
            continue
        fi
        
        # Run benchmark
        echo ""
        echo "📊 Running benchmark..."
        
        OUTPUT_FILE="${OUTPUT_DIR}/benchmark_${MODEL_SANITIZED}_${method}_${ENGINE}_${TIMESTAMP}.csv"
        
        MEASURE_ARGS=(
            "--method" "${method}"
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
        
        if python3 measure.py "${MEASURE_ARGS[@]}"; then
            echo "✅ Benchmark completed"
            RESULT_PROFILES+=("${method}/${ENGINE}")
            SUCCESSFUL_TESTS=$((SUCCESSFUL_TESTS + 1))
        else
            echo "❌ Benchmark failed"
            FAILED_TESTS+=("${method}/${ENGINE}: Benchmark execution failed")
            FAILED_TESTS_COUNT=$((FAILED_TESTS_COUNT + 1))
        fi
        
        # Stop container
        echo ""
        echo "🛑 Stopping container..."
        python3 docker.py stop --container-name "${CONTAINER_NAME}"
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
        
        echo ""
        echo "✅ Test ${TEST_NUM}/${TOTAL_TESTS} complete"
        echo ""
        
        # Small delay between tests
        sleep 5
    done
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================================================="
echo "🎉 Benchmark Suite Complete!"
echo "=============================================================================="
echo ""
echo "📊 Summary:"
echo "   Total tests: ${TOTAL_TESTS}"
echo "   Successful: ${SUCCESSFUL_TESTS}"
echo "   Failed: ${FAILED_TESTS_COUNT}"
echo ""

if [ ${#RESULT_PROFILES[@]} -gt 0 ]; then
    echo "✅ Successful benchmarks:"
    for profile in "${RESULT_PROFILES[@]}"; do
        echo "   - ${profile}"
    done
    echo ""
    echo "Results saved in: ${OUTPUT_DIR}/"
    echo ""
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "❌ Failed tests:"
    for failed in "${FAILED_TESTS[@]}"; do
        echo "   - ${failed}"
    done
    echo ""
fi

echo "Key metrics to analyze:"
echo "   - Time to First Token (TTFT)"
echo "   - Inter-Token Latency (ITL)"
echo "   - Request Latency"
echo "   - Throughput (tokens/sec)"
echo ""
echo "Example analysis:"
echo "   import pandas as pd"
echo "   df = pd.read_csv('${OUTPUT_DIR}/benchmark_*.csv')"
echo "   df.groupby(['method', 'engine'])[['request_throughput_avg', 'ttft_avg']].mean()"
echo ""
echo "✅ Done!"
echo ""

