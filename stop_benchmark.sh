#!/bin/bash

# Stop running benchmark and cleanup
# Usage: ./stop_benchmark.sh

echo "=============================================================================="
echo "🛑 Stopping Benchmark and Cleaning Up"
echo "=============================================================================="
echo ""

# Kill running benchmark processes
echo "📋 Stopping benchmark processes..."
MEASURE_PIDS=$(pgrep -f "measure.py")
if [ ! -z "${MEASURE_PIDS}" ]; then
    echo "   Found measure.py processes: ${MEASURE_PIDS}"
    pkill -f "measure.py"
    echo "   ✅ Stopped measure.py"
fi

RUNALL_PIDS=$(pgrep -f "run_all.sh")
if [ ! -z "${RUNALL_PIDS}" ]; then
    echo "   Found run_all.sh processes: ${RUNALL_PIDS}"
    pkill -f "run_all.sh"
    echo "   ✅ Stopped run_all.sh"
fi

sleep 2

# Stop and remove Docker containers
echo ""
echo "🐳 Stopping Docker containers..."
CONTAINER_PATTERNS=("-hf-" "-triton-" "-nim-" "-unim-")

for pattern in "${CONTAINER_PATTERNS[@]}"; do
    EXISTING=$(docker ps -a --filter "name=${pattern}" --format "{{.Names}}" 2>/dev/null || true)
    if [ ! -z "${EXISTING}" ]; then
        echo "   Found containers with pattern '${pattern}':"
        echo "${EXISTING}" | while read container; do
            echo "     - ${container}"
        done
        docker ps -a --filter "name=${pattern}" --format "{{.Names}}" | xargs -r docker stop 2>/dev/null || true
        docker ps -a --filter "name=${pattern}" --format "{{.Names}}" | xargs -r docker rm 2>/dev/null || true
        echo "   ✅ Cleaned up containers"
    fi
done

# Check GPU status
echo ""
echo "📊 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -n 1

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "To restart with optimized settings:"
echo "   ./run_all.sh"
echo ""


