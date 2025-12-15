# Quick Fix Guide

## What's Happening Now

Your HF/vLLM benchmark is **running but extremely slow** because:

1. ❌ **Only using 1 GPU** instead of multiple GPUs with tensor parallelism
2. ❌ **KV cache at 100%** - severe memory bottleneck  
3. ❌ **Too aggressive parameters**: 30K input, 3K output, 40 concurrent, 1000 requests
4. ⏰ **Estimated completion**: 8-16 hours at current speed

**Current status:**
- Container: Running (42+ minutes)
- Throughput: ~670-730 tokens/s (should be 2500+ tokens/s)
- Requests: 24 running, 16-18 waiting (constant queue)

## Quick Actions

### Option 1: Stop and Restart with Fix (RECOMMENDED)

I've already updated `run_all.sh` with optimizations:
- ✅ Tensor parallelism for 30B models (4 GPUs)
- ✅ Reduced to 16K input, 1K output  
- ✅ Reduced to 16 concurrent, 200 requests
- ✅ Increased GPU memory to 0.95

**Steps:**
```bash
cd /localhome/local-tranminhq/dockers

# Stop current benchmark
pkill -f "run_all.sh"
pkill -f "measure.py"
docker stop qwen-qwen3-30b-a3b-thinking-2507-hf-vllm
docker rm qwen-qwen3-30b-a3b-thinking-2507-hf-vllm

# Restart with optimized settings
./run_all.sh
```

**Expected results:**
- Throughput: 2500-3500+ tokens/s (4-5x faster)
- KV cache: 25-40% (healthy)
- Completion: ~15-30 minutes (vs 8-16 hours)

### Option 2: Just Wait (NOT RECOMMENDED)

The current run will eventually complete, but it will take **8-16 hours**.

## What Changed in run_all.sh

**Before:**
```bash
ACTUAL_INPUT_LEN=30000
OUTPUT_SEQUENCE_LENGTH=3000
CONCURRENCY=40
REQUEST_COUNT=1000
# No tensor parallelism
```

**After:**
```bash
ACTUAL_INPUT_LEN=16000        # 2x less input
OUTPUT_SEQUENCE_LENGTH=1024   # 3x less output
CONCURRENCY=16                # 2.5x less concurrency
REQUEST_COUNT=200             # 5x fewer requests
DEFAULT_TP_SIZE=4             # Use 4 GPUs for 30B models
DEFAULT_GPU_MEMORY=0.95       # More GPU memory
```

Plus automatic TP size selection:
- 30B models: 4 GPUs
- 70B models: 8 GPUs

## Files Updated

1. ✅ `run_all.sh` - Main benchmark script (optimized)
2. ✅ `stop_benchmark.sh` - Helper to stop current run
3. ✅ `PERFORMANCE_ISSUES.md` - Detailed analysis
4. ✅ `QUICK_FIX.md` - This file

## Verification After Restart

After restarting, check:
```bash
# Should show 4 GPUs in use
nvidia-smi

# Should show healthy KV cache (25-40%)
docker logs qwen-qwen3-30b-a3b-thinking-2507-hf-vllm -f

# Expected log output:
# Avg generation throughput: 2500+ tokens/s
# GPU KV cache usage: 25-40%
```

## Still Too Slow?

If still slow after fix, further reduce:
```bash
# Edit run_all.sh
ACTUAL_INPUT_LEN=8000
CONCURRENCY=8
REQUEST_COUNT=100
```


