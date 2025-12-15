# Performance Issues with HF/vLLM Benchmark

## Current Status

**Problem**: The `run_all.sh` script with HF method and vLLM engine is extremely slow and appears to hang.

**Root Causes**:
1. ✅ Container is running and serving requests (no crash)
2. ❌ **Only using 1 GPU out of 8 available H200 GPUs**
3. ❌ **KV cache usage at 100%** - severe memory pressure
4. ❌ **No tensor parallelism** - 30B model should use multiple GPUs
5. ❌ **Overly aggressive benchmark parameters**:
   - 30K input tokens
   - 3K output tokens  
   - 40 concurrent requests
   - 1000 total requests
   - Only ~130GB memory available on single H200

## Current Performance

- **Generation throughput**: ~670-730 tokens/s
- **Requests**: 24 running, 16-18 waiting (queue buildup)
- **GPU KV cache**: 96-100% utilized
- **Time per request**: Very slow due to memory constraints

## Recommended Fixes

### Option 1: Enable Tensor Parallelism (Best for throughput)

Modify `run_all.sh` to use multiple GPUs:

```bash
# Add tensor parallel size for large models
if [[ "${MODEL}" == *"30B"* ]] || [[ "${MODEL}" == *"70B"* ]]; then
    START_ARGS+=("--tp-size" "4")  # Use 4 GPUs
fi
```

This will:
- Distribute model across 4 GPUs
- Increase available KV cache memory 4x
- Significantly improve throughput

### Option 2: Reduce Benchmark Load (Quick fix)

In `run_all.sh`, reduce the benchmark parameters:

```bash
# More realistic benchmark parameters
ACTUAL_INPUT_LEN=8000          # Reduce from 30000
OUTPUT_SEQUENCE_LENGTH=512     # Reduce from 3000
CONCURRENCY=8                  # Reduce from 40
REQUEST_COUNT=100              # Reduce from 1000
```

### Option 3: Increase GPU Memory Fraction

In `run_all.sh`, increase GPU memory allocation:

```bash
DEFAULT_GPU_MEMORY=0.95  # Increase from 0.9
```

### Option 4: Enable Pipeline Parallelism

For very large models, combine tensor and pipeline parallelism:

```bash
START_ARGS+=(
    "--tp-size" "4"
    "--pp-size" "2"  # Pipeline parallelism
)
```

## Recommended Combined Solution

**For your 8x H200 setup with Qwen3-30B model:**

```bash
# In run_all.sh, update these lines:

# More reasonable benchmark parameters
ACTUAL_INPUT_LEN=16000        # 16K input (still stress test)
OUTPUT_SEQUENCE_LENGTH=1024   # 1K output
CONCURRENCY=16                # 16 concurrent requests
REQUEST_COUNT=200             # Faster completion
DEFAULT_GPU_MEMORY=0.95       # Use more GPU memory

# Add TP size selection based on model
if [[ "${MODEL}" == *"30B"* ]]; then
    START_ARGS+=("--tp-size" "4")
elif [[ "${MODEL}" == *"70B"* ]]; then
    START_ARGS+=("--tp-size" "8")
fi
```

## Why It's Not Stopping

The benchmark **IS running** but taking a very long time because:
1. Each request takes 30-60+ seconds due to memory constraints
2. With 1000 requests total, this could take **8-16 hours**
3. The script will eventually finish, but it's impractical

## Immediate Action

You can:

1. **Wait** - It will eventually complete (many hours)
2. **Kill and restart** with better parameters:
   ```bash
   # Stop current run
   docker stop qwen-qwen3-30b-a3b-thinking-2507-hf-vllm
   docker rm qwen-qwen3-30b-a3b-thinking-2507-hf-vllm
   pkill -f "run_all.sh"
   pkill -f "measure.py"
   ```
3. **Modify and rerun** with the fixes above

## Performance Expectations After Fix

With 4-GPU tensor parallelism:
- **Generation throughput**: 2500-3500+ tokens/s (4-5x faster)
- **KV cache**: 25-40% utilized (healthy)
- **Request latency**: Much lower
- **Completion time**: ~10-30 minutes for benchmark

