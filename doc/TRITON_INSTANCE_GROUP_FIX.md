# CRITICAL FIX: Triton Instance Group Configuration

## The Problem

**Error:** `Engine core initialization failed. Failed core proc(s): {}`

**Root Cause:** Triton was creating **8 separate vLLM engine instances** (one per GPU) instead of one unified instance.

## What Was Happening

### Logs showed:
```
I1214 03:52:08 "TRITONBACKEND_ModelInstanceInitialize: Qwen_..._0_0 (GPU device 0)"
I1214 03:52:08 "TRITONBACKEND_ModelInstanceInitialize: Qwen_..._0_0 (GPU device 1)"
I1214 03:52:08 "TRITONBACKEND_ModelInstanceInitialize: Qwen_..._0_0 (GPU device 2)"
...
I1214 03:52:08 "TRITONBACKEND_ModelInstanceInitialize: Qwen_..._0_0 (GPU device 7)"
```

**Problem:** 8 separate vLLM engines trying to load the SAME 30B model on 8 different GPUs!
- Each instance thinks it has exclusive GPU access
- 30B model doesn't fit on single GPU
- Engines crash during initialization
- "Failed core proc(s): {}" = all 8 instances failed

## The Root Cause

### In config.pbtxt (WRONG):
```protobuf
instance_group [
  {
    count: 1
    kind: KIND_GPU  # ❌ This creates ONE instance PER GPU!
  }
]
```

With 8 GPUs detected, Triton creates:
- GPU 0: Instance 0 (tries to load 30B model) → Fails
- GPU 1: Instance 1 (tries to load 30B model) → Fails  
- GPU 2: Instance 2 (tries to load 30B model) → Fails
- ... (all fail)

### What We Want:
```protobuf
instance_group [
  {
    count: 1
    kind: KIND_MODEL  # ✅ Creates ONE instance, vLLM manages GPUs
  }
]
```

This creates ONE vLLM instance that:
- Uses vLLM's internal tensor parallelism
- Distributes model across GPUs properly
- Handles multi-GPU coordination correctly

## The Fix

### Changed in `docker.py`:

```python
# OLD (WRONG)
instance_group [
  {{
    count: 1
    kind: KIND_GPU  # Creates instance per GPU
  }}
]

# NEW (CORRECT)
instance_group [
  {{
    count: 1
    kind: KIND_MODEL  # Creates ONE instance total
  }}
]
```

## Why This Matters

### Triton Instance Kinds:

| Kind | Behavior | Use Case |
|------|----------|----------|
| `KIND_GPU` | One instance per GPU | Simple models, data parallelism |
| `KIND_CPU` | One instance per CPU core | CPU inference |
| `KIND_MODEL` | One instance total | Model manages GPUs internally |

### For vLLM Specifically:

vLLM has its own **sophisticated multi-GPU management**:
- Tensor parallelism (split model across GPUs)
- Pipeline parallelism (split layers across GPUs)
- Data parallelism (replicate model)

**vLLM needs exclusive GPU control**, not Triton creating separate instances!

## Tensor Parallelism with vLLM

### How it SHOULD work:

```json
// model.json
{
  "model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
  "tensor_parallel_size": 8,  // vLLM uses all 8 GPUs together
  "max_model_len": 32768,
  ...
}
```

With `KIND_MODEL`:
- Triton creates 1 vLLM instance
- vLLM sees `tensor_parallel_size: 8`
- vLLM distributes model across 8 GPUs
- ✅ Works!

### How it WAS failing:

With `KIND_GPU`:
- Triton creates 8 vLLM instances
- Each instance tries to use 1 GPU exclusively
- Each tries to load full 30B model (doesn't fit!)
- All 8 instances crash
- ❌ Fails!

## Testing the Fix

### 1. Remove old container and model repository
```bash
docker rm -f qwen-qwen3-30b-a3b-thinking-2507-triton-vllm
rm -rf .cache/triton/model_repository/Qwen_Qwen3-30B-A3B-Thinking-2507
```

### 2. Start with updated config
```bash
./run_one.sh triton vllm
```

### 3. Watch logs
```bash
docker logs -f qwen-qwen3-30b-a3b-thinking-2507-triton-vllm
```

**Expected:** Only ONE "TRITONBACKEND_ModelInstanceInitialize" message, not 8!

## For Tensor Parallelism

If you want to use tensor parallelism with 8 GPUs:

```bash
python docker.py start \
    --method triton \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --tp-size 8 \
    --max-model-len 32768 \
    --openai-frontend
```

This will:
- Create 1 Triton instance (`KIND_MODEL`)
- Pass `tensor_parallel_size: 8` to vLLM
- vLLM distributes model across all 8 GPUs
- Much more efficient!

## Summary

✅ **Fixed**: Changed `kind: KIND_GPU` → `kind: KIND_MODEL`
✅ **Result**: Triton creates ONE instance, vLLM manages GPUs
✅ **Benefit**: Proper tensor parallelism support
✅ **Compatible**: Works with vLLM's internal GPU management

**This was the critical fix!** Now test it:

```bash
# Clean start
docker rm -f qwen-qwen3-30b-a3b-thinking-2507-triton-vllm
rm -rf .cache/triton/model_repository/Qwen_Qwen3-30B-A3B-Thinking-2507

# Test
./run_one.sh triton vllm
```

