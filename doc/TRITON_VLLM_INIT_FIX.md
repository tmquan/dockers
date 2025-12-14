# Triton vLLM Backend Initialization Fix

## Error

```
RuntimeError: Engine core initialization failed. 
Failed core proc(s): {}
At: /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/...
```

## Root Cause

The Triton vLLM backend is failing to initialize the engine core. This is likely due to one of:

1. **Incompatible vLLM parameters** in `model.json`
2. **Model context length** exceeding GPU memory
3. **vLLM v1 API changes** (newer Triton uses vLLM v1)
4. **GPU memory exhaustion**

## Investigation

### Check Current Configuration

The current `model.json` has these minimal parameters:
```json
{
  "model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
  "dtype": "auto",
  "max_model_len": 30000,
  "gpu_memory_utilization": 0.9,
  "trust_remote_code": true
}
```

### Possible Issues

1. **Max Model Length**: 30000 tokens for a 30B model might be too large
2. **GPU Memory**: 0.9 utilization might be too aggressive
3. **vLLM v1 Changes**: The error path shows `vllm/v1/engine/` which is the new vLLM v1 API

## Solutions to Try

### Solution 1: Reduce Max Model Length ✅ **Try First**

```python
# In docker.py start command:
--max-model-len 16384  # Instead of 30000
```

Or in run_one.sh:
```bash
python docker.py start \
    --method triton \
    --model MODEL \
    --engine vllm \
    --max-model-len 16384 \  # Reduced
    --gpu-memory 0.85        # Slightly reduced
```

### Solution 2: Reduce GPU Memory Utilization

```python
# In docker.py:
--gpu-memory 0.85  # Or even 0.8
```

### Solution 3: Use HF Method Instead

The HF vLLM deployment works without Triton complexity:
```bash
./run_one.sh hf vllm  # This works!
```

### Solution 4: Check Container Logs

```bash
# Get detailed error from container
python docker.py logs --container-name CONTAINER_NAME

# Or
docker logs qwen3-30b-a3b-thinking-2507-triton-vllm
```

## Recommended Quick Fix

**For immediate testing**, use these settings:

```bash
# Stop existing container
python docker.py stop --container-name qwen3-30b-a3b-thinking-2507-triton-vllm

# Start with reduced parameters
python docker.py start \
    --method triton \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --openai-frontend \
    --openai-port 9000 \
    --max-model-len 16384 \
    --gpu-memory 0.85
```

## Debugging Commands

```bash
# Check GPU memory
nvidia-smi

# Check if model is too large
python3 << 'EOF'
# Rough estimate: 30B model needs ~60GB in FP16
# With 30k context: additional ~10-20GB
# Total: ~70-80GB minimum
print("30B model + 30k context needs ~70-80GB GPU memory")
EOF

# Check container logs for OOM
docker logs qwen3-30b-a3b-thinking-2507-triton-vllm 2>&1 | grep -i "out of memory"
```

## Alternative: Use HF Method

The HF method with vLLM works reliably:

```bash
# This works!
python docker.py start \
    --method hf \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --max-model-len 30000 \
    --gpu-memory 0.9

# Then benchmark
python measure.py \
    --method hf \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --endpoint http://localhost:8000 \
    --input-tokens-mean 28000 \
    --output-tokens-mean 200
```

## Understanding the Difference

| Method | vLLM Location | Pros | Cons |
|--------|---------------|------|------|
| HF | Native vLLM container | Simpler, more stable | No Triton features |
| Triton | vLLM as Triton backend | Multi-model, features | More complex, picky |

## Triton-Specific Issues

Triton vLLM backend has additional requirements:

1. **Memory overhead**: Triton adds ~5-10% overhead
2. **Model repository**: Must be correctly structured
3. **Config validation**: Stricter parameter checking
4. **vLLM version**: Triton bundles specific vLLM version

## Recommended Approach

**For 30B model with 30k context:**

```bash
# Option A: HF method (recommended for now)
./run_one.sh hf vllm

# Option B: Triton with reduced context
python docker.py start \
    --method triton \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --max-model-len 8192 \  # Start small
    --gpu-memory 0.85 \
    --openai-frontend
```

## Memory Requirements

| Model Size | Context | FP16 Memory | Recommended GPU |
|------------|---------|-------------|-----------------|
| 30B | 8k | ~65GB | A100 80GB |
| 30B | 16k | ~75GB | A100 80GB |
| 30B | 30k | ~90GB | 2x A100 80GB or H100 |

## Next Steps

1. **Check GPU memory**: `nvidia-smi`
2. **Reduce context length**: Start with 8k or 16k
3. **Check logs**: `docker logs CONTAINER_NAME`
4. **Try HF method**: Known to work
5. **Report findings**: What worked/didn't work

## Quick Test Script

```bash
#!/bin/bash
echo "Testing Triton with reduced settings..."

# Clean up
python docker.py stop --container-name qwen3-30b-a3b-thinking-2507-triton-vllm 2>/dev/null
docker rm -f qwen3-30b-a3b-thinking-2507-triton-vllm 2>/dev/null

# Start with minimal settings
python docker.py start \
    --method triton \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --max-model-len 8192 \
    --gpu-memory 0.8 \
    --openai-frontend \
    --openai-port 9000

# Wait and check
sleep 60
python docker.py status --container-name qwen3-30b-a3b-thinking-2507-triton-vllm

# If fails, show logs
if [ $? -ne 0 ]; then
    echo "Failed! Showing logs..."
    docker logs qwen3-30b-a3b-thinking-2507-triton-vllm
fi
```

## Summary

✅ **Immediate action**: Reduce `--max-model-len` to 8192 or 16384
✅ **Alternative**: Use HF method which works reliably
✅ **Long-term**: Investigate GPU memory requirements for 30k context
⚠️ **Note**: 30B model + 30k context likely needs 2x A100 80GB or H100

