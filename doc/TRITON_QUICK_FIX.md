# Quick Fix for Triton vLLM Initialization

## The Problem

30B model + 30k context = **~70-90GB GPU memory required**

Most systems don't have this much VRAM, causing the engine initialization to fail.

## Quick Solutions

### Solution 1: Reduce Context Length (Recommended)

Edit `run_one.sh`:
```bash
# Change this line:
INPUT_SEQUENCE_LENGTH=32768   # 32k tokens context (max model length)

# To:
INPUT_SEQUENCE_LENGTH=16384   # 16k tokens context (more reasonable)
```

Then run:
```bash
./run_one.sh triton vllm
```

### Solution 2: Use HF Method Instead (Easiest)

```bash
# This works without issues!
./run_one.sh hf vllm
```

The HF method is simpler and more stable than Triton for single-model deployments.

### Solution 3: Manual Start with Reduced Settings

```bash
# Stop any existing container
python docker.py stop --container-name qwen3-30b-a3b-thinking-2507-triton-vllm 2>/dev/null

# Start with reduced context
python docker.py start \
    --method triton \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --max-model-len 16384 \
    --gpu-memory 0.85 \
    --openai-frontend \
    --openai-port 9000

# Wait for startup (2-3 minutes)
sleep 180

# Check status
python docker.py status --container-name qwen3-30b-a3b-thinking-2507-triton-vllm
```

## Why This Happens

| Component | Memory Usage |
|-----------|--------------|
| 30B model (FP16) | ~60GB |
| 30k KV cache | ~10-30GB |
| Triton overhead | ~5GB |
| **Total** | **~75-95GB** |

Most GPUs:
- A100 40GB: ❌ Not enough
- A100 80GB: ⚠️ Barely enough (need to reduce context)
- 2x A100 80GB: ✅ Works with tensor parallelism
- H100: ✅ Works

## Recommended Settings by GPU

### Single A100 80GB
```bash
--max-model-len 16384  # 16k context
--gpu-memory 0.85
```

### 2x A100 80GB
```bash
--max-model-len 32768  # 32k context
--gpu-memory 0.9
--tp-size 2
```

### Testing Which Works

```bash
# Test increasing context lengths
for ctx in 8192 16384 24576 32768; do
    echo "Testing context length: $ctx"
    
    python docker.py stop --container-name test-triton 2>/dev/null
    
    python docker.py start \
        --method triton \
        --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
        --engine vllm \
        --max-model-len $ctx \
        --gpu-memory 0.85 \
        --container-name test-triton \
        --openai-frontend
    
    sleep 120
    
    if python docker.py status --container-name test-triton | grep -q "Ready"; then
        echo "✅ Context length $ctx works!"
        break
    else
        echo "❌ Context length $ctx failed"
    fi
done
```

## TL;DR

**Just use HF method for now:**
```bash
./run_one.sh hf vllm  # This works!
```

**Or reduce context in Triton:**
- Change `INPUT_SEQUENCE_LENGTH` from 32768 to 16384
- Then `./run_one.sh triton vllm`

