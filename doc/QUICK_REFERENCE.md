# Quick Reference Guide

## Command Cheat Sheet

### Single Test (Quick Start)

```bash
# Fastest test (HF + vLLM)
./run_one.sh hf vllm

# Test Triton
./run_one.sh triton vllm

# Custom model
./run_one.sh hf vllm "meta-llama/Llama-3-8B"
```

### Full Benchmark

```bash
# Run all tests
./run_all.sh
```

### Manual Deployment

```bash
# Start
python docker.py start --method hf --model MODEL --engine vllm

# Status
python docker.py status --container-name CONTAINER_NAME

# Logs
python docker.py logs --container-name CONTAINER_NAME -f

# Stop
python docker.py stop --container-name CONTAINER_NAME
```

### Manual Benchmark

```bash
# Synthetic input (32k tokens)
python measure.py \
    --method hf \
    --model MODEL \
    --engine vllm \
    --endpoint http://localhost:8000 \
    --input-tokens-mean 30000 \
    --output-tokens-mean 200 \
    --concurrency 40 \
    --request-count 1000
```

## Supported Combinations

### HuggingFace Method (`hf`)

| Engine   | Port | Status | Notes                          |
|----------|------|--------|--------------------------------|
| vllm     | 8000 | ✅     | Recommended, fastest setup     |
| sglang   | 8000 | ✅     | Good for structured generation |
| trtllm   | 8000 | ✅     | Best performance, auto-convert |

### Triton Method (`triton`)

| Engine   | Port | Frontend | Status | Notes                      |
|----------|------|----------|--------|----------------------------|
| vllm     | 9000 | OpenAI   | ✅     | Recommended                |
| python   | 9000 | OpenAI   | ✅     | Baseline for comparison    |
| trtllm   | 9000 | OpenAI   | ⚠️     | Requires pre-built engines |

## Port Mapping

| Method   | Default Port | Service Type      |
|----------|--------------|-------------------|
| hf       | 8000         | OpenAI-compatible |
| triton   | 9000         | OpenAI frontend   |
| triton   | 8000-8002    | Native Triton v2  |

## Common Issues

### Issue: Container won't start

```bash
# Check Docker
docker info | grep -i nvidia

# Check GPU
nvidia-smi

# View logs (use new naming convention)
python docker.py logs --container-name MODEL_SANITIZED-METHOD-ENGINE
```

### Issue: Port already in use

```bash
# Find process using port
sudo lsof -i :8000

# Use different port
./run_one.sh hf vllm MODEL 8001
```

### Issue: Out of memory

```bash
# Reduce GPU memory usage
python docker.py start --method hf --model MODEL --engine vllm --gpu-memory 0.8

# Or reduce max length
python docker.py start --method hf --model MODEL --engine vllm --max-model-len 16384
```

### Issue: Model download fails

```bash
# Set HF token
export HF_TOKEN="your_token"

# Verify token
huggingface-cli whoami
```

## Performance Tuning

### Small Models (<10B parameters)

```bash
python docker.py start \
    --method hf \
    --model MODEL \
    --engine vllm \
    --gpu-memory 0.9 \
    --max-model-len 32768
```

### Large Models (30B+ parameters)

```bash
python docker.py start \
    --method hf \
    --model MODEL \
    --engine vllm \
    --gpu-memory 0.85 \
    --max-model-len 16384
```

### Multi-GPU (70B+ parameters)

```bash
python docker.py start \
    --method hf \
    --model MODEL \
    --engine vllm \
    --tp-size 8 \
    --gpu-memory 0.85 \
    --max-model-len 8192
```

## Benchmark Parameters

### Quick Test (Fast)

```bash
--concurrency 10 \
--request-count 100 \
--input-tokens-mean 1000 \
--output-tokens-mean 100
```

### Standard Test (Balanced)

```bash
--concurrency 40 \
--request-count 1000 \
--input-tokens-mean 30000 \
--output-tokens-mean 200
```

### Time-based Test (Alternative)

```bash
--concurrency 40 \
--measurement-interval 10000 \  # 10 seconds
--input-tokens-mean 30000 \
--output-tokens-mean 200
# Note: Cannot use both --request-count and --measurement-interval
```

### Stress Test (Heavy)

```bash
--concurrency 100 \
--request-count 5000 \
--input-tokens-mean 30000 \
--output-tokens-mean 500
```

## Result Analysis

### Load Results

```python
import pandas as pd

# Single file
df = pd.read_csv('artifacts/benchmark_*.csv')

# Multiple files
import glob
files = glob.glob('artifacts/benchmark_*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
```

### Compare Methods

```python
# Group by method and engine
summary = df.groupby(['method', 'engine']).agg({
    'request_throughput_avg': 'mean',
    'ttft_avg': 'mean',
    'output_token_throughput_avg': 'mean'
}).round(2)

print(summary)
```

### Key Metrics

| Metric                        | Description                    | Goal      |
|-------------------------------|--------------------------------|-----------|
| request_throughput_avg        | Requests per second            | Higher    |
| ttft_avg                      | Time to first token (ms)       | Lower     |
| inter_token_latency_avg       | Time between tokens (ms)       | Lower     |
| output_token_throughput_avg   | Output tokens per second       | Higher    |
| request_latency_p95           | 95th percentile latency (ms)   | Lower     |

## Container Naming Convention

Format: `{model_sanitized}-{method}-{engine}`

Examples:
- `qwen3-30b-a3b-thinking-2507-hf-vllm`
- `meta-llama-3-8b-triton-vllm`
- `mixtral-8x7b-nim-vllm` (future)

## Benchmark Results Naming

Format: `benchmark_{model_sanitized}_{method}_{engine}_{timestamp}.csv`

Examples:
- `benchmark_qwen3_30b_a3b_thinking_2507_hf_vllm_20241213_143022.csv`
- `benchmark_meta_llama_3_8b_triton_vllm_20241213_150033.csv`

## Artifact Directory Naming

Format: `artifacts/{model_sanitized}_{method}_{engine}`

Examples:
- `artifacts/Qwen_Qwen3-30B-A3B-Thinking-2507_hf_vllm/`
- `artifacts/meta-llama_Llama-3-8B_triton_vllm/`

## File Structure

```
dockers/
├── docker.py              # Main deployment manager
├── measure.py             # Performance measurement
├── run_one.sh             # Single test runner
├── run_all.sh             # Full benchmark suite
├── README.md              # Full documentation
├── QUICK_REFERENCE.md     # This file
├── .cache/                # Model cache
│   ├── hf/                # HuggingFace models
│   ├── triton/            # Triton model repos
│   ├── nim/               # NIM models (future)
│   └── unim/              # UNIM models (future)
└── artifacts/             # Benchmark results
    └── benchmark_*.csv    # CSV result files
```

## Environment Setup

```bash
# Create .env file
cat > .env << 'EOF'
HF_TOKEN=your_huggingface_token
NGC_API_KEY=your_nvidia_ngc_key
NVIDIA_API_KEY=your_nvidia_api_key
EOF

# Load variables
source .env
```

## Best Practices

1. **Always test with `run_one.sh` first** before running full suite
2. **Monitor GPU memory** with `nvidia-smi -l 1`
3. **Check logs** if container fails to start
4. **Use appropriate concurrency** based on GPU memory
5. **Save results** with descriptive filenames
6. **Clean up** containers after benchmarking

## Quick Examples

### Example 1: Test vLLM performance

```bash
./run_one.sh hf vllm
```

### Example 2: Compare HF vs Triton

```bash
./run_one.sh hf vllm
./run_one.sh triton vllm
```

### Example 3: Test all HF engines

```bash
for engine in vllm sglang trtllm; do
    ./run_one.sh hf $engine
done
```

### Example 4: Custom benchmark

```bash
# Start container
python docker.py start --method hf --model MODEL --engine vllm --port 8000

# Wait for ready
sleep 60

# Run custom benchmark
python measure.py \
    --method hf \
    --model MODEL \
    --engine vllm \
    --endpoint http://localhost:8000 \
    --input-tokens-mean 10000 \
    --output-tokens-mean 500 \
    --concurrency 20 \
    --request-count 500

# Stop
python docker.py stop --container-name hf-model-vllm
```

## Tips

- Use `--streaming` for realistic latency measurements
- Use `--no-streaming` for maximum throughput tests
- Increase `--warmup-request-count` for more stable results
- Monitor GPU utilization during benchmarks
- Run multiple iterations for statistical significance
- Compare similar configurations (same model, context length)

---

For detailed documentation, see [README.md](README.md)

