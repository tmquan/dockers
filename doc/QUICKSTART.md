# Quick Start Guide

Get started with the Unified LLM Deployment Suite in 5 minutes!

## Prerequisites (One-time setup)

```bash
# 1. Install Docker and NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y docker.io nvidia-docker2
sudo systemctl restart docker

# 2. Set your HuggingFace token
export HF_TOKEN="your_huggingface_token_here"

# 3. (Optional) Set NVIDIA NGC key for genai-perf Docker image
export NGC_API_KEY="your_nvidia_ngc_key"

# 4. Navigate to dockers directory
cd /path/to/dockers/
```

## Test 1: Quick Validation (2 minutes)

Run a single quick test with default settings:

```bash
./run_one.sh
```

This will:
- Deploy `Qwen/Qwen3-30B-A3B-Thinking-2507` with `hf` method and `vllm` engine
- Wait for service to be ready
- Run a benchmark with 30k token input, 200 token output
- Save results to `artifacts/benchmark_*.csv`
- Clean up the container

**Expected output:**
```
🚀 Starting hf container with vllm engine...
⏳ Waiting for service to be ready...
✅ Service is ready!
📊 Running benchmark...
✅ Benchmark completed successfully!
```

## Test 2: Compare Methods (5 minutes)

Test both HF direct and Triton deployment:

```bash
# Test HF direct deployment
./run_one.sh hf vllm

# Test Triton with OpenAI frontend
./run_one.sh triton vllm
```

Compare the results:

```bash
# View results
ls -lh artifacts/benchmark_*.csv

# Quick analysis
python3 << 'EOF'
import pandas as pd
import glob

files = glob.glob('artifacts/benchmark_*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print("\n=== Performance Comparison ===\n")
print(df[['method', 'engine', 'request_throughput_avg', 'ttft_avg', 'output_token_throughput_avg']].to_string())
EOF
```

## Test 3: Custom Model (Varies)

Test with your own model:

```bash
./run_one.sh hf vllm "meta-llama/Llama-3-8B" 8000
```

## Test 4: Full Benchmark Suite (30-60 minutes)

Run comprehensive tests across all methods and engines:

```bash
./run_all.sh
```

This will test:
- HF with vLLM
- HF with SGLang
- HF with TensorRT-LLM
- Triton with vLLM
- Triton with Python backend

## Understanding the Results

### Container Names
Format: `{model}-{method}-{engine}`

Example: `qwen3-30b-a3b-thinking-2507-hf-vllm`

### Result Files
Format: `benchmark_{model}_{method}_{engine}_{timestamp}.csv`

Example: `benchmark_qwen3_30b_a3b_thinking_2507_hf_vllm_20241213_143022.csv`

### Key Metrics

Open a result CSV and look for:

| Metric | Description | Better |
|--------|-------------|--------|
| `request_throughput_avg` | Requests per second | Higher |
| `ttft_avg` | Time to first token (ms) | Lower |
| `inter_token_latency_avg` | Time between tokens (ms) | Lower |
| `output_token_throughput_avg` | Output tokens/sec | Higher |

### Quick Analysis

```python
import pandas as pd

# Load your result
df = pd.read_csv('artifacts/benchmark_*.csv')

# Show key metrics
print(f"Throughput: {df['request_throughput_avg'].values[0]:.2f} req/s")
print(f"TTFT: {df['ttft_avg'].values[0]:.2f} ms")
print(f"Token/s: {df['output_token_throughput_avg'].values[0]:.2f}")
```

## Common Operations

### Start a Container

```bash
python docker.py start \
    --method hf \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --port 8000
```

### Check Status

```bash
# List all containers
docker ps -a | grep -E "(hf|triton|nim|unim)"

# Check specific container
python docker.py status --container-name qwen3-30b-a3b-thinking-2507-hf-vllm
```

### View Logs

```bash
# View logs
python docker.py logs --container-name qwen3-30b-a3b-thinking-2507-hf-vllm

# Follow logs (real-time)
python docker.py logs --container-name qwen3-30b-a3b-thinking-2507-hf-vllm -f
```

### Stop Container

```bash
python docker.py stop --container-name qwen3-30b-a3b-thinking-2507-hf-vllm
```

### Manual Benchmark

```bash
# Start container first
python docker.py start --method hf --model MODEL --engine vllm

# Wait for ready (check with status)
python docker.py status --container-name MODEL-hf-vllm

# Run benchmark
python measure.py \
    --method hf \
    --model MODEL \
    --engine vllm \
    --endpoint http://localhost:8000 \
    --input-tokens-mean 10000 \
    --output-tokens-mean 100 \
    --concurrency 20 \
    --request-count 500

# Stop when done
python docker.py stop --container-name MODEL-hf-vllm
```

## Troubleshooting

### Issue: "Container is already running"

```bash
# Stop and remove existing container
docker stop qwen3-30b-a3b-thinking-2507-hf-vllm
docker rm qwen3-30b-a3b-thinking-2507-hf-vllm

# Or use the stop command
python docker.py stop --container-name qwen3-30b-a3b-thinking-2507-hf-vllm
```

### Issue: "Port already in use"

```bash
# Find what's using the port
sudo lsof -i :8000

# Use a different port
./run_one.sh hf vllm MODEL 8001
```

### Issue: "Out of memory"

```bash
# Reduce GPU memory usage
python docker.py start \
    --method hf \
    --model MODEL \
    --engine vllm \
    --gpu-memory 0.8 \
    --max-model-len 16384
```

### Issue: "Model download fails"

```bash
# Verify HF token is set
echo $HF_TOKEN

# Set if not set
export HF_TOKEN="your_token"

# Test token
huggingface-cli whoami
```

### Issue: "Container fails to start"

```bash
# Check Docker
docker info | grep -i nvidia

# Check GPU
nvidia-smi

# View container logs
docker logs qwen3-30b-a3b-thinking-2507-hf-vllm
```

## Next Steps

1. **Explore Documentation**
   - `README.md` - Full documentation
   - `QUICK_REFERENCE.md` - Command reference
   - `ARCHITECTURE.md` - System design

2. **Customize Settings**
   - Edit default configs in `docker.py`
   - Edit benchmark params in `measure.py`
   - Edit test configs in `run_all.sh`

3. **Analyze Results**
   - Use pandas for data analysis
   - Compare methods and engines
   - Identify optimal configuration

4. **Production Deployment**
   - Select best method/engine combo
   - Tune parameters for your workload
   - Monitor with status/logs commands

## Tips for Success

✅ **Always test with `run_one.sh` first** before running the full suite

✅ **Monitor GPU usage** with `nvidia-smi -l 1` in another terminal

✅ **Check logs** if something goes wrong: `docker logs CONTAINER_NAME`

✅ **Clean up** containers after testing to free resources

✅ **Use appropriate concurrency** based on your GPU memory:
- 1 GPU: concurrency 10-40
- 2-4 GPUs: concurrency 40-100
- 8 GPUs: concurrency 100+

✅ **Save important results** - move CSV files to a safe location

## Example Session

```bash
# 1. Quick validation
./run_one.sh
# ✅ Success! Results in artifacts/

# 2. Test another engine
./run_one.sh hf sglang
# ✅ Success! Can compare with vLLM

# 3. Test Triton
./run_one.sh triton vllm
# ✅ Success! See Triton's performance

# 4. Analyze results
python3 << 'EOF'
import pandas as pd
import glob
files = glob.glob('artifacts/benchmark_*.csv')
df = pd.concat([pd.read_csv(f) for f in files])
print(df.groupby(['method', 'engine'])['output_token_throughput_avg'].mean())
EOF

# 5. Run full suite (optional)
./run_all.sh
# ⏱️ This will take 30-60 minutes
```

## Support

For issues or questions:
1. Check the logs: `docker logs CONTAINER_NAME`
2. Read the documentation: `README.md`
3. Review examples: `QUICK_REFERENCE.md`
4. Check troubleshooting: This guide

Happy benchmarking! 🚀

