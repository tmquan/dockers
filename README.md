# Unified LLM Deployment and Benchmarking Suite

This suite provides a unified framework for deploying and benchmarking Large Language Models (LLMs) across multiple deployment methods and inference engines.


### Core Components

1. **`docker.py`** - Unified deployment manager with abstract base class
   - `BaseModelDeployer` - Abstract base class for all deployers
   - `HFModelDeployer` - Direct HuggingFace model deployment
   - `TritonModelDeployer` - NVIDIA Triton Inference Server deployment
   - `NIMModelDeployer` - NVIDIA NIM deployment (to be implemented)
   - `UNIMModelDeployer` - Universal NIM deployment (✅ Implemented)

2. **`measure.py`** - Performance measurement using GenAI-Perf
   - Supports synthetic input generation (for long contexts)
   - Supports input file (JSONL format)
   - Exports detailed metrics to CSV

3. **`run_one.sh`** - Single test runner for quick validation
4. **`run_all.sh`** - Complete benchmark suite runner

## Supported Configurations

### Deployment Methods

| Method   | Status       | Description                                    | Engines Supported                    |
|----------|--------------|------------------------------------------------|--------------------------------------|
| `hf`     | ✅ Ready     | Direct HuggingFace model deployment            | vllm, trtllm, sglang                |
| `nim`    | 🚧 Planned   | NVIDIA NIM (optimized containers)              | vllm (planned)                      |
| `unim`   | ✅ Ready     | Universal NIM (HuggingFace Safetensors)        | vllm, trtllm, sglang, python        |
| `triton` | ✅ Ready     | NVIDIA Triton Inference Server                 | vllm, trtllm*                       |

*Note: `triton/trtllm` requires pre-built TensorRT-LLM engines*

### Inference Engines

| Engine   | Methods                    | Description                                    | Status                    |
|----------|----------------------------|------------------------------------------------|---------------------------|
| `vllm`   | hf, nim, unim, triton      | Fast and flexible with PagedAttention          | ✅ Production Ready       |
| `trtllm` | hf, unim, triton*          | NVIDIA TensorRT-LLM (maximum performance)      | ✅ Ready (auto-converts)  |
| `sglang` | hf, unim                   | Optimized for structured generation            | ✅ Production Ready       |
| `python` | unim                       | Python backend (safetensors, baseline)         | ✅ Baseline               |

*Note: `triton/trtllm` requires pre-built engines*

## Quick Start

### Prerequisites

```bash
# Install Docker and NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y docker.io nvidia-docker2
sudo systemctl restart docker

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export NGC_API_KEY="your_nvidia_ngc_key"  # Optional for NGC images

# Optional: Install genai-perf locally (otherwise uses Docker)
pip install genai-perf
```

### Single Test (Quick Validation)

```bash
# Test HuggingFace + vLLM (fastest)
./run_one.sh hf vllm

# Test NIM (planned)
# ./run_one.sh nim vllm

# Test UNIM with different engines
./run_one.sh unim vllm
./run_one.sh unim trtllm
./run_one.sh unim sglang
./run_one.sh unim python

# Test Triton + vLLM with OpenAI frontend
./run_one.sh triton vllm

# Test with custom model
./run_one.sh hf vllm "meta-llama/Llama-3-8B"

# Test with custom port
./run_one.sh hf vllm "Qwen/Qwen3-30B-A3B-Thinking-2507" 8001
```

### Full Benchmark Suite

```bash
# Run all methods and engines
./run_all.sh

# Results will be saved in artifacts/ directory
# Logs are saved to run_all.log
```

## Performance Benchmarks

### Benchmark Results Summary

*Test Configuration: Qwen/Qwen3-30B-A3B-Thinking-2507, 30k input tokens, 3k output tokens, 40 concurrency, 1000 requests*

| Method/Engine | TTFT (ms) | ITL (ms) | Throughput (tokens/s) | Request Latency (ms) | Status |
|---------------|-----------|---------|------------------------|----------------------|--------|
| **hf/vllm**   | 61,955    | 43.78   | 556.01                 | 193,240              | ✅     |
| **hf/trtllm** | 72,550    | 38.15   | 568.41                 | 183,762              | ✅     |
| **hf/sglang** | 70,545    | 50.56   | 500.59                 | 222,167              | ✅     |
| **nim/vllm**  | -         | -       | -                      | -                    | 🚧     |
| **unim/vllm** | 74,294    | 44.79   | 515.28                 | 208,507              | ✅     |
| **unim/trtllm** | 84,630  | 39.23   | 526.30                 | 199,040              | ✅     |
| **unim/sglang** | 86,091   | 49.08   | 459.74                 | 233,223              | ✅     |
| **unim/python** | 84,988   | 39.54   | 523.30                 | 200,046              | ✅     |
| **triton/vllm** | 60,661   | 50.67   | 491.40                 | 212,577              | ✅     |
| **triton/trtllm** | -      | -       | -                      | -                    | ⚠️*    |

*Requires pre-built TensorRT-LLM engines*

**Key Observations:**
- **Best TTFT**: `triton/vllm` (60.7s) - Fastest time to first token
- **Best Throughput**: `hf/trtllm` (568 tokens/s) - Highest token generation rate
- **Best ITL**: `hf/trtllm` (38.15ms) - Lowest inter-token latency
- **Best Request Latency**: `unim/trtllm` (199s) - Fastest end-to-end request completion

**Performance Notes:**
- TensorRT-LLM (`trtllm`) generally provides the best throughput and lowest ITL
- vLLM provides good balance across all metrics
- SGLang shows higher ITL but good for structured generation use cases
- UNIM Python backend performs surprisingly well, comparable to optimized engines
- Triton adds slight overhead but provides enterprise features (model management, batching, etc.)

*Full benchmark results available in `run_all.log` and `artifacts/` directory*

## Usage Examples

### 1. Deploy a Model

```bash
# HuggingFace with vLLM
python docker.py start \
    --method hf \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --port 8000 \
    --gpu-memory 0.9 \
    --max-model-len 32768

# Triton with vLLM + OpenAI frontend
python docker.py start \
    --method triton \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --openai-frontend \
    --openai-port 9000 \
    --max-model-len 32768

# With tensor parallelism (8 GPUs)
python docker.py start \
    --method hf \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --tp-size 8 \
    --port 8000
```

### 2. Check Status

```bash
python docker.py status --container-name hf-qwen3-30b-vllm
```

### 3. View Logs

```bash
# View logs
python docker.py logs --container-name hf-qwen3-30b-vllm

# Follow logs (real-time)
python docker.py logs --container-name hf-qwen3-30b-vllm -f
```

### 4. Stop Container

```bash
python docker.py stop --container-name hf-qwen3-30b-vllm
```

### 5. Benchmark Performance

```bash
# With synthetic input (32k context) - count-based
python measure.py \
    --method hf \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --endpoint http://localhost:8000 \
    --input-tokens-mean 30000 \
    --output-tokens-mean 200 \
    --concurrency 40 \
    --request-count 1000 \
    --streaming \
    --output-file results.csv

# Time-based measurement (alternative)
python measure.py \
    --method hf \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --endpoint http://localhost:8000 \
    --input-tokens-mean 30000 \
    --output-tokens-mean 200 \
    --concurrency 40 \
    --measurement-interval 10000 \
    --streaming \
    --output-file results.csv
# Note: Use EITHER --request-count OR --measurement-interval, not both

# With input file
python measure.py \
    --method hf \
    --model meta-llama/Llama-3-8B \
    --engine vllm \
    --endpoint http://localhost:8000 \
    --input-file prompts.jsonl \
    --output-file results.csv
```

## Configuration

### Environment Variables

```bash
# Required for private models
export HF_TOKEN="your_huggingface_token"

# Optional for NVIDIA NGC images
export NGC_API_KEY="your_nvidia_ngc_key"
export NVIDIA_API_KEY="your_nvidia_api_key"
```

### Default Values

All defaults are configurable at the top of each file:

**docker.py:**
```python
DEFAULT_PORT = 8000
DEFAULT_GPU_MEMORY = 0.9
DEFAULT_METHOD = "hf"
DEFAULT_ENGINE = "vllm"
DEFAULT_MAX_MODEL_LEN = 32768
```

**measure.py:**
```python
DEFAULT_CONCURRENCY = 40
DEFAULT_REQUEST_COUNT = 1000
DEFAULT_INPUT_SEQUENCE_LENGTH = 32768
DEFAULT_OUTPUT_SEQUENCE_LENGTH = 200
DEFAULT_STREAMING = True
```

## Output Format

### Benchmark Results (CSV)

Each benchmark produces a CSV file with the following metrics:

| Metric Category       | Key Metrics                                           |
|-----------------------|-------------------------------------------------------|
| Request Metrics       | throughput_avg, latency_avg, latency_p50/p95/p99    |
| Time to First Token   | ttft_avg, ttft_p50, ttft_p95, ttft_p99              |
| Inter-Token Latency   | itl_avg, itl_p50, itl_p95, itl_p99                  |
| Token Throughput      | output_token_throughput_avg                          |
| Sequence Lengths      | input/output_seq_len_avg/p50/p95                    |
| Environment Info      | GPU, CUDA, drivers, platform details                 |

### Analysis Example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('artifacts/benchmark_*.csv')

# Compare methods and engines
summary = df.groupby(['method', 'engine']).agg({
    'request_throughput_avg': 'mean',
    'ttft_avg': 'mean',
    'inter_token_latency_avg': 'mean',
    'output_token_throughput_avg': 'mean'
}).round(2)

print(summary)

# Plot comparison
summary['output_token_throughput_avg'].plot(kind='bar', title='Token Throughput Comparison')
plt.ylabel('Tokens/sec')
plt.tight_layout()
plt.savefig('throughput_comparison.png')
```

## Architecture Details

### Class Hierarchy

```
BaseModelDeployer (ABC)
├── HFModelDeployer ✅
│   ├── Supports: vllm, trtllm, sglang
│   └── Direct engine deployment
├── NIMModelDeployer 🚧
│   └── NVIDIA optimized containers (planned)
├── UNIMModelDeployer ✅
│   ├── Supports: vllm, trtllm, sglang, python
│   ├── HuggingFace Safetensors models
│   └── Universal NIM wrapper
└── TritonModelDeployer ✅
    ├── Supports: vllm, trtllm*
    ├── Creates model repository
    ├── Generates config.pbtxt
    └── Optional OpenAI frontend
```

### Key Methods

All deployers implement:
- `start()` - Start the container
- `stop()` - Stop the container
- `restart()` - Restart the container
- `status()` - Check container status
- `logs(follow=False)` - View container logs

## Troubleshooting

### Container fails to start

```bash
# Check Docker and NVIDIA runtime
docker info | grep -i nvidia

# Check GPU availability
nvidia-smi

# Check logs
python docker.py logs --container-name CONTAINER_NAME
```

### Benchmark fails

```bash
# Check endpoint accessibility
curl http://localhost:8000/v1/models

# Check genai-perf installation
genai-perf --version

# Run with verbose output
python measure.py --method hf --model MODEL --engine vllm --endpoint URL --input-tokens-mean 1000 --output-tokens-mean 100
```

### Port already in use

```bash
# Check what's using the port
sudo lsof -i :8000

# Use a different port
python docker.py start --method hf --model MODEL --engine vllm --port 8001
```

## Performance Tips

1. **GPU Memory**: Adjust `--gpu-memory` based on model size
   - Small models (<10B): 0.9
   - Large models (30B+): 0.8-0.85

2. **Tensor Parallelism**: Use `--tp-size` for multi-GPU
   - 2-4 GPUs for 30B models
   - 8 GPUs for 70B+ models

3. **Context Length**: Adjust `--max-model-len` based on use case
   - Short context: 4096-8192
   - Medium context: 16384-32768
   - Long context: 65536-131072

4. **Benchmarking**: Use appropriate concurrency
   - Low concurrency (1-10): Latency testing
   - Medium concurrency (20-50): Balanced testing
   - High concurrency (100+): Throughput testing

## Future Enhancements

### High Priority
- [ ] Implement NIM deployment (`nim/vllm`)
- [ ] Fix Triton TensorRT-LLM backend (requires pre-built engine support)
- [ ] Add automatic GPU count detection and TP size recommendations

### Medium Priority
- [ ] Add support for multi-node deployment
- [ ] Add model quantization options (INT4, INT8, FP8)
- [ ] Add batch processing support
- [ ] Add dynamic batching configuration

### Low Priority
- [ ] Add cost estimation (per-token pricing)
- [ ] Add performance regression testing
- [ ] Add support for additional engines (TensorRT-LLM native, etc.)
- [ ] Add Kubernetes deployment support

## Contributing

To add a new deployment method:

1. Create a new class inheriting from `BaseModelDeployer`
2. Implement all abstract methods
3. Add to `SUPPORTED_METHODS` and factory function
4. Update documentation

To add a new engine:

1. Add engine to `SUPPORTED_ENGINES` for appropriate methods
2. Implement `_build_<engine>_command()` in relevant deployers
3. Update documentation

## License

[Your License Here]

## Support

For issues, questions, or contributions, please open an issue on the repository.

---

## 📖 Additional Documentation

All detailed documentation is available in the [`doc/`](doc/) folder:

- [Quick Start Guide](doc/QUICKSTART.md) - Get started in 5 minutes
- [Quick Reference](doc/QUICK_REFERENCE.md) - Command cheat sheet  
- [Architecture Guide](doc/ARCHITECTURE.md) - System design and diagrams
- [Refactoring Summary](doc/REFACTORING_SUMMARY.md) - What changed and why
- [Documentation Index](doc/INDEX.md) - Complete documentation list

**First time here?** Start with the [Quick Start Guide](doc/QUICKSTART.md)!
