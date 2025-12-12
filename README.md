# HuggingFace Model Deployment and Performance Benchmarking

This repository contains tools for deploying HuggingFace models using various inference engines and measuring their performance using genai-perf.

## Components

### 1. `docker_hf.py` - Model Deployment Manager

A generic Docker deployment script that supports multiple inference engines for serving HuggingFace models.

#### Supported Engines
- **vLLM** - High-performance inference with PagedAttention
- **TensorRT-LLM** - NVIDIA's optimized inference engine
- **SGLang** - Fast serving with RadixAttention

#### Features
- Automatic container management (start, stop, restart, status, logs)
- Multi-engine support with engine-specific optimizations
- Automatic container naming based on model
- HuggingFace token support for private models
- Configurable GPU memory utilization
- Tensor parallelism support
- Custom context length settings

#### Usage

```bash
# Start a model with vLLM (default engine)
python docker_hf.py start --model Qwen/Qwen3-30B-A3B-Thinking-2507

# Start with specific engine
python docker_hf.py start --model meta-llama/Llama-3-8B --engine sglang

# Start with custom settings
python docker_hf.py start \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --engine vllm \
  --port 8001 \
  --gpu-memory 0.9 \
  --tp-size 2 \
  --max-model-len 262144 \
  --container-name qwen3-30b-a3b-thinking-2507-vllm

# Check status
python docker_hf.py status --container-name qwen3-30b-a3b-thinking-2507-vllm

# View logs
python docker_hf.py logs --container-name qwen3-30b-a3b-thinking-2507-vllm -f

# Stop container
python docker_hf.py stop --container-name qwen3-30b-a3b-thinking-2507-vllm

# Restart with same settings
python docker_hf.py restart \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --engine vllm
```

#### Arguments

**Start Command:**
- `--model`: HuggingFace model name (required)
- `--engine`: Inference engine (`vllm`, `tensorrt-llm`, `sglang`) [default: vllm]
- `--port`: Port to expose [default: 8000]
- `--cache-dir`: Cache directory for model weights [default: ~/.cache/huggingface]
- `--gpu-memory`: GPU memory utilization fraction [default: 0.85]
- `--tp-size`: Tensor parallel size for multi-GPU [default: 1]
- `--max-model-len`: Maximum model context length
- `--container-name`: Custom container name
- `--extra-args`: Additional engine-specific arguments

**Other Commands:**
- `stop`, `status`, `logs`: Require `--container-name`
- `restart`: Same arguments as `start`

### 2. `measure_perf.py` - Performance Benchmarking Tool

Measures model performance using genai-perf and exports detailed metrics to CSV.

#### Features
- Comprehensive performance metrics collection
- CSV export with standardized format
- Support for chat and completion endpoints
- Configurable concurrency and request counts
- Streaming support
- Automatic endpoint health checking

#### Metrics Collected

- **Throughput**: request throughput (req/s), output token throughput (tokens/s)
- **Latency**: avg, p50, p95, p99, min, max, std (ms)
- **TTFT**: Time to First Token - avg, p50, p95 (ms)
- **Sequence Lengths**: input/output avg, p50, p95, min, max (tokens)
- **Configuration**: model, backend, concurrency, streaming, etc.

#### Usage

```bash
# Prerequisites: Install genai-perf
pip install genai-perf

# Basic benchmark
python measure_perf.py \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --backend vllm \
  --endpoint http://localhost:8000/v1 \
  --input input.jsonl \
  --output results.csv

# Custom concurrency and request count
python measure_perf.py \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --backend vllm \
  --endpoint http://localhost:8000/v1 \
  --input input.jsonl \
  --output results.csv \
  --concurrency 100 \
  --request-count 500

# With streaming enabled
python measure_perf.py \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --backend vllm \
  --endpoint http://localhost:8000/v1 \
  --input input.jsonl \
  --output results.csv \
  --streaming
```

#### Arguments

- `--model`: Model name (required)
- `--backend`: Backend/engine name (required, e.g., vllm, tensorrt-llm, sglang)
- `--endpoint`: API endpoint URL (required, e.g., http://localhost:8000/v1)
- `--input`: Input JSONL file with prompts (required)
- `--output`: Output CSV file for metrics (required)
- `--concurrency`: Number of concurrent requests [default: 40]
- `--request-count`: Total number of requests [default: 100]
- `--endpoint-type`: Endpoint type (`chat`, `completions`, `embeddings`) [default: chat]
- `--streaming`: Enable streaming mode
- `--extra-args`: Additional genai-perf arguments

## Complete Workflow Example

### 1. Deploy Qwen3-30B with vLLM

```bash
# Set HuggingFace token if needed
export HF_TOKEN=your_token_here

# Start the model
python docker_hf.py start \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --engine vllm \
  --port 8000 \
  --max-model-len 262144 \
  --gpu-memory 0.85

# Wait for model to load (check logs)
python docker_hf.py logs --container-name qwen3-30b-a3b-thinking-2507-vllm -f

# Check status
python docker_hf.py status --container-name qwen3-30b-a3b-thinking-2507-vllm
```

### 2. Run Performance Benchmark

```bash
# Run benchmark with input.jsonl
python measure_perf.py \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --backend vllm \
  --endpoint http://localhost:8000/v1 \
  --input input.jsonl \
  --output qwen_vllm_results.csv \
  --concurrency 40 \
  --request-count 100
```

### 3. Compare Multiple Engines

```bash
# Deploy with different engines on different ports
python docker_hf.py start --model Qwen/Qwen3-30B-A3B-Thinking-2507 --engine vllm --port 8000
python docker_hf.py start --model Qwen/Qwen3-30B-A3B-Thinking-2507 --engine sglang --port 8001

# Benchmark each engine
python measure_perf.py \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --backend vllm \
  --endpoint http://localhost:8000/v1 \
  --input input.jsonl \
  --output comparison_results.csv

python measure_perf.py \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --backend sglang \
  --endpoint http://localhost:8001/v1 \
  --input input.jsonl \
  --output comparison_results.csv

# View combined results
cat comparison_results.csv
```

### 4. Cleanup

```bash
# Stop containers
python docker_hf.py stop --container-name qwen3-30b-a3b-thinking-2507-vllm
python docker_hf.py stop --container-name qwen3-30b-a3b-thinking-2507-sglang
```

## Input File Format

The `input.jsonl` file should contain prompts in JSONL format. For chat endpoints:

```jsonl
{"text": "Your prompt text here"}
{"text": "Another prompt"}
```

For completion endpoints:

```jsonl
{"prompt": "Your prompt text here"}
{"prompt": "Another prompt"}
```

## Output CSV Format

The performance metrics are exported to CSV with the following columns:

```
model, backend, endpoint_type, concurrency, request_throughput_avg, 
request_latency_avg, request_latency_p50, request_latency_p95, 
request_latency_p99, ttft_avg, output_token_throughput_avg, 
output_seq_len_avg, input_seq_len_avg, request_count, 
request_throughput_unit, request_latency_min, request_latency_max, 
request_latency_std, request_latency_unit, ttft_p50, ttft_p95, 
ttft_unit, output_token_throughput_unit, output_seq_len_p50, 
output_seq_len_p95, output_seq_len_min, output_seq_len_max, 
output_seq_len_unit, input_seq_len_p50, input_seq_len_p95, 
input_seq_len_min, input_seq_len_max, input_seq_len_unit, 
service_kind, streaming, concurrency_from_config, measurement_mode, 
measurement_num
```

## Requirements

### Docker Requirements
- Docker installed and running
- NVIDIA Docker runtime (nvidia-docker2)
- Sufficient GPU memory for the model

### Python Requirements
- Python 3.10+
- genai-perf (for benchmarking)
- Docker Python SDK
- HuggingFace libraries

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Run the setup script
./setup_env.sh

# Or manually create the environment
conda env create -f environment.yml
conda activate deploy

# Verify installation
python docker_hf.py --help
python measure_perf.py --help
```

#### Option 2: Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python docker_hf.py --help
python measure_perf.py --help
```

#### Option 3: Using Triton SDK container (for genai-perf only)

```bash
# Pull and run Triton SDK container
docker pull nvcr.io/nvidia/tritonserver:24.08-py3-sdk
docker run -it --rm --gpus all \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/tritonserver:24.08-py3-sdk bash

# Inside container, genai-perf is pre-installed
cd /workspace
```

## Troubleshooting

### Docker Issues

**Container fails to start:**
- Check if NVIDIA runtime is available: `docker info | grep -i nvidia`
- Check GPU availability: `nvidia-smi`
- Check if port is already in use: `lsof -i :8000`

**Out of memory errors:**
- Reduce `--gpu-memory` value
- Reduce `--max-model-len`
- Increase `--tp-size` for multi-GPU

### Benchmark Issues

**genai-perf not found:**
```bash
pip install genai-perf
```

**Endpoint not accessible:**
- Check container status: `python docker_hf.py status --container-name <name>`
- Check container logs: `python docker_hf.py logs --container-name <name>`
- Verify endpoint URL and port

**Low throughput:**
- Increase `--concurrency`
- Check GPU utilization: `nvidia-smi`
- Increase `--gpu-memory` if GPU is underutilized

## License

See LICENSE file for details.

## References

- [Qwen3-30B-A3B-Thinking-2507 Model Card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [GenAI-Perf Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/README.html)

