#!/usr/bin/env python3

"""
GenAI-Perf Model Performance Measurement Script
This script uses genai-perf to benchmark LLM model performance and export detailed metrics.

Usage:
    python measure_perf.py --model MODEL_NAME --backend BACKEND --endpoint ENDPOINT \
        --input INPUT_FILE --output OUTPUT_FILE [OPTIONS]
"""

import os
import sys
import argparse
import subprocess
import json
import csv
import time
import platform
from pathlib import Path
from datetime import datetime

# Default values
DEFAULT_CONCURRENCY = 40
DEFAULT_REQUEST_COUNT = 100
DEFAULT_ENDPOINT_TYPE = "chat"
DEFAULT_STREAMING = False
GENAI_PERF_IMAGE = "nvcr.io/nvidia/eval-factory/genai-perf:25.11"

class PerformanceMeasure:
    """Manages genai-perf benchmarking and metrics export."""
    
    def __init__(self, model, backend, endpoint, input_file, output_file,
                 concurrency=DEFAULT_CONCURRENCY, request_count=DEFAULT_REQUEST_COUNT,
                 endpoint_type=DEFAULT_ENDPOINT_TYPE, streaming=DEFAULT_STREAMING,
                 extra_args=None, method="hf"):
        self.model = model
        self.backend = backend
        self.endpoint = endpoint
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.concurrency = concurrency
        self.request_count = request_count
        self.endpoint_type = endpoint_type
        self.streaming = streaming
        self.extra_args = extra_args or []
        self.method = method  # Deployment method: hf, nim, unim
        
        # Validate input file
        if not self.input_file.exists():
            print(f"\n❌ Error: Input file not found: {self.input_file}\n")
            sys.exit(1)
    
    def _run_command(self, cmd, capture_output=True, check=False):
        """Run a shell command and return the result."""
        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=check
                )
                return result
            else:
                return subprocess.run(cmd, shell=True, check=check)
        except subprocess.CalledProcessError as e:
            return e
    
    def _collect_environment_info(self):
        """Collect environment details and versions."""
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        }
        
        # Get GPU information
        try:
            result = self._run_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split('\n')[0]
                parts = gpu_info.split(',')
                if len(parts) >= 3:
                    env_info["gpu_name"] = parts[0].strip()
                    env_info["nvidia_driver"] = parts[1].strip()
                    env_info["gpu_memory"] = parts[2].strip()
        except:
            pass
        
        # Get CUDA version
        try:
            result = self._run_command("nvcc --version | grep release | awk '{print $5}' | cut -c2-")
            if result.returncode == 0 and result.stdout.strip():
                env_info["cuda_version"] = result.stdout.strip().rstrip(',')
            else:
                # Try alternative method
                result = self._run_command("nvidia-smi | grep 'CUDA Version' | awk '{print $9}'")
                if result.returncode == 0 and result.stdout.strip():
                    env_info["cuda_version"] = result.stdout.strip()
        except:
            pass
        
        # Get Docker version
        try:
            result = self._run_command("docker --version | awk '{print $3}' | tr -d ','")
            if result.returncode == 0:
                env_info["docker_version"] = result.stdout.strip()
        except:
            pass
        
        # Get genai-perf version
        try:
            # Try local installation first
            result = self._run_command("genai-perf --version 2>&1 | head -n1")
            if result.returncode == 0:
                env_info["genai_perf_version"] = result.stdout.strip()
            else:
                # If local not available, use Docker image version
                env_info["genai_perf_version"] = f"Docker: {GENAI_PERF_IMAGE}"
        except:
            env_info["genai_perf_version"] = f"Docker: {GENAI_PERF_IMAGE}"
        
        # Try to get container/engine version
        try:
            # Check if endpoint is a Docker container
            result = self._run_command("docker ps --format '{{.Names}}:{{.Image}}' | grep -i vllm")
            if result.returncode == 0 and result.stdout.strip():
                container_info = result.stdout.strip().split(':')
                if len(container_info) >= 2:
                    env_info["engine_image"] = container_info[1].strip()
        except:
            pass
        
        return env_info
    
    def _check_genai_perf(self):
        """Check if genai-perf is installed (locally or via Docker)."""
        # First check if local installation exists
        result = self._run_command("which genai-perf")
        if result.returncode == 0:
            print(f"   ✅ Using local genai-perf installation")
            return True
        
        # If not found locally, check if Docker is available
        docker_check = self._run_command("docker --version")
        if docker_check.returncode == 0:
            print(f"   ℹ️  genai-perf not found locally, will use Docker image: {GENAI_PERF_IMAGE}")
            # Pull the Docker image if not already present
            print(f"   🔽 Pulling genai-perf Docker image...")
            pull_result = self._run_command(f"docker pull {GENAI_PERF_IMAGE}", capture_output=False)
            if pull_result.returncode == 0:
                print(f"   ✅ genai-perf Docker image ready")
                return True
            else:
                print(f"   ❌ Failed to pull genai-perf Docker image")
                return False
        
        # Neither local nor Docker available
        print("\n❌ Error: genai-perf is not available!")
        print("   Option 1: Install locally with: pip install genai-perf")
        print(f"   Option 2: Ensure Docker is installed (image will be used: {GENAI_PERF_IMAGE})\n")
        return False
    
    def _check_endpoint(self):
        """Check if the endpoint is accessible."""
        print(f"\n🔍 Checking endpoint: {self.endpoint}")
        
        # Try to reach the health endpoint
        test_cmd = f"curl -s -o /dev/null -w '%{{http_code}}' {self.endpoint}/health --max-time 5"
        result = self._run_command(test_cmd, check=False)
        
        if result.stdout.strip() == '200':
            print(f"   ✅ Endpoint is accessible")
            return True
        else:
            print(f"   ⚠️  Warning: Endpoint may not be ready (HTTP {result.stdout.strip()})")
            print(f"   Continuing anyway...")
            return True
    
    def _warmup_requests(self, num_warmup=5):
        """Send warmup requests to ensure model is ready."""
        print(f"\n🔥 Running {num_warmup} warmup requests...")
        
        # Extract base URL (remove /v1 if present for direct API calls)
        base_url = self.endpoint.rstrip('/')
        # For warmup, use the full OpenAI chat completions endpoint
        if not base_url.endswith('/v1'):
            warmup_url = f"{base_url}/v1/chat/completions"
        else:
            warmup_url = f"{base_url}/chat/completions"
        
        successful = 0
        failed = 0
        
        for i in range(num_warmup):
            try:
                # Simple warmup request
                warmup_cmd = f"""curl -s -X POST {warmup_url} \\
                    -H "Content-Type: application/json" \\
                    -d '{{
                        "model": "{self.model}",
                        "messages": [{{"role": "user", "content": "Hello, respond with OK"}}],
                        "max_tokens": 10
                    }}' \\
                    --max-time 30 \\
                    -w "\\n%{{http_code}}" 2>/dev/null"""
                
                result = self._run_command(warmup_cmd, check=False)
                
                # Check if request was successful (HTTP 200)
                if result.returncode == 0 and '200' in result.stdout:
                    successful += 1
                    print(f"   ✅ Warmup request {i+1}/{num_warmup} successful")
                else:
                    failed += 1
                    print(f"   ⚠️  Warmup request {i+1}/{num_warmup} failed")
                
                # Small delay between warmup requests
                if i < num_warmup - 1:
                    time.sleep(2)
                    
            except Exception as e:
                failed += 1
                print(f"   ❌ Warmup request {i+1}/{num_warmup} error: {e}")
        
        success_rate = (successful / num_warmup) * 100
        print(f"\n   Warmup complete: {successful}/{num_warmup} successful ({success_rate:.1f}%)")
        
        if successful < num_warmup // 2:  # At least 50% success required
            print(f"   ⚠️  Warning: Low warmup success rate. Model may not be fully ready.")
            print(f"   Continuing anyway, but results may be affected...")
        else:
            print(f"   ✅ Model is warmed up and ready for benchmarking")
        
        return successful >= num_warmup // 2
    
    def _build_genai_perf_command(self):
        """Build the genai-perf command according to NVIDIA documentation.
        
        For OpenAI-compatible endpoints (vLLM, SGLang direct, trtllm direct), genai-perf will
        auto-detect the service kind from the URL. The --backend parameter is
        only used for Triton-specific backends like triton-trtllm.
        
        Reference: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
        """
        # Check if genai-perf is available locally
        has_local_genai_perf = self._run_command("which genai-perf").returncode == 0
        
        # genai-perf expects base URL without /v1 for OpenAI endpoints
        # It will automatically append /v1/chat/completions
        base_url = self.endpoint.rstrip('/')
        if base_url.endswith('/v1'):
            base_url = base_url[:-3]  # Remove /v1 suffix
        
        # If using Docker, map localhost to host.docker.internal
        if not has_local_genai_perf:
            base_url = base_url.replace('localhost', 'host.docker.internal')
            base_url = base_url.replace('127.0.0.1', 'host.docker.internal')
        
        # Create artifact directory name with method, model, and backend
        # e.g., artifacts/hf_Qwen_Qwen3-30B-A3B-Thinking-2507_vllm
        model_sanitized = self.model.replace('/', '_')
        artifact_dir_name = f"artifacts/{self.method}_{model_sanitized}_{self.backend}"
        
        # New genai-perf syntax for profile subcommand
        cmd_parts = [
            "genai-perf",
            "profile",
            "--random-seed 42",
            "--warmup-request-count 5",
            f"-m {self.model}",
            f"--endpoint-type {self.endpoint_type}",
            f"--url {base_url}",
            f"--concurrency {self.concurrency}",
            f"--request-count {self.request_count}",
            f"--artifact-dir {artifact_dir_name}",
        ]
        
        # Add backend only for Triton-specific backends
        # triton-trtllm uses Triton Server, so it needs --backend tensorrtllm
        # For OpenAI-compatible endpoints (vLLM, SGLang, trtllm direct), omit --backend
        if self.backend in ["triton-trtllm"]:
            cmd_parts.append("--backend tensorrtllm")
        
        # Add input file if provided
        if self.input_file:
            cmd_parts.append(f"--input-file {self.input_file}")
        
        # Add streaming flag if enabled
        if self.streaming:
            cmd_parts.append("--streaming")
        # The service_kind is auto-detected from the endpoint URL
        # --backend is only for Triton/KServe endpoints
        
        # Add extra arguments
        cmd_parts.extend(self.extra_args)
        
        genai_perf_cmd = " ".join(cmd_parts)
        
        # If genai-perf is not available locally, wrap with Docker
        if not has_local_genai_perf:
            # Get absolute path to current directory for mounting
            current_dir = Path.cwd().absolute()
            
            # Build Docker command
            docker_cmd = f"""docker run --rm \\
    --network host \\
    --add-host host.docker.internal:host-gateway \\
    -v "{current_dir}:/workspace" \\
    -w /workspace \\
    {GENAI_PERF_IMAGE} \\
    {genai_perf_cmd}"""
            
            return docker_cmd
        
        return genai_perf_cmd
    
    def _parse_genai_perf_output(self, artifacts_dir):
        """Parse genai-perf output and extract metrics."""
        print(f"\n📊 Parsing performance metrics...")
        
        # Look for the profile export JSON file
        profile_export_files = list(artifacts_dir.glob("**/profile_export.json"))
        
        if not profile_export_files:
            print(f"   ⚠️  Warning: Could not find profile_export.json in {artifacts_dir}")
            return None
        
        profile_file = profile_export_files[0]
        print(f"   Found profile export: {profile_file}")
        
        try:
            with open(profile_file, 'r') as f:
                data = json.load(f)
            
            # Collect environment information
            env_info = self._collect_environment_info()
            
            # Extract metrics from genai-perf output
            metrics = {
                "model": self.model,
                "backend": self.backend,
                "endpoint_type": self.endpoint_type,
                "concurrency": self.concurrency,
                "streaming": "TRUE" if self.streaming else "FALSE",
                "concurrency_from_config": self.concurrency,
                "measurement_mode": "request_count",
                "measurement_num": self.request_count,
                "service_kind": "openai",
            }
            
            # Add environment information to metrics
            metrics.update(env_info)
            
            # Extract performance metrics
            if "experiments" in data and len(data["experiments"]) > 0:
                exp = data["experiments"][0]
                
                # Request metrics
                if "requests" in exp:
                    req = exp["requests"]
                    metrics.update({
                        "request_throughput_avg": req.get("throughput", {}).get("mean", 0),
                        "request_throughput_unit": "requests/sec",
                        "request_latency_avg": req.get("latency", {}).get("mean", 0),
                        "request_latency_p50": req.get("latency", {}).get("p50", 0),
                        "request_latency_p95": req.get("latency", {}).get("p95", 0),
                        "request_latency_p99": req.get("latency", {}).get("p99", 0),
                        "request_latency_min": req.get("latency", {}).get("min", 0),
                        "request_latency_max": req.get("latency", {}).get("max", 0),
                        "request_latency_std": req.get("latency", {}).get("std", 0),
                        "request_latency_unit": "ms",
                        "request_count": req.get("count", self.request_count),
                    })
                
                # TTFT (Time to First Token) metrics
                if "ttft" in exp:
                    ttft = exp["ttft"]
                    metrics.update({
                        "ttft_avg": ttft.get("mean", 0),
                        "ttft_p50": ttft.get("p50", 0),
                        "ttft_p95": ttft.get("p95", 0),
                        "ttft_unit": "ms",
                    })
                
                # Inter token latency metrics
                if "inter_token_latency" in exp:
                    itl = exp["inter_token_latency"]
                    metrics.update({
                        "inter_token_latency_avg": itl.get("mean", 0),
                        "inter_token_latency_p50": itl.get("p50", 0),
                        "inter_token_latency_p95": itl.get("p95", 0),
                        "inter_token_latency_p99": itl.get("p99", 0),
                        "inter_token_latency_min": itl.get("min", 0),
                        "inter_token_latency_max": itl.get("max", 0),
                        "inter_token_latency_unit": "ms",
                    })
                
                # Token metrics
                if "output_token_throughput" in exp:
                    metrics["output_token_throughput_avg"] = exp["output_token_throughput"].get("mean", 0)
                    metrics["output_token_throughput_unit"] = "tokens/sec"
                
                if "output_sequence_length" in exp:
                    out_seq = exp["output_sequence_length"]
                    metrics.update({
                        "output_seq_len_avg": out_seq.get("mean", 0),
                        "output_seq_len_p50": out_seq.get("p50", 0),
                        "output_seq_len_p95": out_seq.get("p95", 0),
                        "output_seq_len_min": out_seq.get("min", 0),
                        "output_seq_len_max": out_seq.get("max", 0),
                        "output_seq_len_unit": "tokens",
                    })
                
                if "input_sequence_length" in exp:
                    in_seq = exp["input_sequence_length"]
                    metrics.update({
                        "input_seq_len_avg": in_seq.get("mean", 0),
                        "input_seq_len_p50": in_seq.get("p50", 0),
                        "input_seq_len_p95": in_seq.get("p95", 0),
                        "input_seq_len_min": in_seq.get("min", 0),
                        "input_seq_len_max": in_seq.get("max", 0),
                        "input_seq_len_unit": "tokens",
                    })
            
            return metrics
        
        except Exception as e:
            print(f"   ❌ Error parsing profile export: {e}")
            return None
    
    def _export_to_csv(self, metrics):
        """Export metrics to CSV file."""
        # Create timestamped filename if output file doesn't have timestamp
        output_path = Path(self.output_file)
        
        # Check if filename already has a timestamp pattern
        if not any(char.isdigit() for char in output_path.stem[-15:]):
            # Add method, timestamp and backend to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backend_safe = self.backend.replace('-', '_').replace('/', '_')
            new_stem = f"{output_path.stem}_{self.method}_{backend_safe}_{timestamp}"
            output_path = output_path.parent / f"{new_stem}{output_path.suffix}"
        
        print(f"\n💾 Exporting metrics to: {output_path}")
        
        # Define the column order (matching the expected format)
        columns = [
            "model", "backend", "endpoint_type", "concurrency",
            "request_throughput_avg", "request_latency_avg",
            "request_latency_p50", "request_latency_p95", "request_latency_p99",
            "ttft_avg", "ttft_p50", "ttft_p95", "ttft_unit",
            "inter_token_latency_avg", "inter_token_latency_p50", "inter_token_latency_p95", "inter_token_latency_p99",
            "inter_token_latency_min", "inter_token_latency_max", "inter_token_latency_unit",
            "output_token_throughput_avg",
            "output_seq_len_avg", "input_seq_len_avg", "request_count",
            "request_throughput_unit", "request_latency_min", "request_latency_max",
            "request_latency_std", "request_latency_unit",
            "output_token_throughput_unit",
            "output_seq_len_p50", "output_seq_len_p95",
            "output_seq_len_min", "output_seq_len_max", "output_seq_len_unit",
            "input_seq_len_p50", "input_seq_len_p95",
            "input_seq_len_min", "input_seq_len_max", "input_seq_len_unit",
            "service_kind", "streaming", "concurrency_from_config",
            "measurement_mode", "measurement_num",
            # Environment information columns
            "timestamp", "python_version", "platform", "hostname",
            "gpu_name", "nvidia_driver", "gpu_memory", "cuda_version",
            "docker_version", "genai_perf_version", "engine_image"
        ]
        
        # Check if file exists to determine if we need to write headers
        file_exists = output_path.exists()
        
        # Write to CSV
        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(metrics)
        
        print(f"   ✅ Metrics exported successfully")
        print(f"   📄 File: {output_path}")
    
    def run_benchmark(self):
        """Run the performance benchmark."""
        print("=" * 80)
        print("🚀 Running GenAI-Perf Performance Benchmark")
        print("=" * 80)
        print(f"\n📝 Configuration:")
        print(f"   Model: {self.model}")
        print(f"   Backend: {self.backend}")
        print(f"   Endpoint: {self.endpoint}")
        print(f"   Endpoint type: {self.endpoint_type}")
        print(f"   Input file: {self.input_file}")
        print(f"   Output file: {self.output_file}")
        print(f"   Concurrency: {self.concurrency}")
        print(f"   Request count: {self.request_count}")
        print(f"   Streaming: {self.streaming}")
        
        # Check genai-perf installation
        if not self._check_genai_perf():
            sys.exit(1)
        
        # Check endpoint
        self._check_endpoint()
        
        # Run warmup requests
        warmup_success = self._warmup_requests(num_warmup=5)
        if not warmup_success:
            print("\n⚠️  Warning: Warmup phase had low success rate")
            print("   Benchmark may be affected by model loading times")
        
        # Build and run genai-perf command
        print(f"\n⚙️  Running benchmark...")
        genai_perf_cmd = self._build_genai_perf_command()
        print(f"   Command: {genai_perf_cmd}\n")
        
        start_time = time.time()
        result = self._run_command(genai_perf_cmd, capture_output=False, check=False)
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"\n❌ Benchmark failed!")
            print(f"   Check the output above for errors")
            sys.exit(1)
        
        print(f"\n✅ Benchmark completed in {elapsed_time:.2f} seconds")
        
        # Find the artifacts directory (use the custom named directory we specified)
        model_sanitized = self.model.replace('/', '_')
        artifacts_dir = Path.cwd() / "artifacts" / f"{self.method}_{model_sanitized}_{self.backend}"
        
        if not artifacts_dir.exists():
            print(f"   ⚠️  Warning: Artifacts directory not found at {artifacts_dir}")
            print(f"   Metrics export may not be available")
            return
        
        # Parse and export metrics
        metrics = self._parse_genai_perf_output(artifacts_dir)
        
        if metrics:
            self._export_to_csv(metrics)
            print(f"\n📊 Performance Summary:")
            print(f"   Request Throughput: {metrics.get('request_throughput_avg', 'N/A'):.2f} req/s")
            print(f"   Request Latency (avg): {metrics.get('request_latency_avg', 'N/A'):.2f} ms")
            print(f"   Request Latency (p50): {metrics.get('request_latency_p50', 'N/A'):.2f} ms")
            print(f"   Request Latency (p95): {metrics.get('request_latency_p95', 'N/A'):.2f} ms")
            print(f"   Request Latency (p99): {metrics.get('request_latency_p99', 'N/A'):.2f} ms")
            print(f"   Time to first token (avg): {metrics.get('ttft_avg', 'N/A'):.2f} ms")
            print(f"   Time to first token (p50): {metrics.get('ttft_p50', 'N/A'):.2f} ms")
            print(f"   Time to first token (p95): {metrics.get('ttft_p95', 'N/A'):.2f} ms")
            if 'inter_token_latency_avg' in metrics and metrics.get('inter_token_latency_avg', 0) > 0:
                print(f"   Inter token latency (avg): {metrics.get('inter_token_latency_avg', 'N/A'):.2f} ms")
                print(f"   Inter token latency (p50): {metrics.get('inter_token_latency_p50', 'N/A'):.2f} ms")
                print(f"   Inter token latency (p95): {metrics.get('inter_token_latency_p95', 'N/A'):.2f} ms")
                print(f"   Inter token latency (p99): {metrics.get('inter_token_latency_p99', 'N/A'):.2f} ms")
            print(f"   Output Token Throughput: {metrics.get('output_token_throughput_avg', 'N/A'):.2f} tokens/s")
            print(f"   Output Sequence Length (avg): {metrics.get('output_seq_len_avg', 'N/A'):.2f} tokens")
            print(f"   Input Sequence Length (avg): {metrics.get('input_seq_len_avg', 'N/A'):.2f} tokens")
            
            # Print environment info
            print(f"\n🖥️  Environment Info:")
            print(f"   Timestamp: {metrics.get('timestamp', 'N/A')}")
            print(f"   Python: {metrics.get('python_version', 'N/A')}")
            print(f"   Platform: {metrics.get('platform', 'N/A')}")
            if 'gpu_name' in metrics:
                print(f"   GPU: {metrics.get('gpu_name', 'N/A')}")
                print(f"   GPU Memory: {metrics.get('gpu_memory', 'N/A')}")
                print(f"   NVIDIA Driver: {metrics.get('nvidia_driver', 'N/A')}")
            if 'cuda_version' in metrics:
                print(f"   CUDA: {metrics.get('cuda_version', 'N/A')}")
            if 'docker_version' in metrics:
                print(f"   Docker: {metrics.get('docker_version', 'N/A')}")
            if 'engine_image' in metrics:
                print(f"   Engine Image: {metrics.get('engine_image', 'N/A')}")
        else:
            print(f"\n⚠️  Warning: Could not extract metrics from genai-perf output")

def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Measure LLM model performance using genai-perf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python measure_perf.py \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --backend vllm \
    --endpoint http://localhost:8000/v1 \
    --input input.jsonl \
    --output results.csv
  
  # Benchmark with custom settings
  python measure_perf.py \
    --model meta-llama/Llama-3-8B \
    --backend trtllm \
    --endpoint http://localhost:8000/v1 \
    --input prompts.jsonl \
    --output perf_results.csv \
    --concurrency 100 \
    --request-count 500 \
    --streaming
  
  # Multiple backends comparison
  for backend in vllm trtllm sglang; do
    python measure_perf.py \
      --model MODEL_NAME \
      --backend $backend \
      --endpoint http://localhost:8000/v1 \
      --input input.jsonl \
      --output combined_results.csv
  done
        """
    )
    
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--backend', required=True, help='Backend/engine name (e.g., vllm, trtllm, sglang)')
    parser.add_argument('--endpoint', required=True, help='API endpoint URL (e.g., http://localhost:8000/v1)')
    parser.add_argument('--input', '--input-file', dest='input_file', required=True, help='Input JSONL file with prompts')
    parser.add_argument('--output', '--output-file', dest='output_file', required=True, help='Output CSV file for metrics')
    parser.add_argument('--method', default='hf', help='Deployment method (hf, nim, unim) for naming [default: hf]')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY, 
                       help=f'Number of concurrent requests (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--request-count', type=int, default=DEFAULT_REQUEST_COUNT,
                       help=f'Total number of requests (default: {DEFAULT_REQUEST_COUNT})')
    parser.add_argument('--endpoint-type', default=DEFAULT_ENDPOINT_TYPE,
                       choices=['chat', 'completions', 'embeddings'],
                       help=f'Endpoint type (default: {DEFAULT_ENDPOINT_TYPE})')
    parser.add_argument('--streaming', action='store_true', help='Enable streaming mode')
    parser.add_argument('--extra-args', nargs='+', default=[], help='Additional genai-perf arguments')
    
    args = parser.parse_args()
    
    # Create performance measurer and run benchmark
    measurer = PerformanceMeasure(
        model=args.model,
        backend=args.backend,
        endpoint=args.endpoint,
        input_file=args.input_file,
        output_file=args.output_file,
        concurrency=args.concurrency,
        request_count=args.request_count,
        endpoint_type=args.endpoint_type,
        streaming=args.streaming,
        extra_args=args.extra_args,
        method=args.method
    )
    
    measurer.run_benchmark()
    print("\n✅ Done!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

