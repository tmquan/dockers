#!/usr/bin/env python3

"""
GenAI-Perf Model Performance Measurement Script
Unified script for benchmarking LLM models across different deployment methods and engines.

Usage:
    python measure.py --method METHOD --model MODEL_NAME --engine ENGINE --endpoint ENDPOINT [OPTIONS]
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

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_CONCURRENCY = 40
DEFAULT_REQUEST_COUNT = 1000
DEFAULT_WARMUP_REQUEST_COUNT = 100
DEFAULT_ENDPOINT_TYPE = "chat"
DEFAULT_STREAMING = True
DEFAULT_METHOD = "hf"
DEFAULT_ENGINE = "vllm"
DEFAULT_INPUT_SEQUENCE_LENGTH = 30000
DEFAULT_OUTPUT_SEQUENCE_LENGTH = 3000
DEFAULT_MEASUREMENT_INTERVAL = None  # Use request-count mode by default
GENAI_PERF_IMAGE = "nvcr.io/nvidia/eval-factory/genai-perf:25.11"

# ============================================================================
# Performance Measurement Class
# ============================================================================
class PerformanceMeasure:
    """Manages genai-perf benchmarking and metrics export."""
    
    def __init__(self, method, model, engine, endpoint,
                 input_file=None, output_file=None,
                 input_tokens_mean=None, output_tokens_mean=None,
                 concurrency=DEFAULT_CONCURRENCY,
                 request_count=DEFAULT_REQUEST_COUNT,
                 warmup_request_count=DEFAULT_WARMUP_REQUEST_COUNT,
                 endpoint_type=DEFAULT_ENDPOINT_TYPE,
                 streaming=DEFAULT_STREAMING,
                 measurement_interval=DEFAULT_MEASUREMENT_INTERVAL,
                 extra_args=None):
        self.method = method.lower()
        self.model = model
        self.engine = engine.lower()
        self.endpoint = endpoint
        self.input_file = Path(input_file) if input_file else None
        self.output_file = Path(output_file) if output_file else self._get_default_output_file()
        self.input_tokens_mean = input_tokens_mean
        self.output_tokens_mean = output_tokens_mean
        self.concurrency = concurrency
        self.request_count = request_count
        self.warmup_request_count = warmup_request_count
        self.endpoint_type = endpoint_type
        self.streaming = streaming
        self.measurement_interval = measurement_interval
        self.extra_args = extra_args or []
        
        # Validate input
        if self.input_file and not self.input_file.exists():
            print(f"\n❌ Error: Input file not found: {self.input_file}\n")
            sys.exit(1)
        
        if not self.input_file and not self.input_tokens_mean:
            print(f"\n❌ Error: Either --input-file or --input-tokens-mean must be specified\n")
            sys.exit(1)
    
    def _get_default_output_file(self):
        """Generate default output filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_sanitized = self.model.replace('/', '_').replace('-', '_')
        return Path(f"benchmark_{model_sanitized}_{self.method}_{self.engine}_{timestamp}.csv")
    
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
            result = self._run_command("genai-perf --version 2>&1 | head -n1")
            if result.returncode == 0:
                env_info["genai_perf_version"] = result.stdout.strip()
            else:
                env_info["genai_perf_version"] = f"Docker: {GENAI_PERF_IMAGE}"
        except:
            env_info["genai_perf_version"] = f"Docker: {GENAI_PERF_IMAGE}"
        
        return env_info
    
    def _check_genai_perf(self):
        """Check if genai-perf is available."""
        result = self._run_command("which genai-perf")
        if result.returncode == 0:
            print(f"   ✅ Using local genai-perf")
            return "local"
        
        docker_check = self._run_command("docker --version")
        if docker_check.returncode == 0:
            print(f"   ℹ️  Using Docker-based genai-perf: {GENAI_PERF_IMAGE}")
            return "docker"
        
        print("\n❌ Error: genai-perf is not available!")
        print("   Install with: pip install genai-perf")
        print("   Or ensure Docker is installed\n")
        sys.exit(1)
    
    def _check_endpoint(self):
        """Check if the endpoint is accessible."""
        print(f"\n🔍 Checking endpoint: {self.endpoint}")
        
        # Determine health endpoint based on method
        if self.method == "triton":
            health_endpoint = "/v2/health/ready"
        else:
            health_endpoint = "/v1/models"
        
        test_cmd = f"curl -s -o /dev/null -w '%{{http_code}}' {self.endpoint}{health_endpoint} --max-time 5"
        result = self._run_command(test_cmd, check=False)
        
        if result.stdout.strip() == '200':
            print(f"   ✅ Endpoint is accessible")
            return True
        else:
            print(f"   ⚠️  Warning: Endpoint may not be ready (HTTP {result.stdout.strip()})")
            print(f"   Continuing anyway...")
            return True
    
    def _build_genai_perf_command(self, mode):
        """Build genai-perf command.
        
        Args:
            mode: 'local' or 'docker'
        """
        # Prepare endpoint URL
        base_url = self.endpoint.rstrip('/')
        if base_url.endswith('/v1'):
            base_url = base_url[:-3]
        
        # For Docker, map localhost to host.docker.internal
        if mode == "docker":
            base_url = base_url.replace('localhost', 'host.docker.internal')
            base_url = base_url.replace('127.0.0.1', 'host.docker.internal')
        
        # For Triton, sanitize model name (replace / with _)
        # Triton model repository uses underscores, not slashes
        model_name_for_request = self.model
        if self.method == "triton":
            model_name_for_request = self.model.replace('/', '_')
        
        # Create artifact directory name
        model_sanitized = self.model.replace('/', '_')
        artifact_dir_name = f"artifacts/{model_sanitized}_{self.method}_{self.engine}"
        
        # Build command
        cmd_parts = [
            "genai-perf",
            "profile",
            "--random-seed 42",
            f"--warmup-request-count {self.warmup_request_count}",
            f"-m {model_name_for_request}",
            f"--endpoint-type {self.endpoint_type}",
            f"-u {base_url}",
            f"--concurrency {self.concurrency}",
            f"--artifact-dir {artifact_dir_name}",
            f"--tokenizer {self.model}",
        ]
        
        # Add measurement mode: either request-count OR measurement-interval
        # GenAI-Perf doesn't allow both at the same time
        if self.request_count:
            cmd_parts.append(f"--request-count {self.request_count}")
        else:
            cmd_parts.append(f"--measurement-interval {self.measurement_interval}")
        
        # Add input source (file or synthetic)
        if self.input_file:
            cmd_parts.append(f"--input-file {self.input_file}")
        elif self.input_tokens_mean:
            cmd_parts.append(f"--synthetic-input-tokens-mean {self.input_tokens_mean}")
            cmd_parts.append(f"--synthetic-input-tokens-stddev 0")
        
        # Add output tokens
        if self.output_tokens_mean:
            cmd_parts.append(f"--output-tokens-mean {self.output_tokens_mean}")
            cmd_parts.append(f"--extra-inputs max_tokens:{self.output_tokens_mean}")
            cmd_parts.append(f"--extra-inputs min_tokens:{self.output_tokens_mean}")
            cmd_parts.append(f"--extra-inputs ignore_eos:true")
        
        # Add streaming flag
        if self.streaming:
            cmd_parts.append("--streaming")
        
        # Add extra arguments
        cmd_parts.extend(self.extra_args)
        
        genai_perf_cmd = " ".join(cmd_parts)
        
        # Wrap with Docker if needed
        if mode == "docker":
            current_dir = Path.cwd().absolute()
            docker_cmd = f"""docker run --rm \\
    --network host \\
    --add-host host.docker.internal:host-gateway \\
    --gpus=all \\
    -v "{current_dir}:/workdir" \\
    -w /workdir \\
    {GENAI_PERF_IMAGE} \\
    {genai_perf_cmd}"""
            return docker_cmd
        
        return genai_perf_cmd
    
    def _parse_genai_perf_output(self):
        """Parse genai-perf output and extract metrics."""
        print(f"\n📊 Parsing performance metrics...")
        
        # Find artifacts directory
        model_sanitized = self.model.replace('/', '_')
        artifacts_base = Path.cwd() / "artifacts"
        
        # Look for the profile export file
        artifact_pattern = f"{model_sanitized}_{self.method}_{self.engine}"
        artifact_dirs = list(artifacts_base.glob(f"{artifact_pattern}*"))
        
        if not artifact_dirs:
            print(f"   ⚠️  Warning: Could not find artifacts directory matching {artifact_pattern}")
            return None
        
        # Use the most recent directory
        artifacts_dir = sorted(artifact_dirs, key=lambda p: p.stat().st_mtime)[-1]
        
        profile_files = list(artifacts_dir.glob("**/profile_export.json"))
        if not profile_files:
            print(f"   ⚠️  Warning: Could not find profile_export.json in {artifacts_dir}")
            return None
        
        profile_file = profile_files[0]
        print(f"   Found profile: {profile_file}")
        
        try:
            with open(profile_file, 'r') as f:
                data = json.load(f)
            
            # Collect environment information
            env_info = self._collect_environment_info()
            
            # Extract metrics
            metrics = {
                "method": self.method,
                "model": self.model,
                "engine": self.engine,
                "endpoint_type": self.endpoint_type,
                "concurrency": self.concurrency,
                "streaming": "TRUE" if self.streaming else "FALSE",
                "request_count": self.request_count,
            }
            
            # Add environment information
            metrics.update(env_info)
            
            # Extract performance metrics
            if "experiments" in data and len(data["experiments"]) > 0:
                exp = data["experiments"][0]
                
                # Request metrics
                if "requests" in exp:
                    req = exp["requests"]
                    metrics.update({
                        "request_throughput_avg": req.get("throughput", {}).get("mean", 0),
                        "request_latency_avg": req.get("latency", {}).get("mean", 0),
                        "request_latency_p50": req.get("latency", {}).get("p50", 0),
                        "request_latency_p95": req.get("latency", {}).get("p95", 0),
                        "request_latency_p99": req.get("latency", {}).get("p99", 0),
                        "request_latency_min": req.get("latency", {}).get("min", 0),
                        "request_latency_max": req.get("latency", {}).get("max", 0),
                        "request_latency_std": req.get("latency", {}).get("std", 0),
                    })
                
                # TTFT metrics
                if "ttft" in exp:
                    ttft = exp["ttft"]
                    metrics.update({
                        "ttft_avg": ttft.get("mean", 0),
                        "ttft_p50": ttft.get("p50", 0),
                        "ttft_p95": ttft.get("p95", 0),
                        "ttft_p99": ttft.get("p99", 0),
                    })
                
                # Inter-token latency metrics
                if "inter_token_latency" in exp:
                    itl = exp["inter_token_latency"]
                    metrics.update({
                        "inter_token_latency_avg": itl.get("mean", 0),
                        "inter_token_latency_p50": itl.get("p50", 0),
                        "inter_token_latency_p95": itl.get("p95", 0),
                        "inter_token_latency_p99": itl.get("p99", 0),
                    })
                
                # Token metrics
                if "output_token_throughput" in exp:
                    metrics["output_token_throughput_avg"] = exp["output_token_throughput"].get("mean", 0)
                
                if "output_sequence_length" in exp:
                    out_seq = exp["output_sequence_length"]
                    metrics.update({
                        "output_seq_len_avg": out_seq.get("mean", 0),
                        "output_seq_len_p50": out_seq.get("p50", 0),
                        "output_seq_len_p95": out_seq.get("p95", 0),
                    })
                
                if "input_sequence_length" in exp:
                    in_seq = exp["input_sequence_length"]
                    metrics.update({
                        "input_seq_len_avg": in_seq.get("mean", 0),
                        "input_seq_len_p50": in_seq.get("p50", 0),
                        "input_seq_len_p95": in_seq.get("p95", 0),
                    })
            
            return metrics
        
        except Exception as e:
            print(f"   ❌ Error parsing profile: {e}")
            return None
    
    def _export_to_csv(self, metrics):
        """Export metrics to CSV file."""
        print(f"\n💾 Exporting metrics to: {self.output_file}")
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Column order
        columns = [
            "method", "model", "engine", "endpoint_type", "concurrency",
            "request_throughput_avg", "request_latency_avg",
            "request_latency_p50", "request_latency_p95", "request_latency_p99",
            "request_latency_min", "request_latency_max", "request_latency_std",
            "ttft_avg", "ttft_p50", "ttft_p95", "ttft_p99",
            "inter_token_latency_avg", "inter_token_latency_p50",
            "inter_token_latency_p95", "inter_token_latency_p99",
            "output_token_throughput_avg",
            "output_seq_len_avg", "output_seq_len_p50", "output_seq_len_p95",
            "input_seq_len_avg", "input_seq_len_p50", "input_seq_len_p95",
            "request_count", "streaming",
            "timestamp", "python_version", "platform", "hostname",
            "gpu_name", "nvidia_driver", "gpu_memory", "cuda_version",
            "docker_version", "genai_perf_version"
        ]
        
        file_exists = self.output_file.exists()
        
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(metrics)
        
        print(f"   ✅ Metrics exported successfully")
    
    def run_benchmark(self):
        """Run the performance benchmark."""
        print("=" * 80)
        print("🚀 Running GenAI-Perf Benchmark")
        print("=" * 80)
        print(f"\n📝 Configuration:")
        print(f"   Method: {self.method}")
        print(f"   Model: {self.model}")
        print(f"   Engine: {self.engine}")
        print(f"   Endpoint: {self.endpoint}")
        print(f"   Concurrency: {self.concurrency}")
        print(f"   Request count: {self.request_count}")
        print(f"   Streaming: {self.streaming}")
        if self.input_file:
            print(f"   Input: {self.input_file}")
        else:
            print(f"   Input: Synthetic ({self.input_tokens_mean} tokens)")
        if self.output_tokens_mean:
            print(f"   Output: {self.output_tokens_mean} tokens")
        
        # Check prerequisites
        mode = self._check_genai_perf()
        self._check_endpoint()
        
        # Run benchmark
        print(f"\n⚙️  Running benchmark...")
        genai_perf_cmd = self._build_genai_perf_command(mode)
        print(f"   Command: {genai_perf_cmd[:200]}...\n")
        
        start_time = time.time()
        result = self._run_command(genai_perf_cmd, capture_output=False, check=False)
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"\n❌ Benchmark failed!")
            sys.exit(1)
        
        print(f"\n✅ Benchmark completed in {elapsed_time:.2f} seconds")
        
        # Parse and export metrics
        metrics = self._parse_genai_perf_output()
        
        if metrics:
            self._export_to_csv(metrics)
            
            print(f"\n📊 Performance Summary:")
            print(f"   Request Throughput: {metrics.get('request_throughput_avg', 0):.2f} req/s")
            print(f"   Request Latency (avg): {metrics.get('request_latency_avg', 0):.2f} ms")
            print(f"   Request Latency (p95): {metrics.get('request_latency_p95', 0):.2f} ms")
            print(f"   Time to First Token (avg): {metrics.get('ttft_avg', 0):.2f} ms")
            print(f"   Time to First Token (p95): {metrics.get('ttft_p95', 0):.2f} ms")
            if metrics.get('inter_token_latency_avg', 0) > 0:
                print(f"   Inter-token Latency (avg): {metrics.get('inter_token_latency_avg', 0):.2f} ms")
            print(f"   Output Token Throughput: {metrics.get('output_token_throughput_avg', 0):.2f} tokens/s")
            print(f"   Output Tokens (avg): {metrics.get('output_seq_len_avg', 0):.2f}")
            print(f"   Input Tokens (avg): {metrics.get('input_seq_len_avg', 0):.2f}")
        else:
            print(f"\n⚠️  Warning: Could not extract metrics")


# ============================================================================
# Main CLI
# ============================================================================
def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Measure LLM model performance using genai-perf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with synthetic input (32k tokens)
  python measure.py \\
    --method hf \\
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \\
    --engine vllm \\
    --endpoint http://localhost:8000 \\
    --input-tokens-mean 30000 \\
    --output-tokens-mean 200 \\
    --concurrency 40 \\
    --request-count 1000
  
  # Benchmark with input file
  python measure.py \\
    --method hf \\
    --model meta-llama/Llama-3-8B \\
    --engine trtllm \\
    --endpoint http://localhost:8000 \\
    --input-file prompts.jsonl \\
    --output results.csv
  
  # Triton with OpenAI frontend
  python measure.py \\
    --method triton \\
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \\
    --engine vllm \\
    --endpoint http://localhost:9000 \\
    --input-tokens-mean 30000 \\
    --output-tokens-mean 200
        """
    )
    
    parser.add_argument('--method', default=DEFAULT_METHOD,
                       help=f'Deployment method (default: {DEFAULT_METHOD})')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--engine', required=True, help='Engine name (vllm, sglang, trtllm, python)')
    parser.add_argument('--endpoint', required=True, help='API endpoint URL (e.g., http://localhost:8000)')
    
    # Input/output
    parser.add_argument('--input-file', help='Input JSONL file with prompts')
    parser.add_argument('--output-file', help='Output CSV file (auto-generated if not specified)')
    parser.add_argument('--input-tokens-mean', type=int,
                       help='Synthetic input tokens mean')
    parser.add_argument('--output-tokens-mean', type=int,
                       help='Output tokens mean')
    
    # Benchmark parameters
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                       help=f'Concurrent requests (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--request-count', type=int, default=DEFAULT_REQUEST_COUNT,
                       help=f'Total requests (default: {DEFAULT_REQUEST_COUNT}). Mutually exclusive with --measurement-interval')
    parser.add_argument('--warmup-request-count', type=int, default=DEFAULT_WARMUP_REQUEST_COUNT,
                       help=f'Warmup requests (default: {DEFAULT_WARMUP_REQUEST_COUNT})')
    parser.add_argument('--endpoint-type', default=DEFAULT_ENDPOINT_TYPE,
                       choices=['chat', 'completions', 'embeddings'],
                       help=f'Endpoint type (default: {DEFAULT_ENDPOINT_TYPE})')
    parser.add_argument('--streaming', action='store_true', default=DEFAULT_STREAMING,
                       help='Enable streaming (default: enabled)')
    parser.add_argument('--no-streaming', action='store_false', dest='streaming',
                       help='Disable streaming')
    parser.add_argument('--measurement-interval', type=int,
                       help='Measurement interval in ms. If set, --request-count is ignored. Use one or the other, not both.')
    parser.add_argument('--extra-args', nargs='+', default=[],
                       help='Additional genai-perf arguments')
    
    args = parser.parse_args()
    
    # Create measurer and run benchmark
    measurer = PerformanceMeasure(
        method=args.method,
        model=args.model,
        engine=args.engine,
        endpoint=args.endpoint,
        input_file=args.input_file,
        output_file=args.output_file,
        input_tokens_mean=args.input_tokens_mean,
        output_tokens_mean=args.output_tokens_mean,
        concurrency=args.concurrency,
        request_count=args.request_count,
        warmup_request_count=args.warmup_request_count,
        endpoint_type=args.endpoint_type,
        streaming=args.streaming,
        measurement_interval=args.measurement_interval,
        extra_args=args.extra_args
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

