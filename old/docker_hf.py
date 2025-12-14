#!/usr/bin/env python3

"""
HuggingFace Model Deployment Manager
This script manages the lifecycle of HuggingFace model Docker containers using various inference engines.

Supported engines: vllm, sglang, trtllm

Usage:
    python docker_hf.py start --model MODEL_NAME --engine ENGINE [OPTIONS]
    python docker_hf.py stop [--container-name NAME]
    python docker_hf.py restart --model MODEL_NAME --engine ENGINE [OPTIONS]
    python docker_hf.py status [--container-name NAME]
    python docker_hf.py logs [--container-name NAME] [-f]
"""

import os
import sys
import time
import argparse
import subprocess
import re
from pathlib import Path

# ============================================================================
# Version Configurations - Update these when upgrading engine versions
# ============================================================================
VLLM_VERSION = "v0.12.0"  # Supports Qwen3 MoE models (Dec 2024)
SGLANG_VERSION = "v0.5.6.post2"  # Use latest for newest model support
TRTLLM_VERSION = "1.2.0rc4"  # NVIDIA TensorRT-LLM 25.11 (recommended for best performance)

# ============================================================================
# Docker Image Configurations
# ============================================================================
VLLM_IMAGE = f"vllm/vllm-openai:{VLLM_VERSION}"
SGLANG_IMAGE = f"lmsysorg/sglang:{SGLANG_VERSION}"
TRTLLM_IMAGE = f"nvcr.io/nvidia/tensorrt-llm/release:{TRTLLM_VERSION}"

# ============================================================================
# Default Configurations
# ============================================================================
DEFAULT_PORT = 8000
DEFAULT_GPU_MEMORY = 0.9
DEFAULT_OPT_ENGINE = "vllm"
CONTAINER_CACHE_PATH = "/root/.cache/huggingface"

# ============================================================================
# Engine Configurations
# ============================================================================
ENGINE_CONFIGS = {
    "vllm": {
        "image": VLLM_IMAGE,
        "health_endpoint": "/v1/models",
        "supports_openai": True,
        "description": "vLLM - Fast and flexible inference with PagedAttention (recommended for most use cases)",
    },
    "sglang": {
        "image": SGLANG_IMAGE,
        "health_endpoint": "/v1/models",
        "supports_openai": True,
        "description": "SGLang - Optimized for structured generation and complex prompts",
    },
    "trtllm": {
        "image": TRTLLM_IMAGE,
        "health_endpoint": "/v1/models",
        "supports_openai": True,
        "description": "NVIDIA TensorRT-LLM - Maximum performance with TensorRT optimizations (best for production)",
    },
}

class HFModelDeployer:
    """Manages HuggingFace model Docker container lifecycle with multiple engine support."""
    
    def __init__(self, model, engine=DEFAULT_OPT_ENGINE, cache_dir=None, port=DEFAULT_PORT, 
                 gpu_memory=DEFAULT_GPU_MEMORY, container_name=None, tp_size=1, 
                 max_model_len=None, extra_args=None):
        self.model = model
        self.engine = engine.lower()
        # Use .cache/hf in current directory by default (for HuggingFace models)
        # Future: .cache/nim will be used for NIM models
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        else:
            self.cache_dir = Path.cwd() / ".cache" / "hf"
        self.port = port
        self.gpu_memory = gpu_memory
        self.tp_size = tp_size
        self.max_model_len = max_model_len
        self.extra_args = extra_args or []
        
        # Generate container name if not provided
        if container_name:
            self.container_name = container_name
        else:
            # Create sanitized container name from model name
            sanitized_model = re.sub(r'[^a-zA-Z0-9_-]', '-', model.split('/')[-1].lower())
            self.container_name = f"{sanitized_model}-{self.engine}"
        
        # Validate engine
        if self.engine not in ENGINE_CONFIGS:
            print(f"\n❌ Error: Unsupported engine '{self.engine}'")
            print(f"   Supported engines: {', '.join(ENGINE_CONFIGS.keys())}\n")
            sys.exit(1)
        
        self.engine_config = ENGINE_CONFIGS[self.engine]
        
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
    
    def _get_container_id(self):
        """Get the container ID if running."""
        result = self._run_command(
            f"docker ps -q -f name=^{self.container_name}$"
        )
        return result.stdout.strip() if result.returncode == 0 else None
    
    def _get_container_status(self):
        """Get container status (running, exited, or not found)."""
        result = self._run_command(
            f"docker ps -a -f name=^{self.container_name}$ --format '{{{{.Status}}}}'"
        )
        return result.stdout.strip() if result.returncode == 0 else None
    
    def _check_docker(self):
        """Check if Docker is running."""
        result = self._run_command("docker info")
        if result.returncode != 0:
            print("\n❌ Error: Docker is not running!")
            print("   Please start Docker and try again.\n")
            sys.exit(1)
    
    def _check_nvidia_runtime(self):
        """Check if NVIDIA Docker runtime is available."""
        result = self._run_command("docker info | grep -i nvidia")
        if result.returncode != 0 or not result.stdout.strip():
            print("\n⚠️  Warning: NVIDIA runtime not detected!")
            print("   LLM models require GPU support. Container may not work properly.")
            return False
        return True
    
    def _setup_cache_directory(self):
        """Create and set up cache directory with proper permissions."""
        print(f"\n📁 Setting up cache directory: {self.cache_dir}")
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set permissions for container access
        self._run_command(f"chmod -R 777 {self.cache_dir}")
        
        print(f"   ✅ Cache directory ready with proper permissions")
    
    def _build_vllm_command(self, image):
        """Build Docker command for vLLM engine."""
        cmd_args = [
            f"--model {self.model}",
            f"--gpu-memory-utilization {self.gpu_memory}",
            "--trust-remote-code",
        ]
        
        if self.tp_size > 1:
            cmd_args.append(f"--tensor-parallel-size {self.tp_size}")
        
        if self.max_model_len:
            cmd_args.append(f"--max-model-len {self.max_model_len}")
        else:
            # Try to use a reasonable default
            cmd_args.append("--max-model-len 32768")
        
        cmd_args.append("--enable-chunked-prefill")
        cmd_args.extend(self.extra_args)
        
        return f"""docker run -d \\
    --name {self.container_name} \\
    --runtime=nvidia \\
    --gpus all \\
    --shm-size=16g \\
    -e HF_TOKEN={os.getenv('HF_TOKEN', '')} \\
    -e HUGGING_FACE_HUB_TOKEN={os.getenv('HF_TOKEN', '')} \\
    -e HF_HOME=/root/.cache/huggingface \\
    -p {self.port}:8000 \\
    -v "{self.cache_dir}:/root/.cache/huggingface" \\
    {image} \\
    {' '.join(cmd_args)}"""
    
    def _build_sglang_command(self, image):
        """Build Docker command for SGLang engine."""
        cmd_args = [
            f"--model-path {self.model}",
            f"--host 0.0.0.0",  # Bind to all interfaces for Docker port mapping
            f"--port 8000",
            f"--mem-frac {self.gpu_memory}",
            "--trust-remote-code",
        ]
        
        if self.tp_size > 1:
            cmd_args.append(f"--tp {self.tp_size}")
        
        if self.max_model_len:
            cmd_args.append(f"--context-length {self.max_model_len}")
        
        cmd_args.extend(self.extra_args)
        
        return f"""docker run -d \\
    --name {self.container_name} \\
    --runtime=nvidia \\
    --gpus all \\
    --shm-size=16g \\
    -e HF_TOKEN={os.getenv('HF_TOKEN', '')} \\
    -e HUGGING_FACE_HUB_TOKEN={os.getenv('HF_TOKEN', '')} \\
    -e HF_HOME=/root/.cache/huggingface \\
    -p {self.port}:8000 \\
    -v "{self.cache_dir}:/root/.cache/huggingface" \\
    {image} \\
    python3 -m sglang.launch_server {' '.join(cmd_args)}"""
    
    def _build_trtllm_command(self, image):
        """Build Docker command for NVIDIA TensorRT-LLM engine.
        
        Based on: https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
        
        This uses NVIDIA's TensorRT-LLM with trtllm-serve command.
        Provides best performance for TensorRT-optimized inference.
        """
        
        # Use max_model_len if specified, otherwise let trtllm-serve decide
        max_seq_len = self.max_model_len if self.max_model_len else 131072  # 128k
        # max_num_tokens controls the maximum tokens that can be processed per batch
        # Should be at least as large as the longest expected prompt
        max_num_tokens = max_seq_len  # Allow full sequence length per request
        
        # TensorRT-LLM container needs to bind to 0.0.0.0 for external access
        # Use --host 0.0.0.0 to bind to all interfaces, not just localhost
        return f"""docker run -d \\
    --name {self.container_name} \\
    --runtime=nvidia \\
    --gpus all \\
    --ipc host \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    --shm-size=16g \\
    -e HF_TOKEN={os.getenv('HF_TOKEN', '')} \\
    -e HUGGING_FACE_HUB_TOKEN={os.getenv('HF_TOKEN', '')} \\
    -e HF_HOME=/root/.cache/huggingface \\
    -p {self.port}:8000 \\
    -v "{self.cache_dir}:/root/.cache/huggingface" \\
    {image} \\
    trtllm-serve serve {self.model} --max_seq_len {max_seq_len} --max_num_tokens {max_num_tokens} --host 0.0.0.0"""
    
    
    def _build_docker_command(self):
        """Build the appropriate Docker command based on engine."""
        image = self.engine_config["image"]
        
        if self.engine == "vllm":
            return self._build_vllm_command(image)
        elif self.engine == "sglang":
            return self._build_sglang_command(image)
        elif self.engine == "trtllm":
            return self._build_trtllm_command(image)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
    
    def status(self):
        """Check the status of the model container."""
        print("=" * 80)
        print(f"🔍 Checking Model Container Status: {self.container_name}")
        print("=" * 80)
        
        container_id = self._get_container_id()
        status = self._get_container_status()
        
        if not status:
            print(f"\n❌ Container '{self.container_name}' not found")
            print(f"   Use 'python docker_hf.py start --model <model> --engine <engine>' to create and start the container\n")
            return False
        
        if container_id:
            print(f"\n✅ Container '{self.container_name}' is RUNNING")
            print(f"   Container ID: {container_id}")
            print(f"   Status: {status}")
            print(f"   Model: {self.model}")
            print(f"   Engine: {self.engine}")
            print(f"   Service URL: http://localhost:{self.port}")
            
            health_endpoint = self.engine_config["health_endpoint"]
            print(f"   Health check: http://localhost:{self.port}{health_endpoint}")
            
            if self.engine_config["supports_openai"]:
                print(f"   OpenAI API: http://localhost:{self.port}/v1")
            
            # Test if API is responsive
            test_result = self._run_command(
                f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{self.port}{health_endpoint} --max-time 2",
                check=False
            )
            if test_result.stdout.strip() == '200':
                print(f"   API Status: ✅ Ready")
            else:
                print(f"   API Status: ⏳ Starting up (may take several minutes)")
        else:
            print(f"\n⚠️  Container '{self.container_name}' exists but is NOT RUNNING")
            print(f"   Status: {status}")
            print(f"   Use 'python docker_hf.py start --model {self.model} --engine {self.engine}' to start it")
        
        print()
        return container_id is not None
    
    def start(self):
        """Start the model container."""
        print("=" * 80)
        print(f"🚀 Starting Model Container: {self.model}")
        print(f"   Engine: {self.engine}")
        print("=" * 80)
        
        # Check Docker
        self._check_docker()
        
        # Check if already running
        if self._get_container_id():
            print("\n⚠️  Container is already running!")
            self.status()
            return
        
        # Check NVIDIA runtime
        has_nvidia = self._check_nvidia_runtime()
        if not has_nvidia:
            print("\n❌ Error: NVIDIA runtime is required for LLM inference!")
            sys.exit(1)
        
        # Setup cache directory
        self._setup_cache_directory()
        
        # Check if container exists but is stopped
        status = self._get_container_status()
        if status and "Exited" in status:
            print(f"\n🔄 Found stopped container, removing it...")
            self._run_command(f"docker rm {self.container_name}")
        
        # Pull the image
        image = self.engine_config["image"]
        print(f"\n🔽 Checking for {self.engine} image: {image}")
        image_check = self._run_command(f"docker image inspect {image}")
        if image_check.returncode != 0:
            print(f"   Image not found locally, pulling from registry...")
            print(f"   ⏳ This may take several minutes...")
            pull_result = self._run_command(f"docker pull {image}", capture_output=False)
            if pull_result.returncode != 0:
                print(f"\n❌ Failed to pull image. Check your internet connection.\n")
                sys.exit(1)
        else:
            print(f"   ✅ Image found locally")
        
        # Build and execute Docker command
        print(f"\n🐳 Starting Docker container...")
        print(f"   Container name: {self.container_name}")
        print(f"   Host port: {self.port}")
        print(f"   Cache directory: {self.cache_dir}")
        print(f"   Model: {self.model}")
        print(f"   Engine: {self.engine}")
        print(f"   GPU memory utilization: {self.gpu_memory}")
        if self.tp_size > 1:
            print(f"   Tensor parallel size: {self.tp_size}")
        
        docker_cmd = self._build_docker_command()
        result = self._run_command(docker_cmd.strip())
        
        if result.returncode == 0:
            print(f"\n✅ Container started successfully!")
            print(f"   Container ID: {result.stdout.strip()}")
            print(f"\n⏳ Model is initializing... This may take several minutes.")
            print(f"   First run will download the model weights.")
            print(f"   Service URL: http://localhost:{self.port}")
            health_endpoint = self.engine_config["health_endpoint"]
            print(f"   Health check: http://localhost:{self.port}{health_endpoint}")
            print(f"\n💡 Use 'python docker_hf.py status --container-name {self.container_name}' to check if the API is ready")
            print(f"   Use 'python docker_hf.py logs --container-name {self.container_name} -f' to watch the startup logs\n")
        else:
            print(f"\n❌ Failed to start container!")
            print(f"   Error: {result.stderr}")
            print(f"\n💡 Common issues:")
            print(f"   - NVIDIA Docker runtime not installed (nvidia-docker2)")
            print(f"   - GPU not available or insufficient GPU memory")
            print(f"   - Port {self.port} already in use")
            print(f"   - Insufficient disk space in {self.cache_dir}")
            print(f"   - Model not found or access denied (check HF_TOKEN)\n")
            sys.exit(1)
    
    def stop(self):
        """Stop the model container."""
        print("=" * 80)
        print(f"🛑 Stopping Container: {self.container_name}")
        print("=" * 80)
        
        container_id = self._get_container_id()
        
        if not container_id:
            print(f"\n⚠️  Container '{self.container_name}' is not running\n")
            return
        
        print(f"\n🐳 Stopping container: {container_id}")
        result = self._run_command(f"docker stop {self.container_name}")
        
        if result.returncode == 0:
            print(f"✅ Container stopped successfully!\n")
        else:
            print(f"❌ Failed to stop container: {result.stderr}\n")
            sys.exit(1)
    
    def restart(self):
        """Restart the model container."""
        print("=" * 80)
        print(f"🔄 Restarting Container: {self.container_name}")
        print("=" * 80)
        print()
        
        if self._get_container_id():
            self.stop()
            time.sleep(2)
        
        self.start()
    
    def logs(self, follow=False):
        """Show container logs."""
        print("=" * 80)
        print(f"📋 Container Logs: {self.container_name}")
        print("=" * 80)
        print()
        
        if not self._get_container_id():
            print(f"❌ Container '{self.container_name}' is not running\n")
            return
        
        follow_flag = "-f" if follow else ""
        cmd = f"docker logs {follow_flag} {self.container_name}"
        
        if follow:
            print("📡 Following logs (Ctrl+C to exit)...\n")
        
        self._run_command(cmd, capture_output=False)

def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Manage HuggingFace Model Docker containers with multiple inference engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a model with vLLM (default engine)
  python docker_hf.py start --model Qwen/Qwen3-30B-A3B-Thinking-2507
  
  # Start with specific engine
  python docker_hf.py start --model meta-llama/Llama-3-8B --engine sglang
  
  # Start with custom settings
  python docker_hf.py start --model MODEL_NAME --engine vllm \\
    --port 8001 --gpu-memory 0.9 --tp-size 2 --max-model-len 16384
  
  # Check container status
  python docker_hf.py status --container-name qwen3-30b-vllm
  
  # Stop the container
  python docker_hf.py stop --container-name qwen3-30b-vllm
  
  # View logs
  python docker_hf.py logs --container-name qwen3-30b-vllm -f
  
Supported engines:
  - vllm: Fast and flexible inference with PagedAttention (recommended for most use cases)
  - sglang: Optimized for structured generation and complex prompts
  - trtllm: NVIDIA TensorRT-LLM with maximum performance (best for production)
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List engines command
    list_parser = subparsers.add_parser('list-engines', help='List all available inference engines')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a model container')
    start_parser.add_argument('--model', required=True, help='HuggingFace model name (e.g., meta-llama/Llama-3-8B)')
    start_parser.add_argument('--engine', default=DEFAULT_OPT_ENGINE, 
                             choices=list(ENGINE_CONFIGS.keys()),
                             help=f'Inference engine to use (default: {DEFAULT_OPT_ENGINE})')
    start_parser.add_argument('--cache-dir', help='Cache directory (default: .cache/hf in current directory)')
    start_parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'Port to expose (default: {DEFAULT_PORT})')
    start_parser.add_argument('--gpu-memory', type=float, default=DEFAULT_GPU_MEMORY, 
                             help=f'GPU memory utilization (default: {DEFAULT_GPU_MEMORY})')
    start_parser.add_argument('--container-name', help='Custom container name (auto-generated if not provided)')
    start_parser.add_argument('--tp-size', type=int, default=1, help='Tensor parallel size (default: 1)')
    start_parser.add_argument('--max-model-len', type=int, help='Maximum model context length')
    start_parser.add_argument('--extra-args', nargs='+', default=[], help='Additional engine-specific arguments')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop a model container')
    stop_parser.add_argument('--container-name', required=True, help='Container name to stop')
    
    # Restart command
    restart_parser = subparsers.add_parser('restart', help='Restart a model container')
    restart_parser.add_argument('--model', required=True, help='HuggingFace model name')
    restart_parser.add_argument('--engine', default=DEFAULT_OPT_ENGINE, 
                               choices=list(ENGINE_CONFIGS.keys()),
                               help=f'Inference engine to use (default: {DEFAULT_OPT_ENGINE})')
    restart_parser.add_argument('--cache-dir', help='Cache directory (default: .cache/hf in current directory)')
    restart_parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'Port to expose (default: {DEFAULT_PORT})')
    restart_parser.add_argument('--container-name', help='Custom container name')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check container status')
    status_parser.add_argument('--container-name', required=True, help='Container name to check')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View container logs')
    logs_parser.add_argument('--container-name', required=True, help='Container name')
    logs_parser.add_argument('-f', '--follow', action='store_true', help='Follow log output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create deployer instance based on command
    if args.command == 'start':
        deployer = HFModelDeployer(
            model=args.model,
            engine=args.engine,
            cache_dir=args.cache_dir,
            port=args.port,
            gpu_memory=args.gpu_memory,
            container_name=args.container_name,
            tp_size=args.tp_size,
            max_model_len=args.max_model_len,
            extra_args=args.extra_args
        )
        deployer.start()
    elif args.command == 'restart':
        deployer = HFModelDeployer(
            model=args.model,
            engine=args.engine,
            cache_dir=args.cache_dir,
            port=args.port,
            container_name=args.container_name
        )
        deployer.restart()
    elif args.command == 'stop':
        # For stop/status/logs, we just need container name
        deployer = HFModelDeployer(
            model="dummy",  # Not used for these commands
            container_name=args.container_name
        )
        deployer.stop()
    elif args.command == 'status':
        deployer = HFModelDeployer(
            model="dummy",
            container_name=args.container_name
        )
        deployer.status()
    elif args.command == 'logs':
        deployer = HFModelDeployer(
            model="dummy",
            container_name=args.container_name
        )
        deployer.logs(follow=args.follow)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
