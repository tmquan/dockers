#!/usr/bin/env python3

"""
Base Model Deployer - Abstract base class for model deployment managers.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod

# ============================================================================
# Default Configurations
# ============================================================================
DEFAULT_PORT = 8000
DEFAULT_GRPC_PORT = 8001
DEFAULT_METRICS_PORT = 8002
DEFAULT_OPENAI_PORT = 9000
DEFAULT_GPU_MEMORY = 0.9
DEFAULT_METHOD = "hf"
DEFAULT_ENGINE = "vllm"
DEFAULT_MAX_MODEL_LEN = 32768
CONTAINER_CACHE_PATH = "/root/.cache/huggingface"

# ============================================================================
# Supported Methods and Engines
# ============================================================================
SUPPORTED_METHODS = ["hf", "triton", "nim", "unim"]
SUPPORTED_ENGINES = {
    "hf": ["vllm", "sglang", "trtllm"],
    "triton": ["python", "vllm", "trtllm"], 
    "nim": ["vllm"],  # To be implemented
    "unim": ["python", "vllm", "trtllm"],  # Python (safetensors), vLLM, TensorRT-LLM
}


# ============================================================================
# Base Model Deployer (Abstract Base Class)
# ============================================================================
class BaseModelDeployer(ABC):
    """Abstract base class for model deployment managers."""
    
    def __init__(self, method, model, engine, cache_dir=None, port=DEFAULT_PORT,
                 gpu_memory=DEFAULT_GPU_MEMORY, container_name=None, tp_size=1,
                 max_model_len=None, extra_args=None):
        self.method = method.lower()
        self.model = model
        self.engine = engine.lower()
        self.port = port
        self.gpu_memory = gpu_memory
        self.tp_size = tp_size
        self.max_model_len = max_model_len or DEFAULT_MAX_MODEL_LEN
        self.extra_args = extra_args or []
        
        # Validate engine support first
        self._validate_engine()
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        else:
            self.cache_dir = self._get_default_cache_dir()
        
        # Generate container name if not provided
        if container_name:
            self.container_name = container_name
        else:
            self.container_name = self._generate_container_name()
    
    def _get_default_cache_dir(self) -> Path:
        """Get the default cache directory for this deployment method."""
        return Path.cwd() / ".cache" / self.method
    
    def _validate_engine(self):
        """Validate that the engine is supported for this deployment method."""
        if self.method not in SUPPORTED_ENGINES:
            print(f"\n❌ Error: Unknown method '{self.method}'")
            print(f"   Supported methods: {', '.join(SUPPORTED_ENGINES.keys())}\n")
            sys.exit(1)
        
        if self.engine not in SUPPORTED_ENGINES[self.method]:
            print(f"\n❌ Error: Engine '{self.engine}' not supported for {self.method} method")
            print(f"   Supported engines: {', '.join(SUPPORTED_ENGINES[self.method])}\n")
            sys.exit(1)
    
    @abstractmethod
    def _generate_container_name(self) -> str:
        """Generate a container name based on model and engine."""
        pass
    
    @abstractmethod
    def _build_docker_command(self) -> str:
        """Build the Docker run command."""
        pass
    
    @abstractmethod
    def _get_health_endpoint(self) -> str:
        """Get the health check endpoint for this deployment method."""
        pass
    
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
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._run_command(f"chmod -R 777 {self.cache_dir}")
        print(f"   ✅ Cache directory ready")
    
    def status(self):
        """Check the status of the model container."""
        print("=" * 80)
        print(f"🔍 Checking Container Status: {self.container_name}")
        print("=" * 80)
        
        container_id = self._get_container_id()
        status = self._get_container_status()
        
        if not status:
            print(f"\n❌ Container '{self.container_name}' not found\n")
            return False
        
        if container_id:
            print(f"\n✅ Container '{self.container_name}' is RUNNING")
            print(f"   Container ID: {container_id}")
            print(f"   Status: {status}")
            print(f"   Model: {self.model}")
            print(f"   Engine: {self.engine}")
            print(f"   Service URL: http://localhost:{self.port}")
            
            health_endpoint = self._get_health_endpoint()
            print(f"   Health check: http://localhost:{self.port}{health_endpoint}")
            
            # Test if API is responsive
            test_result = self._run_command(
                f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{self.port}{health_endpoint} --max-time 2",
                check=False
            )
            if test_result.stdout.strip() == '200':
                print(f"   API Status: ✅ Ready")
            else:
                print(f"   API Status: ⏳ Starting up")
        else:
            print(f"\n⚠️  Container '{self.container_name}' exists but is NOT RUNNING")
            print(f"   Status: {status}")
        
        print()
        return container_id is not None
    
    def start(self):
        """Start the model container."""
        print("=" * 80)
        print(f"🚀 Starting Container: {self.model}")
        print(f"   Engine: {self.engine}")
        print("=" * 80)
        
        self._check_docker()
        
        if self._get_container_id():
            print("\n⚠️  Container is already running!")
            self.status()
            return
        
        has_nvidia = self._check_nvidia_runtime()
        if not has_nvidia:
            print("\n❌ Error: NVIDIA runtime is required!")
            sys.exit(1)
        
        self._setup_cache_directory()
        
        # Remove stopped container if exists
        status = self._get_container_status()
        if status and "Exited" in status:
            print(f"\n🔄 Found stopped container, removing it...")
            self._run_command(f"docker rm {self.container_name}")
        
        # Build and execute Docker command
        print(f"\n🐳 Starting Docker container...")
        print(f"   Container name: {self.container_name}")
        print(f"   Port: {self.port}")
        print(f"   GPU memory: {self.gpu_memory}")
        if self.tp_size > 1:
            print(f"   Tensor parallel size: {self.tp_size}")
        
        docker_cmd = self._build_docker_command()
        result = self._run_command(docker_cmd.strip())
        
        if result.returncode == 0:
            print(f"\n✅ Container started successfully!")
            print(f"   Container ID: {result.stdout.strip()}")
            print(f"\n⏳ Model is initializing... This may take several minutes.")
            print(f"   Service URL: http://localhost:{self.port}")
            print(f"\n💡 Use 'python docker.py status --container-name {self.container_name}' to check status")
            print(f"   Use 'python docker.py logs --container-name {self.container_name} -f' to watch logs\n")
        else:
            print(f"\n❌ Failed to start container!")
            print(f"   Error: {result.stderr}\n")
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

