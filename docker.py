#!/usr/bin/env python3

"""
Unified Model Deployment Manager
Supports multiple deployment methods (HF, Triton, NIM, UNIM) and engines (vLLM, SGLang, TensorRT-LLM, Python).

Usage:
    python docker.py start --method METHOD --model MODEL_NAME --engine ENGINE [OPTIONS]
    python docker.py stop --container-name NAME
    python docker.py restart --method METHOD --model MODEL_NAME --engine ENGINE [OPTIONS]
    python docker.py status --container-name NAME
    python docker.py logs --container-name NAME [-f]
"""

import os
import sys
import time
import argparse
import subprocess
import re
import json
from pathlib import Path
from abc import ABC, abstractmethod

# ============================================================================
# Version Configurations
# ============================================================================
VLLM_VERSION = "v0.12.0"
SGLANG_VERSION = "v0.5.6.post2"
TRTLLM_VERSION = "1.2.0rc4"
TRITON_VERSION = "25.11"
NEMOFW_VERSION = "25.11"
NIM_VERSION = "latest"  # To be implemented
UNIM_VERSION = "latest"  # To be implemented

# ============================================================================
# Docker Image Configurations
# ============================================================================
# Direct engine images (for HF method)
VLLM_IMAGE = f"vllm/vllm-openai:{VLLM_VERSION}"
SGLANG_IMAGE = f"lmsysorg/sglang:{SGLANG_VERSION}"
TRTLLM_IMAGE = f"nvcr.io/nvidia/tensorrt-llm/release:{TRTLLM_VERSION}"

# Triton images (for Triton method)
TRITON_VLLM_IMAGE = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION}-vllm-python-py3"
TRITON_PYTHON_IMAGE = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION}-trtllm-python-py3"
TRITON_TRTLLM_IMAGE = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION}-trtllm-python-py3"

# NIM images (for NIM method - to be implemented)
NIM_IMAGE = f"nvcr.io/nim/nvidia/nim:{NIM_VERSION}"
UNIM_IMAGE = f"nvcr.io/nim/nvidia/llm-nim:{UNIM_VERSION}"

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
    "triton": ["vllm", "python", "trtllm"],
    "nim": ["vllm"],  # To be implemented
    "unim": ["vllm"],  # To be implemented
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


# ============================================================================
# HuggingFace Direct Deployment (vLLM, SGLang, TensorRT-LLM)
# ============================================================================
class HFModelDeployer(BaseModelDeployer):
    """Direct HuggingFace model deployment with inference engines."""
    
    def __init__(self, model, engine, cache_dir=None, port=DEFAULT_PORT,
                 gpu_memory=DEFAULT_GPU_MEMORY, container_name=None, tp_size=1,
                 max_model_len=None, extra_args=None):
        super().__init__(
            method="hf",
            model=model,
            engine=engine,
            cache_dir=cache_dir,
            port=port,
            gpu_memory=gpu_memory,
            container_name=container_name,
            tp_size=tp_size,
            max_model_len=max_model_len,
            extra_args=extra_args
        )
    
    def _generate_container_name(self) -> str:
        sanitized_model = re.sub(r'[^a-zA-Z0-9_-]', '-', self.model.split('/')[-1].lower())
        return f"{sanitized_model}-hf-{self.engine}"
    
    def _get_health_endpoint(self) -> str:
        return "/v1/models"
    
    def _build_vllm_command(self) -> str:
        """Build Docker command for vLLM engine."""
        cmd_args = [
            f"--model {self.model}",
            f"--gpu-memory-utilization {self.gpu_memory}",
            f"--max-model-len {self.max_model_len}",
            "--trust-remote-code",
            "--enable-chunked-prefill",
        ]
        
        if self.tp_size > 1:
            cmd_args.append(f"--tensor-parallel-size {self.tp_size}")
        
        cmd_args.extend(self.extra_args)
        
        return f"""docker run -d \\
    --name {self.container_name} \\
    --runtime=nvidia \\
    --gpus all \\
    --shm-size=16g \\
    -e HF_TOKEN={os.getenv('HF_TOKEN', '')} \\
    -e HUGGING_FACE_HUB_TOKEN={os.getenv('HF_TOKEN', '')} \\
    -e HF_HOME={CONTAINER_CACHE_PATH} \\
    -p {self.port}:8000 \\
    -v "{self.cache_dir}:{CONTAINER_CACHE_PATH}" \\
    {VLLM_IMAGE} \\
    {' '.join(cmd_args)}"""
    
    def _build_sglang_command(self) -> str:
        """Build Docker command for SGLang engine."""
        cmd_args = [
            f"--model-path {self.model}",
            "--host 0.0.0.0",
            "--port 8000",
            f"--mem-frac {self.gpu_memory}",
            f"--context-length {self.max_model_len}",
            "--trust-remote-code",
        ]
        
        if self.tp_size > 1:
            cmd_args.append(f"--tp {self.tp_size}")
        
        cmd_args.extend(self.extra_args)
        
        return f"""docker run -d \\
    --name {self.container_name} \\
    --runtime=nvidia \\
    --gpus all \\
    --shm-size=16g \\
    -e HF_TOKEN={os.getenv('HF_TOKEN', '')} \\
    -e HUGGING_FACE_HUB_TOKEN={os.getenv('HF_TOKEN', '')} \\
    -e HF_HOME={CONTAINER_CACHE_PATH} \\
    -p {self.port}:8000 \\
    -v "{self.cache_dir}:{CONTAINER_CACHE_PATH}" \\
    {SGLANG_IMAGE} \\
    python3 -m sglang.launch_server {' '.join(cmd_args)}"""
    
    def _build_trtllm_command(self) -> str:
        """Build Docker command for TensorRT-LLM engine."""
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
    -e HF_HOME={CONTAINER_CACHE_PATH} \\
    -p {self.port}:8000 \\
    -v "{self.cache_dir}:{CONTAINER_CACHE_PATH}" \\
    {TRTLLM_IMAGE} \\
    trtllm-serve serve {self.model} --max_seq_len {self.max_model_len} --max_num_tokens {self.max_model_len} --host 0.0.0.0"""
    
    def _build_docker_command(self) -> str:
        """Build the appropriate Docker command based on engine."""
        if self.engine == "vllm":
            return self._build_vllm_command()
        elif self.engine == "sglang":
            return self._build_sglang_command()
        elif self.engine == "trtllm":
            return self._build_trtllm_command()
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")


# ============================================================================
# Triton Inference Server Deployment
# ============================================================================
class TritonModelDeployer(BaseModelDeployer):
    """Triton Inference Server deployment with multiple backends."""
    
    def __init__(self, model, engine, cache_dir=None, port=DEFAULT_PORT,
                 gpu_memory=DEFAULT_GPU_MEMORY, container_name=None, tp_size=1,
                 max_model_len=None, extra_args=None, openai_frontend=False,
                 openai_port=DEFAULT_OPENAI_PORT, grpc_port=DEFAULT_GRPC_PORT,
                 metrics_port=DEFAULT_METRICS_PORT):
        self.openai_frontend = openai_frontend
        self.openai_port = openai_port
        self.grpc_port = grpc_port
        self.metrics_port = metrics_port
        super().__init__(
            method="triton",
            model=model,
            engine=engine,
            cache_dir=cache_dir,
            port=port,
            gpu_memory=gpu_memory,
            container_name=container_name,
            tp_size=tp_size,
            max_model_len=max_model_len,
            extra_args=extra_args
        )
        
        # Additional validation for Python backend
        if self.engine == "python":
            print(f"\n⚠️  Note: Python backend is a baseline implementation")
            print(f"   For production, use vllm or trtllm backends\n")
    
    def _generate_container_name(self) -> str:
        sanitized_model = re.sub(r'[^a-zA-Z0-9_-]', '-', self.model.split('/')[-1].lower())
        return f"{sanitized_model}-triton-{self.engine}"
    
    def _get_health_endpoint(self) -> str:
        if self.openai_frontend:
            return "/health/ready"
        return "/v2/health/ready"
    
    def _setup_model_repository(self):
        """Create Triton model repository with proper structure."""
        print(f"\n📦 Setting up Triton model repository...")
        
        model_repo_dir = self.cache_dir / "model_repository"
        model_repo_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = self.model.replace('/', '_')
        model_dir = model_repo_dir / model_name
        model_version_dir = model_dir / "1"
        model_version_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate config.pbtxt
        config_content = self._generate_config_pbtxt(model_name)
        config_file = model_dir / "config.pbtxt"
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"   ✅ Created config.pbtxt for {self.engine} backend")
        
        # Create backend-specific files
        if self.engine == "vllm":
            self._create_vllm_model_json(model_version_dir)
        elif self.engine == "python":
            self._create_python_backend_files(model_dir)
        elif self.engine == "trtllm":
            print(f"   ⚠️  TRT-LLM backend requires pre-built engines")
            print(f"   Place engines in: {model_version_dir}")
        
        self._run_command(f"chmod -R 777 {model_repo_dir}")
        
        return model_repo_dir
    
    def _generate_config_pbtxt(self, model_name):
        """Generate config.pbtxt for Triton model repository."""
        if self.engine == "vllm":
            # vLLM backend configuration for Triton
            # Reference: https://github.com/triton-inference-server/vllm_backend
            # Note: vLLM handles GPU management internally, Triton should only create ONE instance
            return f'''name: "{model_name}"
backend: "vllm"
max_batch_size: 0

model_transaction_policy {{
  decoupled: True
}}

input [
  {{
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }},
  {{
    name: "stream"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  }},
  {{
    name: "sampling_parameters"
    data_type: TYPE_STRING
    dims: [ -1 ]
    optional: true
  }}
]

output [
  {{
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_MODEL
  }}
]

parameters: {{
  key: "model"
  value: {{
    string_value: "{self.model}"
  }}
}}

parameters: {{
  key: "gpu_memory_utilization"
  value: {{
    string_value: "{self.gpu_memory}"
  }}
}}

parameters: {{
  key: "max_model_len"
  value: {{
    string_value: "{self.max_model_len}"
  }}
}}

parameters: {{
  key: "trust_remote_code"
  value: {{
    string_value: "true"
  }}
}}

parameters: {{
  key: "enforce_eager"
  value: {{
    string_value: "true"
  }}
}}

parameters: {{
  key: "enable_chunked_prefill"
  value: {{
    string_value: "true"
  }}
}}
'''
        
        elif self.engine == "python":
            return f'''name: "{model_name}"
backend: "python"
max_batch_size: 32

input [
  {{
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }},
  {{
    name: "max_tokens"
    data_type: TYPE_INT32
    dims: [ -1 ]
    optional: true
  }}
]

output [
  {{
    name: "text_output"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
'''
        
        elif self.engine == "trtllm":
            return f'''name: "{model_name}"
backend: "tensorrtllm"
max_batch_size: 128

model_transaction_policy {{
  decoupled: True
}}

parameters: {{
  key: "gpt_model_type"
  value: {{
    string_value: "inflight_fused_batching"
  }}
}}

parameters: {{
  key: "gpt_model_path"
  value: {{
    string_value: "$TRITON_MODEL_DIRECTORY/1"
  }}
}}

parameters: {{
  key: "tokenizer_dir"
  value: {{
    string_value: "{self.model}"
  }}
}}

parameters: {{
  key: "kv_cache_free_gpu_mem_fraction"
  value: {{
    string_value: "{self.gpu_memory}"
  }}
}}
'''
        
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
    
    def _create_vllm_model_json(self, model_version_dir):
        """Create model.json file for vLLM backend."""
        print(f"   Creating model.json for vLLM backend...")
        
        # vLLM engine arguments for Triton backend
        # Reference: https://github.com/triton-inference-server/vllm_backend
        vllm_engine_args = {
            "model": self.model,
            "dtype": "auto",
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory,
            "trust_remote_code": True,
            "enforce_eager": True,  # Disable torch.compile for stability
            "enable_chunked_prefill": True,
            "disable_log_stats": False,
        }
        
        if self.tp_size > 1:
            vllm_engine_args["tensor_parallel_size"] = self.tp_size
        
        model_json_path = model_version_dir / "model.json"
        with open(model_json_path, 'w') as f:
            json.dump(vllm_engine_args, f, indent=2)
        
        print(f"   ✅ Created model.json")
    
    def _create_python_backend_files(self, model_dir):
        """Create Python backend model.py (simple echo implementation)."""
        print(f"   Creating Python backend files...")
        
        model_version_dir = model_dir / "1"
        model_py = model_version_dir / "model.py"
        
        python_code = f'''import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    """Simple Python backend for baseline comparison."""
    
    def initialize(self, args):
        """Initialize the model."""
        self.model_name = "{self.model}"
        print(f"Python backend initialized: {{self.model_name}}")
    
    def execute(self, requests):
        """Execute inference (echo mode for testing)."""
        responses = []
        
        for request in requests:
            try:
                text_input = pb_utils.get_input_tensor_by_name(request, "text_input")
                text_input_str = text_input.as_numpy()[0].decode('utf-8')
                
                output_text = f"Echo: {{text_input_str[:100]}}..."
                
                output_tensor = pb_utils.Tensor(
                    "text_output",
                    np.array([output_text.encode('utf-8')], dtype=object)
                )
                
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
            except Exception as e:
                error_tensor = pb_utils.Tensor(
                    "text_output",
                    np.array([f"Error: {{str(e)}}".encode('utf-8')], dtype=object)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[error_tensor]))
        
        return responses
    
    def finalize(self):
        """Clean up resources."""
        print("Python backend finalized")
'''
        
        with open(model_py, 'w') as f:
            f.write(python_code)
        
        print(f"   ✅ Created model.py for Python backend")
    
    def _build_docker_command(self) -> str:
        """Build Docker command for Triton Server."""
        model_repo_dir = self._setup_model_repository()
        
        env_vars = [
            f"-e HF_TOKEN={os.getenv('HF_TOKEN', '')}",
            f"-e HUGGING_FACE_HUB_TOKEN={os.getenv('HF_TOKEN', '')}",
            f"-e HF_HOME={CONTAINER_CACHE_PATH}",
        ]
        
        if self.engine == "vllm":
            env_vars.extend([
                "-e VLLM_LOGGING_LEVEL=INFO",
                "-e TORCH_COMPILE_DISABLE=1",
            ])
        
        env_vars_str = " \\\n    ".join(env_vars)
        
        if self.openai_frontend:
            cache_cleanup = ""
            if self.engine == "vllm":
                cache_cleanup = "rm -rf /root/.triton/cache /root/.cache/torch_extensions/* /root/.cache/torch_inductor/* && "
            
            server_cmd = f"bash -c '{cache_cleanup}cd /opt/tritonserver/python/openai && pip install -q /opt/tritonserver/python/triton*.whl && pip install -q -r requirements.txt && python3 openai_frontend/main.py --model-repository /models --tokenizer {self.model}'"
            port_mappings = f"-p {self.openai_port}:9000"
        else:
            server_cmd = "tritonserver --model-repository=/models --http-port=8000 --grpc-port=8001 --metrics-port=8002"
            port_mappings = f"-p {self.port}:8000 \\\n    -p {self.grpc_port}:8001 \\\n    -p {self.metrics_port}:8002"
        
        # Select appropriate image
        if self.engine == "vllm":
            image = TRITON_VLLM_IMAGE
        elif self.engine == "python":
            image = TRITON_PYTHON_IMAGE
        elif self.engine == "trtllm":
            image = TRITON_TRTLLM_IMAGE
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
        
        return f"""docker run -d \\
    --name {self.container_name} \\
    --runtime=nvidia \\
    --gpus all \\
    --ipc host \\
    --shm-size=16g \\
    {env_vars_str} \\
    {port_mappings} \\
    -v "{model_repo_dir}:/models" \\
    -v "{self.cache_dir}:{CONTAINER_CACHE_PATH}" \\
    {image} \\
    {server_cmd}"""


# ============================================================================
# NVIDIA NIM Deployment (To be implemented)
# ============================================================================
class NIMModelDeployer(BaseModelDeployer):
    """NVIDIA NIM deployment (optimized containers)."""
    
    def __init__(self, model, engine, cache_dir=None, port=DEFAULT_PORT,
                 gpu_memory=DEFAULT_GPU_MEMORY, container_name=None, tp_size=1,
                 max_model_len=None, extra_args=None):
        super().__init__(
            method="nim",
            model=model,
            engine=engine,
            cache_dir=cache_dir,
            port=port,
            gpu_memory=gpu_memory,
            container_name=container_name,
            tp_size=tp_size,
            max_model_len=max_model_len,
            extra_args=extra_args
        )
    
    def _generate_container_name(self) -> str:
        sanitized_model = re.sub(r'[^a-zA-Z0-9_-]', '-', self.model.split('/')[-1].lower())
        return f"{sanitized_model}-nim-{self.engine}"
    
    def _get_health_endpoint(self) -> str:
        return "/v1/health"
    
    def _build_docker_command(self) -> str:
        raise NotImplementedError("NIM deployment is not yet implemented")
    
    def start(self):
        print("\n❌ Error: NIM deployment is not yet implemented\n")
        sys.exit(1)


# ============================================================================
# Universal NIM Deployment (To be implemented)
# ============================================================================
class UNIMModelDeployer(BaseModelDeployer):
    """Universal NIM deployment (generic container wrapper)."""
    
    def __init__(self, model, engine, cache_dir=None, port=DEFAULT_PORT,
                 gpu_memory=DEFAULT_GPU_MEMORY, container_name=None, tp_size=1,
                 max_model_len=None, extra_args=None):
        super().__init__(
            method="unim",
            model=model,
            engine=engine,
            cache_dir=cache_dir,
            port=port,
            gpu_memory=gpu_memory,
            container_name=container_name,
            tp_size=tp_size,
            max_model_len=max_model_len,
            extra_args=extra_args
        )
    
    def _generate_container_name(self) -> str:
        sanitized_model = re.sub(r'[^a-zA-Z0-9_-]', '-', self.model.split('/')[-1].lower())
        return f"{sanitized_model}-unim-{self.engine}"
    
    def _get_health_endpoint(self) -> str:
        return "/v1/health"
    
    def _build_docker_command(self) -> str:
        raise NotImplementedError("UNIM deployment is not yet implemented")
    
    def start(self):
        print("\n❌ Error: UNIM deployment is not yet implemented\n")
        sys.exit(1)


# ============================================================================
# Factory Function
# ============================================================================
def create_deployer(method, model, engine, **kwargs):
    """Factory function to create the appropriate deployer."""
    method = method.lower()
    
    if method == "hf":
        return HFModelDeployer(model, engine, **kwargs)
    elif method == "triton":
        return TritonModelDeployer(model, engine, **kwargs)
    elif method == "nim":
        return NIMModelDeployer(model, engine, **kwargs)
    elif method == "unim":
        return UNIMModelDeployer(model, engine, **kwargs)
    else:
        print(f"\n❌ Error: Unknown method '{method}'")
        print(f"   Supported methods: {', '.join(SUPPORTED_METHODS)}\n")
        sys.exit(1)


# ============================================================================
# Main CLI
# ============================================================================
def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Model Deployment Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # HuggingFace with vLLM
  python docker.py start --method hf --model Qwen/Qwen3-30B-A3B-Thinking-2507 --engine vllm
  
  # Triton with vLLM backend + OpenAI frontend
  python docker.py start --method triton --model Qwen/Qwen3-30B-A3B-Thinking-2507 --engine vllm --openai-frontend
  
  # Check status
  python docker.py status --container-name hf-qwen3-30b-vllm
  
  # Stop container
  python docker.py stop --container-name hf-qwen3-30b-vllm
  
  # View logs
  python docker.py logs --container-name hf-qwen3-30b-vllm -f

Supported methods:
  - hf: Direct HuggingFace deployment (engines: vllm, sglang, trtllm)
  - triton: Triton Inference Server (engines: vllm, python, trtllm)
  - nim: NVIDIA NIM (to be implemented)
  - unim: Universal NIM (to be implemented)
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a container')
    start_parser.add_argument('--method', default=DEFAULT_METHOD, 
                             choices=SUPPORTED_METHODS,
                             help=f'Deployment method (default: {DEFAULT_METHOD})')
    start_parser.add_argument('--model', required=True, help='HuggingFace model name')
    start_parser.add_argument('--engine', default=DEFAULT_ENGINE,
                             help=f'Inference engine (default: {DEFAULT_ENGINE})')
    start_parser.add_argument('--cache-dir', help='Cache directory')
    start_parser.add_argument('--port', type=int, default=DEFAULT_PORT, 
                             help=f'Port (default: {DEFAULT_PORT})')
    start_parser.add_argument('--gpu-memory', type=float, default=DEFAULT_GPU_MEMORY,
                             help=f'GPU memory utilization (default: {DEFAULT_GPU_MEMORY})')
    start_parser.add_argument('--container-name', help='Custom container name')
    start_parser.add_argument('--tp-size', type=int, default=1, 
                             help='Tensor parallel size (default: 1)')
    start_parser.add_argument('--max-model-len', type=int,
                             help=f'Maximum model length (default: {DEFAULT_MAX_MODEL_LEN})')
    start_parser.add_argument('--extra-args', nargs='+', default=[],
                             help='Additional engine-specific arguments')
    
    # Triton-specific arguments
    start_parser.add_argument('--openai-frontend', action='store_true',
                             help='Use OpenAI-compatible frontend (Triton only)')
    start_parser.add_argument('--openai-port', type=int, default=DEFAULT_OPENAI_PORT,
                             help=f'OpenAI frontend port (default: {DEFAULT_OPENAI_PORT})')
    start_parser.add_argument('--grpc-port', type=int, default=DEFAULT_GRPC_PORT,
                             help=f'GRPC port (default: {DEFAULT_GRPC_PORT})')
    start_parser.add_argument('--metrics-port', type=int, default=DEFAULT_METRICS_PORT,
                             help=f'Metrics port (default: {DEFAULT_METRICS_PORT})')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop a container')
    stop_parser.add_argument('--container-name', required=True, help='Container name')
    
    # Restart command
    restart_parser = subparsers.add_parser('restart', help='Restart a container')
    restart_parser.add_argument('--method', default=DEFAULT_METHOD,
                               choices=SUPPORTED_METHODS)
    restart_parser.add_argument('--model', required=True)
    restart_parser.add_argument('--engine', default=DEFAULT_ENGINE)
    restart_parser.add_argument('--cache-dir', help='Cache directory')
    restart_parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    restart_parser.add_argument('--container-name', help='Custom container name')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check container status')
    status_parser.add_argument('--container-name', required=True)
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View container logs')
    logs_parser.add_argument('--container-name', required=True)
    logs_parser.add_argument('-f', '--follow', action='store_true', help='Follow logs')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create deployer and execute command
    if args.command == 'start':
        deployer_kwargs = {
            'cache_dir': args.cache_dir,
            'port': args.port,
            'gpu_memory': args.gpu_memory,
            'container_name': args.container_name,
            'tp_size': args.tp_size,
            'max_model_len': args.max_model_len,
            'extra_args': args.extra_args,
        }
        
        # Add Triton-specific arguments
        if args.method == 'triton':
            deployer_kwargs.update({
                'openai_frontend': args.openai_frontend,
                'openai_port': args.openai_port,
                'grpc_port': args.grpc_port,
                'metrics_port': args.metrics_port,
            })
        
        deployer = create_deployer(args.method, args.model, args.engine, **deployer_kwargs)
        deployer.start()
        
    elif args.command == 'restart':
        deployer_kwargs = {
            'cache_dir': args.cache_dir,
            'port': args.port,
            'container_name': args.container_name,
        }
        deployer = create_deployer(args.method, args.model, args.engine, **deployer_kwargs)
        deployer.restart()
        
    elif args.command == 'stop':
        # For stop, we need a dummy deployer just to use the container name
        deployer = HFModelDeployer(
            model="dummy",
            engine="vllm",
            container_name=args.container_name
        )
        deployer.stop()
        
    elif args.command == 'status':
        deployer = HFModelDeployer(
            model="dummy",
            engine="vllm",
            container_name=args.container_name
        )
        deployer.status()
        
    elif args.command == 'logs':
        deployer = HFModelDeployer(
            model="dummy",
            engine="vllm",
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

