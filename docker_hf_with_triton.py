#!/usr/bin/env python3

"""
NVIDIA Triton Server Model Deployment Manager
This script manages model deployments using Triton Inference Server with multiple backends.

Supports TWO modes:
1. OpenAI-compatible frontend (--openai-frontend) - Exposes /v1/chat/completions, /v1/completions
2. Native Triton v2 protocol (default) - Exposes /v2/models/{model}/infer

Supported backends: vllm, trtllm

Usage:
    python docker_hf_with_triton.py start --model MODEL_NAME --backend BACKEND [OPTIONS]
    python docker_hf_with_triton.py start --model MODEL_NAME --backend BACKEND --openai-frontend [OPTIONS]
    python docker_hf_with_triton.py stop [--container-name NAME]
    python docker_hf_with_triton.py status [--container-name NAME]
    python docker_hf_with_triton.py logs [--container-name NAME] [-f]
"""

import os
import sys
import time
import argparse
import subprocess
import re
from pathlib import Path

# ============================================================================
# Version Configuration
# ============================================================================
TRITON_VERSION = "25.11-py3"  # NVIDIA Triton Server version

# ============================================================================
# Default Configurations
# ============================================================================
DEFAULT_PORT = 8000
DEFAULT_GRPC_PORT = 8001
DEFAULT_METRICS_PORT = 8002
DEFAULT_GPU_MEMORY = 0.85
DEFAULT_BACKEND = "trtllm"
CONTAINER_CACHE_PATH = "/root/.cache/huggingface"

# ============================================================================
# Backend Configurations
# ============================================================================
BACKEND_CONFIGS = {
    "trtllm": {
        "display_name": "TensorRT-LLM",
        "health_endpoint": "/v2/health/ready",
        "openai_health_endpoint": "/health/ready",
        "supports_openai_frontend": True,
        "api_type": "triton_v2",
    },
    "vllm": {
        "display_name": "vLLM (Triton Backend)",
        "health_endpoint": "/v2/health/ready",
        "openai_health_endpoint": "/health/ready",
        "supports_openai_frontend": True,
        "api_type": "triton_v2",
    },
    "python": {
        "display_name": "Python Backend",
        "health_endpoint": "/v2/health/ready",
        "openai_health_endpoint": "/health/ready",
        "supports_openai_frontend": False,  # Python backend requires custom implementation
        "api_type": "triton_v2",
    },
}

class TritonModelDeployer:
    """Manages Triton Server model container lifecycle with multiple backend support."""
    
    def __init__(self, model, backend=DEFAULT_BACKEND, cache_dir=None, 
                 port=DEFAULT_PORT, grpc_port=DEFAULT_GRPC_PORT, metrics_port=DEFAULT_METRICS_PORT,
                 gpu_memory=DEFAULT_GPU_MEMORY, container_name=None, tp_size=1, 
                 max_model_len=None, extra_args=None, openai_frontend=False, openai_port=9000):
        self.model = model
        self.backend = backend.lower()
        self.openai_frontend = openai_frontend
        self.openai_port = openai_port
        
        # Use .cache/triton in current directory
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        else:
            self.cache_dir = Path.cwd() / ".cache" / "triton"
        
        self.port = port
        self.grpc_port = grpc_port
        self.metrics_port = metrics_port
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
            self.container_name = f"hf-triton-{sanitized_model}-{self.backend}"
        
        # Validate backend
        if self.backend not in BACKEND_CONFIGS:
            print(f"\n❌ Error: Unsupported backend '{self.backend}'")
            print(f"   Supported backends: {', '.join(BACKEND_CONFIGS.keys())}\n")
            sys.exit(1)
        
        self.backend_config = BACKEND_CONFIGS[self.backend]
        
        # Select appropriate Triton image based on backend
        if self.backend == "vllm":
            self.image = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION.replace('-py3', '')}-vllm-python-py3"
        elif self.backend == "trtllm":
            self.image = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION.replace('-py3', '')}-trtllm-python-py3"
        else:
            # Default for python backend and others
            self.image = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION}"
        
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
        
        print(f"   ✅ Cache directory ready")
    
    def _setup_model_repository(self):
        """Create Triton model repository with proper structure."""
        print(f"\n📦 Setting up Triton model repository...")
        
        # Create model repository directory
        model_repo_dir = self.cache_dir / "model_repository"
        model_repo_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize model name for directory (remove special chars)
        model_name = self.model.split('/')[-1].replace('-', '_').replace('.', '_')
        model_dir = model_repo_dir / model_name
        model_version_dir = model_dir / "1"
        model_version_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   Model repository: {model_repo_dir}")
        print(f"   Model name: {model_name}")
        
        # Generate config.pbtxt based on backend
        config_content = self._generate_config_pbtxt(model_name)
        config_file = model_dir / "config.pbtxt"
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"   ✅ Created config.pbtxt for {self.backend} backend")
        
        # Create backend-specific files
        if self.backend == "vllm":
            self._create_vllm_model_json(model_version_dir)
        elif self.backend == "python":
            self._create_python_backend_files(model_dir)
        
        # Set permissions
        self._run_command(f"chmod -R 777 {model_repo_dir}")
        
        return model_repo_dir
    
    def _generate_config_pbtxt(self, model_name):
        """Generate config.pbtxt for Triton model repository."""
        
        if self.backend == "vllm":
            # vLLM backend configuration for Triton
            # Based on: https://github.com/triton-inference-server/vllm_backend
            max_model_len = self.max_model_len if self.max_model_len else 131072
            
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
    kind: KIND_GPU
  }}
]

parameters: {{
  key: "model"
  value: {{
    string_value: "{self.model}"
  }}
}}

parameters: {{
  key: "dtype"
  value: {{
    string_value: "auto"
  }}
}}

parameters: {{
  key: "max_model_len"
  value: {{
    string_value: "{max_model_len}"
  }}
}}

parameters: {{
  key: "gpu_memory_utilization"
  value: {{
    string_value: "{self.gpu_memory}"
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
    string_value: "false"
  }}
}}

parameters: {{
  key: "enable_chunked_prefill"
  value: {{
    string_value: "true"
  }}
}}
'''
        
        elif self.backend == "trtllm":
            # TensorRT-LLM backend for Triton
            # Based on: https://github.com/triton-inference-server/trtllm_backend
            max_model_len = self.max_model_len if self.max_model_len else 131072
            
            return f'''name: "{model_name}"
backend: "trtllm"
max_batch_size: 128

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
    name: "max_tokens"
    data_type: TYPE_INT32
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "bad_words"
    data_type: TYPE_STRING
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "stop_words"
    data_type: TYPE_STRING
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "stream"
    data_type: TYPE_BOOL
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "temperature"
    data_type: TYPE_FP32
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "top_k"
    data_type: TYPE_INT32
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "top_p"
    data_type: TYPE_FP32
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

parameters: {{
  key: "gpt_model_type"
  value: {{
    string_value: "inflight_fused_batching"
  }}
}}

parameters: {{
  key: "gpt_model_path"
  value: {{
    string_value: "{self.model}"
  }}
}}

parameters: {{
  key: "batch_scheduler_policy"
  value: {{
    string_value: "guaranteed_no_evict"
  }}
}}

parameters: {{
  key: "max_queue_delay_microseconds"
  value: {{
    string_value: "1000"
  }}
}}

parameters: {{
  key: "max_beam_width"
  value: {{
    string_value: "1"
  }}
}}

parameters: {{
  key: "max_tokens_in_paged_kv_cache"
  value: {{
    string_value: "{max_model_len}"
  }}
}}

parameters: {{
  key: "max_attention_window_size"
  value: {{
    string_value: "{max_model_len}"
  }}
}}

parameters: {{
  key: "enable_kv_cache_reuse"
  value: {{
    string_value: "false"
  }}
}}

parameters: {{
  key: "enable_chunked_context"
  value: {{
    string_value: "false"
  }}
}}
'''
        
        elif self.backend == "python":
            # Python backend wrapper for custom inference
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
    name: "parameters"
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
    kind: KIND_GPU
  }}
]

parameters: {{
  key: "EXECUTION_ENV_PATH"
  value: {{
    string_value: "$$TRITON_MODEL_DIRECTORY/python_env.tar.gz"
  }}
}}
'''
        
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _create_vllm_model_json(self, model_version_dir):
        """Create model.json file required by vLLM backend."""
        import json
        
        print(f"   Creating model.json for vLLM backend...")
        
        max_model_len = self.max_model_len if self.max_model_len else 131072
        
        # vLLM engine arguments
        # Based on: https://github.com/triton-inference-server/vllm_backend
        vllm_engine_args = {
            "model": self.model,
            "dtype": "auto",
            "max_model_len": max_model_len,
            "gpu_memory_utilization": self.gpu_memory,
            "trust_remote_code": True,
            "enforce_eager": False,
            "enable_chunked_prefill": True,
        }
        
        # Add tensor parallel size if specified
        if self.tp_size > 1:
            vllm_engine_args["tensor_parallel_size"] = self.tp_size
        
        model_json_path = model_version_dir / "model.json"
        with open(model_json_path, 'w') as f:
            json.dump(vllm_engine_args, f, indent=2)
        
        print(f"   ✅ Created model.json with vLLM engine args")
    
    def _create_python_backend_files(self, model_dir):
        """Create Python backend model.py for custom inference."""
        print(f"   Creating Python backend files...")
        
        model_version_dir = model_dir / "1"
        model_py = model_version_dir / "model.py"
        
        # Create a Python backend wrapper that loads HuggingFace models
        python_code = f'''import json
import numpy as np
import triton_python_backend_utils as pb_utils

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    # Will be installed in container
    pass

class TritonPythonModel:
    """Python backend model for Triton with HuggingFace support."""
    
    def initialize(self, args):
        """Initialize the model."""
        self.model_config = json.loads(args['model_config'])
        
        # Initialize HuggingFace model
        self.model_name = "{self.model}"
        print(f"Loading model: {{self.model_name}}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Model loaded successfully on {{self.device}}")
        except Exception as e:
            print(f"Error loading model: {{e}}")
            # Fallback to echo mode for testing
            self.model = None
            self.tokenizer = None
    
    def execute(self, requests):
        """Execute inference on a batch of requests."""
        responses = []
        
        for request in requests:
            try:
                # Get input text
                text_input = pb_utils.get_input_tensor_by_name(request, "text_input")
                text_input_str = text_input.as_numpy()[0].decode('utf-8')
                
                # Parse parameters if provided
                params = {{}}
                try:
                    params_tensor = pb_utils.get_input_tensor_by_name(request, "parameters")
                    if params_tensor is not None:
                        params_str = params_tensor.as_numpy()[0].decode('utf-8')
                        params = json.loads(params_str)
                except:
                    pass
                
                max_tokens = params.get('max_tokens', 100)
                temperature = params.get('temperature', 0.7)
                
                # Generate response
                if self.model is not None and self.tokenizer is not None:
                    inputs = self.tokenizer(text_input_str, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            do_sample=True
                        )
                    
                    output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # Fallback echo mode
                    output_text = f"Echo (Python backend): {{text_input_str}}"
                
                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "text_output",
                    np.array([output_text.encode('utf-8')], dtype=object)
                )
                
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)
                
            except Exception as e:
                error_message = f"Error in inference: {{str(e)}}"
                print(error_message)
                
                error_tensor = pb_utils.Tensor(
                    "text_output",
                    np.array([error_message.encode('utf-8')], dtype=object)
                )
                
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[error_tensor]
                ))
        
        return responses
    
    def finalize(self):
        """Clean up resources."""
        print("Cleaning up model resources")
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
'''
        
        with open(model_py, 'w') as f:
            f.write(python_code)
        
        print(f"   ✅ Created model.py for Python backend")
    
    def _build_triton_command(self):
        """Build Docker command for Triton Server with model repository."""
        
        # Setup model repository first
        model_repo_dir = self._setup_model_repository()
        
        # Environment variables for HuggingFace
        env_vars = [
            f"-e HF_TOKEN={os.getenv('HF_TOKEN', '')}",
            f"-e HUGGING_FACE_HUB_TOKEN={os.getenv('HF_TOKEN', '')}",
            f"-e HF_HOME=/root/.cache/huggingface",
        ]
        
        # Add TRTLLM_ORCHESTRATOR for TensorRT-LLM
        if self.backend == "trtllm":
            env_vars.append("-e TRTLLM_ORCHESTRATOR=1")
        
        env_vars_str = " \\\n    ".join(env_vars)
        
        # Choose command based on frontend mode
        if self.openai_frontend:
            # Use OpenAI-compatible frontend
            # Reference: https://github.com/triton-inference-server/server/tree/main/python/openai
            # Note: The OpenAI frontend listens on port 9000 by default internally
            # We map the container's 9000 to the host's desired openai_port
            cmd_parts = [
                "bash -c 'cd /opt/tritonserver/python/openai &&",
                "pip install -q /opt/tritonserver/python/triton*.whl &&",
                "pip install -q -r requirements.txt &&",
                f"python3 openai_frontend/main.py",
                f"--model-repository /models",
                f"--tokenizer {self.model}'",
            ]
            server_cmd = " ".join(cmd_parts)
            
            # Map container port 9000 (OpenAI frontend default) to host port
            port_mappings = f"-p {self.openai_port}:9000"
        else:
            # Use native Triton v2 protocol
            triton_args = [
                "--model-repository=/models",
                "--http-port=8000",
                "--grpc-port=8001",
                "--metrics-port=8002",
                f"--backend-config={self.backend},default-max-batch-size=128",
                "--log-verbose=1",
                "--disable-auto-complete-config",
            ]
            
            server_cmd = "tritonserver " + " ".join(triton_args)
            
            # Map Triton v2 ports
            port_mappings = f"-p {self.port}:8000 \\\n    -p {self.grpc_port}:8001 \\\n    -p {self.metrics_port}:8002"
        
        return f"""docker run -d \\
    --name {self.container_name} \\
    --runtime=nvidia \\
    --gpus all \\
    --ipc host \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    --shm-size=16g \\
    {env_vars_str} \\
    {port_mappings} \\
    -v "{model_repo_dir}:/models" \\
    -v "{self.cache_dir}:/root/.cache/huggingface" \\
    {self.image} \\
    {server_cmd}"""
    
    def status(self):
        """Check the status of the Triton container."""
        print("=" * 80)
        print(f"🔍 Checking Triton Container Status: {self.container_name}")
        print("=" * 80)
        
        container_id = self._get_container_id()
        status = self._get_container_status()
        
        if not status:
            print(f"\n❌ Container '{self.container_name}' not found")
            print(f"   Use 'python docker_hf_with_triton.py start --model <model> --backend <backend>' to create it\n")
            return False
        
        if container_id:
            print(f"\n✅ Container '{self.container_name}' is RUNNING")
            print(f"   Container ID: {container_id}")
            print(f"   Status: {status}")
            print(f"   Model: {self.model}")
            print(f"   Backend: {self.backend_config['display_name']}")
            
            if self.openai_frontend:
                print(f"   API Type: OpenAI-compatible")
                print(f"   OpenAI Service: http://localhost:{self.openai_port}")
                print(f"   Endpoints: /v1/chat/completions, /v1/completions")
                health_endpoint = self.backend_config["openai_health_endpoint"]
                health_url = f"http://localhost:{self.openai_port}{health_endpoint}"
            else:
                print(f"   API Type: {self.backend_config['api_type']}")
                print(f"   HTTP Service: http://localhost:{self.port}")
                print(f"   GRPC Service: localhost:{self.grpc_port}")
                print(f"   Metrics: http://localhost:{self.metrics_port}/metrics")
                health_endpoint = self.backend_config["health_endpoint"]
                health_url = f"http://localhost:{self.port}{health_endpoint}"
            
            print(f"   Health check: {health_url}")
            
            # Test if API is responsive
            test_result = self._run_command(
                f"curl -s -o /dev/null -w '%{{http_code}}' {health_url} --max-time 2",
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
        """Start the Triton container."""
        print("=" * 80)
        print(f"🚀 Starting Triton Server with {self.backend_config['display_name']} Backend")
        print(f"   Model: {self.model}")
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
        print(f"\n🔽 Checking for Triton Server image: {self.image}")
        image_check = self._run_command(f"docker image inspect {self.image}")
        if image_check.returncode != 0:
            print(f"   Image not found locally, pulling from registry...")
            print(f"   ⏳ This may take several minutes...")
            pull_result = self._run_command(f"docker pull {self.image}", capture_output=False)
            if pull_result.returncode != 0:
                print(f"\n❌ Failed to pull image. Check your internet connection.\n")
                sys.exit(1)
        else:
            print(f"   ✅ Image found locally")
        
        # Build and execute Docker command
        print(f"\n🐳 Starting Triton container...")
        print(f"   Container name: {self.container_name}")
        print(f"   HTTP port: {self.port}")
        print(f"   GRPC port: {self.grpc_port}")
        print(f"   Metrics port: {self.metrics_port}")
        print(f"   Model: {self.model}")
        print(f"   Backend: {self.backend_config['display_name']}")
        
        if self.openai_frontend:
            print(f"   Frontend: OpenAI-compatible (port {self.openai_port})")
            print(f"   This will expose /v1/chat/completions, /v1/completions endpoints")
        
        print(f"\n⚙️  Configuring Triton Server with {self.backend} backend...")
        print(f"   This will create model repository and config files.")
        
        docker_cmd = self._build_triton_command()
        result = self._run_command(docker_cmd.strip())
        
        if result.returncode == 0:
            print(f"\n✅ Container started!")
            print(f"   Container ID: {result.stdout.strip()}")
            print(f"\n⏳ Triton Server is initializing... This may take several minutes.")
            print(f"   The server will load the model and start the {self.backend} backend.")
            
            if self.openai_frontend:
                print(f"   Service URL (OpenAI-compatible): http://localhost:{self.openai_port}")
                print(f"   Endpoints: /v1/chat/completions, /v1/completions, /v1/embeddings")
            else:
                print(f"   Service URL: http://localhost:{self.port}")
                print(f"   GRPC URL: localhost:{self.grpc_port}")
                print(f"   Metrics: http://localhost:{self.metrics_port}/metrics")
            
            print(f"\n   Use 'python docker_hf_with_triton.py status --container-name {self.container_name}' to check status")
            print(f"   Use 'python docker_hf_with_triton.py logs --container-name {self.container_name} -f' to view logs\n")
        else:
            print(f"\n❌ Failed to start container!")
            print(f"   Error: {result.stderr}\n")
            sys.exit(1)
    
    def stop(self):
        """Stop the Triton container."""
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

def test_all_backends(model, port_base=9000, max_model_len=131072):
    """Test all backends sequentially."""
    backends = ["vllm", "trtllm", "python"]
    
    print("=" * 80)
    print("🧪 BRUTE FORCE TEST: All Triton Backends")
    print("=" * 80)
    print(f"\nModel: {model}")
    print(f"Backends to test: {', '.join(backends)}")
    print(f"Base port: {port_base}")
    print(f"\n⚠️  NOTE: Triton uses v2 Inference API, not OpenAI Chat Completions API\n")
    
    results = {}
    
    for idx, backend in enumerate(backends):
        port = port_base + (idx * 10)
        grpc_port = port + 1
        metrics_port = port + 2
        
        print("\n" + "=" * 80)
        print(f"Testing Backend {idx + 1}/{len(backends)}: {backend.upper()}")
        print("=" * 80)
        
        deployer = TritonModelDeployer(
            model=model,
            backend=backend,
            port=port,
            grpc_port=grpc_port,
            metrics_port=metrics_port,
            max_model_len=max_model_len
        )
        
        try:
            # Start the container
            deployer.start()
            
            # Wait a moment
            print(f"\n⏳ Waiting 10 seconds for initialization...")
            time.sleep(10)
            
            # Check status
            is_running = deployer.status()
            results[backend] = "STARTED" if is_running else "FAILED"
            
            print(f"\n   Result: {results[backend]}")
            
            # Stop the container
            deployer.stop()
            
            # Clean up
            subprocess.run(f"docker rm {deployer.container_name}", shell=True, capture_output=True)
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Test interrupted for {backend}")
            deployer.stop()
            break
        except Exception as e:
            print(f"\n❌ Error testing {backend}: {e}")
            results[backend] = f"ERROR: {e}"
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    for backend, result in results.items():
        status_icon = "✅" if result == "STARTED" else "❌"
        print(f"{status_icon} {backend:15} : {result}")
    print()

def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Manage Triton Server deployments with multiple backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with vLLM backend + OpenAI-compatible frontend (RECOMMENDED for benchmarks)
  python docker_hf_with_triton.py start --model Qwen/Qwen3-30B-A3B-Thinking-2507 --backend vllm --openai-frontend
  
  # Start with TensorRT-LLM backend (default, Triton v2 protocol)
  python docker_hf_with_triton.py start --model Qwen/Qwen3-30B-A3B-Thinking-2507
  
  # Start with vLLM backend + OpenAI frontend on custom port
  python docker_hf_with_triton.py start --model meta-llama/Llama-3-8B --backend vllm --openai-frontend --openai-port 8080
  
  # Check status
  python docker_hf_with_triton.py status --container-name hf-triton-qwen3-30b-trtllm
  
  # Stop container
  python docker_hf_with_triton.py stop --container-name hf-triton-qwen3-30b-trtllm
  
  # Test all backends
  python docker_hf_with_triton.py test-all --model MODEL_NAME

Supported backends:
  - vllm: vLLM backend for Triton
  - trtllm: TensorRT-LLM backend for Triton
  - python: Python backend for custom models (uses HuggingFace Transformers)

API Modes:
  - Default: Triton v2 protocol (/v2/models/{name}/infer)
  - With --openai-frontend: OpenAI-compatible API (/v1/chat/completions)
    Note: Python backend works best with Triton v2 protocol
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a Triton container')
    start_parser.add_argument('--model', required=True, help='HuggingFace model name')
    start_parser.add_argument('--backend', default=DEFAULT_BACKEND, 
                             choices=list(BACKEND_CONFIGS.keys()),
                             help=f'Backend to use (default: {DEFAULT_BACKEND})')
    start_parser.add_argument('--openai-frontend', action='store_true',
                             help='Use OpenAI-compatible frontend (exposes /v1/chat/completions)')
    start_parser.add_argument('--openai-port', type=int, default=9000,
                             help='Port for OpenAI frontend (default: 9000)')
    start_parser.add_argument('--cache-dir', help='Cache directory')
    start_parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'HTTP port (default: {DEFAULT_PORT})')
    start_parser.add_argument('--grpc-port', type=int, default=DEFAULT_GRPC_PORT, help=f'GRPC port (default: {DEFAULT_GRPC_PORT})')
    start_parser.add_argument('--metrics-port', type=int, default=DEFAULT_METRICS_PORT, help=f'Metrics port (default: {DEFAULT_METRICS_PORT})')
    start_parser.add_argument('--gpu-memory', type=float, default=DEFAULT_GPU_MEMORY, help=f'GPU memory utilization (default: {DEFAULT_GPU_MEMORY})')
    start_parser.add_argument('--container-name', help='Custom container name')
    start_parser.add_argument('--tp-size', type=int, default=1, help='Tensor parallel size (default: 1)')
    start_parser.add_argument('--max-model-len', type=int, help='Maximum model context length')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop a container')
    stop_parser.add_argument('--container-name', required=True, help='Container name to stop')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check container status')
    status_parser.add_argument('--container-name', required=True, help='Container name to check')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View container logs')
    logs_parser.add_argument('--container-name', required=True, help='Container name')
    logs_parser.add_argument('-f', '--follow', action='store_true', help='Follow log output')
    
    # Test all backends command
    test_parser = subparsers.add_parser('test-all', help='Test all backends sequentially')
    test_parser.add_argument('--model', required=True, help='HuggingFace model name')
    test_parser.add_argument('--port-base', type=int, default=9000, help='Base port for testing (default: 9000)')
    test_parser.add_argument('--max-model-len', type=int, default=131072, help='Max model length (default: 131072)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle test-all command
    if args.command == 'test-all':
        test_all_backends(args.model, args.port_base, args.max_model_len)
        return
    
    # Create deployer instance based on command
    if args.command == 'start':
        deployer = TritonModelDeployer(
            model=args.model,
            backend=args.backend,
            cache_dir=args.cache_dir,
            port=args.port,
            grpc_port=args.grpc_port,
            metrics_port=args.metrics_port,
            gpu_memory=args.gpu_memory,
            container_name=args.container_name,
            tp_size=args.tp_size,
            max_model_len=args.max_model_len,
            openai_frontend=args.openai_frontend,
            openai_port=args.openai_port
        )
        deployer.start()
    elif args.command == 'stop':
        deployer = TritonModelDeployer(
            model="dummy",
            container_name=args.container_name
        )
        deployer.stop()
    elif args.command == 'status':
        deployer = TritonModelDeployer(
            model="dummy",
            container_name=args.container_name
        )
        deployer.status()
    elif args.command == 'logs':
        deployer = TritonModelDeployer(
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

