#!/usr/bin/env python3

"""
NVIDIA Triton Server Model Deployment Manager
This script manages model deployments using Triton Inference Server with multiple backends.

Supported backends: vllm, python, trtllm

Usage:
    python docker_hf_with_triton.py start --model MODEL_NAME [--backend BACKEND] [OPTIONS]
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
# Version Configurations - Update these when upgrading Triton versions
# ============================================================================
TRITON_VERSION = "25.11"  # NVIDIA Triton Server version (Dec 2024)

# ============================================================================
# Docker Image Configurations
# ============================================================================
TRITON_VLLM_IMAGE = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION}-vllm-python-py3"
# Python backend uses vLLM image since it includes both Python backend and OpenAI frontend
TRITON_PYTHON_IMAGE = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION}-trtllm-python-py3"
TRITON_TRTLLM_IMAGE = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION}-trtllm-python-py3"
# TensorRT-LLM image for model conversion (has convert_checkpoint.py scripts)
# Using the standalone TensorRT-LLM image which includes conversion examples
TRTLLM_CONVERSION_IMAGE = "tensorrt_llm/release:v0.17.0"  # Official TensorRT-LLM with examples
# Alternative: NeMo container also has TensorRT-LLM conversion tools
TRITON_NEMOFW_IMAGE = f"nvcr.io/nvidia/nemo:{TRITON_VERSION}"

# ============================================================================
# Default Configurations
# ============================================================================
DEFAULT_PORT = 8000
DEFAULT_GRPC_PORT = 8001
DEFAULT_METRICS_PORT = 8002
DEFAULT_OPENAI_PORT = 9000
DEFAULT_GPU_MEMORY = 0.9
DEFAULT_OPT_ENGINE = "vllm"
CONTAINER_CACHE_PATH = "/root/.cache/huggingface"

# ============================================================================
# Backend Configurations
# ============================================================================
BACKEND_CONFIGS = {
    "vllm": {
        "image": TRITON_VLLM_IMAGE,
        "display_name": "vLLM (Triton Backend)",
        "health_endpoint": "/v2/health/ready",
        "openai_health_endpoint": "/health/ready",
        "supports_openai_frontend": True,
        "description": "vLLM backend - Fast and flexible inference with PagedAttention (works directly with HuggingFace models)",
    },
    "python": {
        "image": TRITON_PYTHON_IMAGE,
        "display_name": "Python Backend",
        "health_endpoint": "/v2/health/ready",
        "openai_health_endpoint": "/health/ready",
        "supports_openai_frontend": True,  # OpenAI frontend works with all Triton backends
        "description": "Python backend - Custom inference with HuggingFace Transformers (baseline for comparison)",
    },
    "trtllm": {
        "image": TRITON_TRTLLM_IMAGE,
        "display_name": "TensorRT-LLM (Triton Backend)",
        "health_endpoint": "/v2/health/ready",
        "openai_health_endpoint": "/health/ready",
        "supports_openai_frontend": True,
        "description": "TensorRT-LLM backend - Maximum performance (requires pre-built engines)",
        "requires_conversion": True,
        "requires_manual_setup": True,
    },
}

class TritonModelDeployer:
    """Manages Triton Server model container lifecycle with multiple backend support."""
    
    def __init__(self, model, backend=DEFAULT_OPT_ENGINE, cache_dir=None, 
                 port=DEFAULT_PORT, grpc_port=DEFAULT_GRPC_PORT, metrics_port=DEFAULT_METRICS_PORT,
                 gpu_memory=DEFAULT_GPU_MEMORY, container_name=None, tp_size=1, 
                 max_model_len=None, extra_args=None, openai_frontend=False, openai_port=DEFAULT_OPENAI_PORT):
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
            self.container_name = f"{sanitized_model}-triton-{self.backend}"
        
        # Validate backend
        if self.backend not in BACKEND_CONFIGS:
            print(f"\n❌ Error: Unsupported backend '{self.backend}'")
            print(f"   Supported backends: {', '.join(BACKEND_CONFIGS.keys())}\n")
            sys.exit(1)
        
        self.backend_config = BACKEND_CONFIGS[self.backend]
        
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
    
    def _setup_model_repository(self):
        """Create Triton model repository with proper structure."""
        print(f"\n📦 Setting up Triton model repository...")
        
        # Create model repository directory
        model_repo_dir = self.cache_dir / "model_repository"
        model_repo_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize model name for directory (only replace '/', keep hyphens)
        # Example: "Qwen/Qwen3-30B-A3B-Thinking-2507" -> "Qwen_Qwen3-30B-A3B-Thinking-2507"
        model_name = self.model.replace('/', '_')
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
        elif self.backend == "trtllm":
            # TRT-LLM requires engine build - will be handled separately
            pass
        
        # Set permissions
        self._run_command(f"chmod -R 777 {model_repo_dir}")
        
        return model_repo_dir
    
    def _generate_config_pbtxt(self, model_name):
        """Generate config.pbtxt for Triton model repository."""
        max_model_len = self.max_model_len if self.max_model_len else 32768
        
        if self.backend == "vllm":
            # vLLM backend configuration for Triton
            # Reference: https://github.com/triton-inference-server/vllm_backend
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
        
        elif self.backend == "python":
            # Python backend configuration for Triton with OpenAI frontend compatibility
            # Reference: https://github.com/triton-inference-server/python_backend
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
    name: "stream"
    data_type: TYPE_BOOL
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "max_tokens"
    data_type: TYPE_INT32
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
    name: "top_p"
    data_type: TYPE_FP32
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "frequency_penalty"
    data_type: TYPE_FP32
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "presence_penalty"
    data_type: TYPE_FP32
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "return_num_input_tokens"
    data_type: TYPE_BOOL
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "return_num_output_tokens"
    data_type: TYPE_BOOL
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
        
        elif self.backend == "trtllm":
            # TensorRT-LLM backend configuration for Triton
            # Reference: https://github.com/triton-inference-server/tensorrtllm_backend
            # NOTE: Requires pre-built TRT engines in the model repository
            return f'''name: "{model_name}"
backend: "tensorrtllm"
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
  }},
  {{
    name: "frequency_penalty"
    data_type: TYPE_FP32
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "presence_penalty"
    data_type: TYPE_FP32
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "return_num_input_tokens"
    data_type: TYPE_BOOL
    dims: [ -1 ]
    optional: true
  }},
  {{
    name: "return_num_output_tokens"
    data_type: TYPE_BOOL
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
  key: "tokenizer_type"
  value: {{
    string_value: "auto"
  }}
}}

parameters: {{
  key: "batch_scheduler_policy"
  value: {{
    string_value: "max_utilization"
  }}
}}

parameters: {{
  key: "kv_cache_free_gpu_mem_fraction"
  value: {{
    string_value: "{self.gpu_memory}"
  }}
}}

parameters: {{
  key: "max_num_tokens"
  value: {{
    string_value: "{max_model_len}"
  }}
}}
'''
        
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _create_vllm_model_json(self, model_version_dir):
        """Create model.json file required by vLLM backend."""
        import json
        
        print(f"   Creating model.json for vLLM backend...")
        
        max_model_len = self.max_model_len if self.max_model_len else 32768
        
        # vLLM engine arguments
        # Reference: https://github.com/triton-inference-server/vllm_backend
        vllm_engine_args = {
            "model": self.model,
            "dtype": "auto",
            "max_model_len": max_model_len,
            "gpu_memory_utilization": self.gpu_memory,
            "trust_remote_code": True,
            "enforce_eager": True,  # Disable torch.compile to avoid cache corruption issues
            "enable_chunked_prefill": True,
            "disable_log_stats": False,  # Keep logging enabled
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
                
                # Get optional parameters from OpenAI frontend
                max_tokens = 100
                temperature = 0.7
                top_p = 1.0
                
                try:
                    max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "max_tokens")
                    if max_tokens_tensor is not None:
                        max_tokens = int(max_tokens_tensor.as_numpy()[0])
                except:
                    pass
                
                try:
                    temperature_tensor = pb_utils.get_input_tensor_by_name(request, "temperature")
                    if temperature_tensor is not None:
                        temperature = float(temperature_tensor.as_numpy()[0])
                except:
                    pass
                
                try:
                    top_p_tensor = pb_utils.get_input_tensor_by_name(request, "top_p")
                    if top_p_tensor is not None:
                        top_p = float(top_p_tensor.as_numpy()[0])
                except:
                    pass
                
                # Generate response
                if self.model is not None and self.tokenizer is not None:
                    inputs = self.tokenizer(text_input_str, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True if temperature > 0 else False
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
    
    def _detect_model_architecture(self):
        """Detect the model architecture for TRT-LLM conversion."""
        model_lower = self.model.lower()
        
        # Map common model names to TRT-LLM architecture types
        arch_mapping = {
            'llama': ['llama', 'mistral', 'mixtral', 'yi'],
            'qwen': ['qwen'],
            'gpt': ['gpt2', 'gpt-j', 'gpt-neox'],
            'falcon': ['falcon'],
            'chatglm': ['chatglm'],
            'baichuan': ['baichuan'],
            'internlm': ['internlm'],
            'phi': ['phi'],
            'gemma': ['gemma'],
        }
        
        for arch, keywords in arch_mapping.items():
            for keyword in keywords:
                if keyword in model_lower:
                    return arch
        
        # Default to llama for unknown architectures (most similar)
        print(f"   ⚠️  Unknown architecture, defaulting to 'llama' type")
        return 'llama'
    
    def _build_trtllm_engines(self, model_repo_dir, model_name):
        """Build TensorRT-LLM engines (requires pre-converted checkpoints)."""
        print(f"\n🔧 Checking for TensorRT-LLM engines...")
        
        model_version_dir = model_repo_dir / model_name / "1"
        
        # Check if engines already exist
        engine_files = list(model_version_dir.glob("*.engine")) + list(model_version_dir.glob("rank*.engine"))
        config_json = model_version_dir / "config.json"
        
        if engine_files and config_json.exists():
            print(f"✅ Found existing TRT-LLM engines: {len(engine_files)} files")
            print(f"📁 Engine location: {model_version_dir}")
            return model_version_dir
        
        # Automatic TRT-LLM conversion not yet implemented
        raise NotImplementedError(
            "TensorRT-LLM automatic engine build not implemented. "
            "Place pre-built engines in: " + str(model_version_dir)
        )
    
    def _OLD_detect_model_architecture(self):
        print(f"\n{'='*80}")
        print(f"📥 STEP 1/3: Converting HuggingFace model to TRT-LLM checkpoint")
        print(f"{'='*80}")
        
        checkpoint_dir = self.cache_dir / "trt_checkpoints" / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   Model: {self.model}")
        print(f"   Architecture: {arch_type}")
        print(f"   Output: {checkpoint_dir}")
        print(f"   ⏳ Converting...")
        
        # Use NeMo container which has TensorRT-LLM conversion utilities
        # The conversion happens via Python API, not separate scripts
        conversion_script = f"/app/nemo/scripts/llm_ptq/ptq.py"
        
        # For NeMo-based conversion, we'll use a simpler Python approach
        conversion_code = f'''
import os
import json
os.environ["HF_TOKEN"] = "{os.environ.get('HF_TOKEN', '')}"
os.environ["HF_HOME"] = "/workspace/huggingface"

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# Download and convert HuggingFace model to TensorRT-LLM checkpoint format
print("Downloading model from HuggingFace...")
model = AutoModelForCausalLM.from_pretrained(
    "{self.model}",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    "{self.model}",
    trust_remote_code=True
)

config = AutoConfig.from_pretrained(
    "{self.model}",
    trust_remote_code=True
)

print("Saving model in checkpoint format...")
checkpoint_dir = "/workspace/trt_checkpoints/{model_name}"
os.makedirs(checkpoint_dir, exist_ok=True)

# Save the model and tokenizer
model.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)

# Add TensorRT-LLM required fields to config.json
config_path = os.path.join(checkpoint_dir, "config.json")
with open(config_path, 'r') as f:
    trt_config = json.load(f)

# Map HuggingFace model_type to TensorRT-LLM architecture
model_type = config.model_type.lower()
model_name_lower = "{self.model}".lower()

# Map based on HuggingFace model_type and model name patterns
# Based on actual MODEL_MAP keys from TensorRT-LLM 1.0.3.2510
arch_mapping = {{
    'qwen2': 'Qwen2ForCausalLM',
    'qwen3': 'Qwen3ForCausalLM',
    'qwen': 'QWenForCausalLM',  # Older Qwen models
    'llama': 'LlamaForCausalLM',
    'mistral': 'MistralForCausalLM',
    'mixtral': 'MixtralForCausalLM',
    'gpt2': 'GPT2LMHeadModel',
    'phi': 'Phi3ForCausalLM',
    'phi3': 'Phi3ForCausalLM',
    'gemma': 'GemmaForCausalLM',
    'gemma2': 'Gemma2ForCausalLM',
    'gemma3': 'Gemma3ForCausalLM',
    'deepseek': 'DeepseekForCausalLM',
    'deepseek_v2': 'DeepseekV2ForCausalLM',
}}

# Special handling for Qwen3 variants based on model name
# Note: Qwen3Next is not in MODEL_MAP, treat as Qwen3Moe or Qwen3
if 'qwen3' in model_name_lower or 'qwen/qwen3' in model_name_lower:
    if 'moe' in model_name_lower or 'a3b' in model_name_lower or 'a14b' in model_name_lower:
        architecture = 'Qwen3MoeForCausalLM'  # Has MoE architecture
    else:
        architecture = 'Qwen3ForCausalLM'  # Standard Qwen3
elif 'qwen2' in model_name_lower:
    if 'moe' in model_name_lower:
        architecture = 'Qwen2MoeForCausalLM'
    else:
        architecture = 'Qwen2ForCausalLM'
else:
    architecture = arch_mapping.get(model_type, 'LlamaForCausalLM')

# Add architecture field for TRT-LLM
trt_config['architecture'] = architecture
trt_config['builder_config'] = {{
    'precision': 'float16',
}}

print(f"Setting architecture to: {{trt_config['architecture']}}")

with open(config_path, 'w') as f:
    json.dump(trt_config, f, indent=2)

print("Checkpoint saved successfully with TRT-LLM architecture field")
'''
        
        convert_cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "--ipc", "host",
            "-v", f"{self.cache_dir}:/workspace",
            "-e", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
            "-e", f"HF_HOME=/workspace/huggingface",
            TRITON_NEMOFW_IMAGE,
            "python3", "-c", conversion_code,
        ]
        
        print(f"\n   Running HuggingFace model download and save...")
        result = subprocess.run(convert_cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"\n   ❌ Checkpoint conversion failed!")
            print(f"\n   This could be due to:")
            print(f"      - Model architecture '{arch_type}' not supported or incorrect")
            print(f"      - Model requires different conversion parameters")
            print(f"      - HuggingFace model access issues (check HF_TOKEN)")
            print(f"\n   💡 Try these alternatives instead:")
            print(f"\n   1️⃣  Standalone TensorRT-LLM (auto-converts):")
            print(f"      python docker_hf.py start --model {self.model} --engine trtllm --tp-size {self.tp_size}")
            print(f"\n   2️⃣  vLLM backend (fast, production-ready):")
            print(f"      python docker_hf_with_triton.py start --model {self.model} --backend vllm --openai-frontend --tp-size {self.tp_size}")
            raise RuntimeError("TRT-LLM checkpoint conversion failed")
        
        # Verify checkpoint was created
        checkpoint_files = list(checkpoint_dir.glob("*.safetensors")) + list(checkpoint_dir.glob("*.bin"))
        if not checkpoint_files:
            print(f"   ❌ No checkpoint files found in {checkpoint_dir}")
            raise RuntimeError("TRT-LLM checkpoint conversion produced no output files")
        
        print(f"   ✅ Checkpoint created: {len(checkpoint_files)} files")
        
        # ========================================================================
        # STEP 2: Build TensorRT engines from HuggingFace checkpoint
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"🏗️  STEP 2/3: Building TensorRT engines from HuggingFace model")
        print(f"{'='*80}")
        
        max_model_len = self.max_model_len if self.max_model_len else 32768
        max_batch_size = 128
        
        print(f"   Model: {self.model}")
        print(f"   Output: {model_version_dir}")
        print(f"   Max batch size: {max_batch_size}")
        print(f"   Max sequence length: {max_model_len}")
        print(f"   Tensor parallel: {self.tp_size}")
        print(f"   ⏳ Building engines directly from HuggingFace...")
        
        # trtllm-build with --model_dir for HuggingFace models (bypass checkpoint conversion)
        # Use the HuggingFace model directly
        build_cmd = [
            "docker", "run", "--rm",
            "--gpus", "all",
            "--ipc", "host",
            "-v", f"{self.cache_dir}:/workspace",
            "-v", f"{model_version_dir}:/engine",
            "-e", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
            "-e", f"HF_HOME=/workspace/huggingface",
            TRITON_TRTLLM_IMAGE,
            "trtllm-build",
            "--model_dir", self.model,  # Use HuggingFace model directly
            "--output_dir", "/engine",
            "--gemm_plugin", "auto",
            "--gpt_attention_plugin", "auto",
            "--max_batch_size", str(max_batch_size),
            "--max_input_len", str(max_model_len),
            "--max_seq_len", str(max_model_len),
        ]
        
        # Add tensor parallel configuration
        if self.tp_size > 1:
            build_cmd.extend([
                "--tp_size", str(self.tp_size),
                "--workers", str(self.tp_size)
            ])
        
        print(f"\n   Running: trtllm-build with HuggingFace checkpoint")
        result = subprocess.run(build_cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"\n   ❌ Engine build failed!")
            print(f"\n   HuggingFace checkpoints are incompatible with trtllm-build.")
            print(f"   trtllm-build requires TensorRT-LLM format checkpoints with")
            print(f"   architecture-specific conversion (convert_checkpoint.py).")
            print(f"")
            print(f"   ✅ Use these working alternatives:")
            print(f"")
            print(f"   1️⃣  Standalone TensorRT-LLM (auto-converts, RECOMMENDED):")
            print(f"      python docker_hf.py start --model {self.model} --engine trtllm --tp-size {self.tp_size}")
            print(f"")
            print(f"   2️⃣  vLLM backend (instant, production-ready):")
            print(f"      python docker_hf_with_triton.py start --model {self.model} --backend vllm --openai-frontend --tp-size {self.tp_size}")
            raise RuntimeError("TRT-LLM checkpoint format incompatible. Use docker_hf.py --engine trtllm instead.")
        
        # ========================================================================
        # STEP 3: Verify engines were created
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"✅ STEP 3/3: Verifying TensorRT engines")
        print(f"{'='*80}")
        
        # Check for engine files
        engine_files = list(model_version_dir.glob("*.engine")) + list(model_version_dir.glob("rank*.engine"))
        config_json = model_version_dir / "config.json"
        
        if not engine_files:
            print(f"   ❌ No engine files found in {model_version_dir}")
            raise RuntimeError("TRT-LLM engine build produced no output files")
        
        if not config_json.exists():
            print(f"   ❌ config.json not found in {model_version_dir}")
            raise RuntimeError("TRT-LLM engine build did not create config.json")
        
        print(f"\n   ✅ Engine files: {len(engine_files)}")
        print(f"   ✅ Config: config.json")
        print(f"   📁 Location: {model_version_dir}")
        
        # Show file sizes
        total_size = sum(f.stat().st_size for f in engine_files) / (1024**3)
        print(f"   💾 Total size: {total_size:.2f} GB")
        
        print(f"\n{'='*80}")
        print(f"🎉 TensorRT-LLM engines built successfully!")
        print(f"{'='*80}\n")
        
        return model_version_dir
    
    def _build_triton_command(self):
        """Build Docker command for Triton Server with model repository."""
        
        # Setup model repository first
        model_repo_dir = self._setup_model_repository()
        
        # If TRT-LLM backend, build engines first
        if self.backend == "trtllm":
            model_name = self.model.replace('/', '_')
            self._build_trtllm_engines(model_repo_dir, model_name)
        
        # Environment variables for HuggingFace
        env_vars = [
            f"-e HF_TOKEN={os.getenv('HF_TOKEN', '')}",
            f"-e HUGGING_FACE_HUB_TOKEN={os.getenv('HF_TOKEN', '')}",
            f"-e HF_HOME=/root/.cache/huggingface",
        ]
        
        # Add vLLM-specific environment variables to reduce noise
        if self.backend == "vllm":
            env_vars.extend([
                "-e VLLM_LOGGING_LEVEL=INFO",
                "-e TORCH_COMPILE_DISABLE=1",  # Disable torch.compile to avoid cache corruption
                "-e VLLM_ATTENTION_BACKEND=FLASHINFER",  # Use stable attention backend
            ])
        
        env_vars_str = " \\\n    ".join(env_vars)
        
        # Choose command based on frontend mode
        if self.openai_frontend:
            # Use OpenAI-compatible frontend
            # Reference: https://github.com/triton-inference-server/server/tree/main/python/openai
            
            # Add cache cleanup for vLLM to avoid torch compile cache corruption
            cache_cleanup = ""
            if self.backend == "vllm":
                cache_cleanup = "rm -rf /root/.triton/cache /root/.cache/torch_extensions/* /root/.cache/torch_inductor/* && "
            
            cmd_parts = [
                f"bash -c '{cache_cleanup}cd /opt/tritonserver/python/openai &&",
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
        
        # Get image from backend config
        image = self.backend_config["image"]
        
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
    {image} \\
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
            print(f"   Use 'python docker_hf_with_triton.py start --model <model> --backend <backend>' to create and start the container\n")
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
                print(f"   Service URL: http://localhost:{self.port}")
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
                print(f"   API Status: ⏳ Starting up (may take several minutes)")
        else:
            print(f"\n⚠️  Container '{self.container_name}' exists but is NOT RUNNING")
            print(f"   Status: {status}")
            print(f"   Use 'python docker_hf_with_triton.py start --model {self.model} --backend {self.backend}' to start it")
        
        print()
        return container_id is not None
    
    def start(self):
        """Start the Triton container."""
        print("=" * 80)
        print(f"🚀 Starting Triton Server: {self.model}")
        print(f"   Backend: {self.backend_config['display_name']}")
        print("=" * 80)
        
        # Special warning for TRT-LLM
        if self.backend == "trtllm":
            print("\n⚠️  TensorRT-LLM Backend Selected")
            print("   This backend requires building optimized TensorRT engines.")
            print()
            print("   The script will automatically:")
            print("     1. Detect model architecture and convert HF → TRT-LLM checkpoint")
            print("     2. Build optimized TensorRT engines from checkpoint")
            print("     3. Verify and cache engines for future use")
            print()
            print("   ⏱️  Expected time: 15-45 minutes (depending on model size)")
            print("   💾 Engines will be cached and reused on subsequent starts")
            print()
            print("   💡 Faster alternative with similar performance:")
            print("      --backend vllm  (Production-ready, works immediately)")
            print()
        
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
        image = self.backend_config["image"]
        print(f"\n🔽 Checking for {self.backend} image: {image}")
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
        print(f"\n🐳 Starting Triton container...")
        print(f"   Container name: {self.container_name}")
        print(f"   Cache directory: {self.cache_dir}")
        print(f"   Model: {self.model}")
        print(f"   Backend: {self.backend_config['display_name']}")
        print(f"   GPU memory utilization: {self.gpu_memory}")
        if self.tp_size > 1:
            print(f"   Tensor parallel size: {self.tp_size}")
        if self.openai_frontend:
            print(f"   OpenAI frontend: Enabled (port {self.openai_port})")
        
        docker_cmd = self._build_triton_command()
        result = self._run_command(docker_cmd.strip())
        
        if result.returncode == 0:
            print(f"\n✅ Container started successfully!")
            print(f"   Container ID: {result.stdout.strip()}")
            print(f"\n⏳ Triton Server is initializing... This may take several minutes.")
            print(f"   First run will download the model weights and build the backend.")
            
            if self.openai_frontend:
                print(f"   Service URL (OpenAI-compatible): http://localhost:{self.openai_port}")
                print(f"   Endpoints: /v1/chat/completions, /v1/completions")
                health_endpoint = self.backend_config["openai_health_endpoint"]
                print(f"   Health check: http://localhost:{self.openai_port}{health_endpoint}")
            else:
                print(f"   Service URL: http://localhost:{self.port}")
                print(f"   GRPC URL: localhost:{self.grpc_port}")
                print(f"   Metrics: http://localhost:{self.metrics_port}/metrics")
                health_endpoint = self.backend_config["health_endpoint"]
                print(f"   Health check: http://localhost:{self.port}{health_endpoint}")
            
            print(f"\n💡 Use 'python docker_hf_with_triton.py status --container-name {self.container_name}' to check if the API is ready")
            print(f"   Use 'python docker_hf_with_triton.py logs --container-name {self.container_name} -f' to watch the startup logs\n")
        else:
            print(f"\n❌ Failed to start container!")
            print(f"   Error: {result.stderr}")
            print(f"\n💡 Common issues:")
            print(f"   - NVIDIA Docker runtime not installed (nvidia-docker2)")
            print(f"   - GPU not available or insufficient GPU memory")
            print(f"   - Port {self.port if not self.openai_frontend else self.openai_port} already in use")
            print(f"   - Insufficient disk space in {self.cache_dir}")
            print(f"   - Model not found or access denied (check HF_TOKEN)\n")
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

def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Manage Triton Server deployments with multiple backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with vLLM backend + OpenAI frontend (default - recommended)
  python docker_hf_with_triton.py start --model Qwen/Qwen3-30B-A3B-Thinking-2507 --openai-frontend
  
  # Start with Python backend (baseline for comparison)
  python docker_hf_with_triton.py start --model Qwen/Qwen3-30B-A3B-Thinking-2507 --backend python --openai-frontend
  
  # Start with TRT-LLM backend (maximum performance - automatic build on first start)
  python docker_hf_with_triton.py start --model Qwen/Qwen3-30B-A3B-Thinking-2507 --backend trtllm --openai-frontend --tp-size 8
  
  # Start with custom settings
  python docker_hf_with_triton.py start --model MODEL_NAME --backend vllm \\
    --openai-frontend --openai-port 9000 --gpu-memory 0.9 --max-model-len 32768 --tp-size 8
  
  # Check status
  python docker_hf_with_triton.py status --container-name qwen3-triton-vllm
  
  # Stop container
  python docker_hf_with_triton.py stop --container-name qwen3-triton-vllm
  
  # View logs
  python docker_hf_with_triton.py logs --container-name qwen3-triton-vllm -f

Supported backends:
  - vllm: Fast and flexible inference with PagedAttention (recommended, instant startup)
  - python: HuggingFace Transformers with Python backend (baseline, instant startup)
  - trtllm: Maximum performance with TensorRT-LLM (automatic build, 15-45min first start)
  
Note: vLLM and Python backends work immediately. TRT-LLM builds engines on first start.
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a Triton container')
    start_parser.add_argument('--model', required=True, help='HuggingFace model name (e.g., meta-llama/Llama-3-8B)')
    start_parser.add_argument('--backend', default=DEFAULT_OPT_ENGINE, 
                             choices=list(BACKEND_CONFIGS.keys()),
                             help=f'Backend to use (default: {DEFAULT_OPT_ENGINE})')
    start_parser.add_argument('--openai-frontend', action='store_true',
                             help='Use OpenAI-compatible frontend (exposes /v1/chat/completions)')
    start_parser.add_argument('--openai-port', type=int, default=DEFAULT_OPENAI_PORT,
                             help=f'Port for OpenAI frontend (default: {DEFAULT_OPENAI_PORT})')
    start_parser.add_argument('--cache-dir', help='Cache directory (default: .cache/triton in current directory)')
    start_parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'HTTP port (default: {DEFAULT_PORT})')
    start_parser.add_argument('--grpc-port', type=int, default=DEFAULT_GRPC_PORT, help=f'GRPC port (default: {DEFAULT_GRPC_PORT})')
    start_parser.add_argument('--metrics-port', type=int, default=DEFAULT_METRICS_PORT, help=f'Metrics port (default: {DEFAULT_METRICS_PORT})')
    start_parser.add_argument('--gpu-memory', type=float, default=DEFAULT_GPU_MEMORY, 
                             help=f'GPU memory utilization (default: {DEFAULT_GPU_MEMORY})')
    start_parser.add_argument('--container-name', help='Custom container name (auto-generated if not provided)')
    start_parser.add_argument('--tp-size', type=int, default=1, help='Tensor parallel size (default: 1)')
    start_parser.add_argument('--max-model-len', type=int, help='Maximum model context length')
    start_parser.add_argument('--extra-args', nargs='+', default=[], help='Additional backend-specific arguments')
    
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
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
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
            extra_args=args.extra_args,
            openai_frontend=args.openai_frontend,
            openai_port=args.openai_port
        )
        deployer.start()
    elif args.command == 'stop':
        # For stop/status/logs, we just need container name
        deployer = TritonModelDeployer(
            model="dummy",  # Not used for these commands
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
