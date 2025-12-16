#!/usr/bin/env python3

"""
Triton Inference Server Deployment - Triton Inference Server deployment with multiple backends.
"""

import os
import re
import json
from pathlib import Path
from BaseModelDeployer import (
    BaseModelDeployer,
    DEFAULT_PORT,
    DEFAULT_GPU_MEMORY,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_OPENAI_PORT,
    DEFAULT_GRPC_PORT,
    DEFAULT_METRICS_PORT,
    CONTAINER_CACHE_PATH
)

# ============================================================================
# Version Configurations
# ============================================================================
TRITON_VERSION = "25.11"

# ============================================================================
# Docker Image Configurations
# ============================================================================
TRITON_VLLM_IMAGE = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION}-vllm-python-py3"
TRITON_TRTLLM_IMAGE = f"nvcr.io/nvidia/tritonserver:{TRITON_VERSION}-trtllm-python-py3"


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

