#!/usr/bin/env python3

"""
Universal NIM Deployment - Universal NIM deployment for HuggingFace Safetensors models.
Supports Python (safetensors), vLLM, SGLang, and TensorRT-LLM backends.
"""

import os
import re
import subprocess
from pathlib import Path
from BaseModelDeployer import (
    BaseModelDeployer,
    DEFAULT_PORT,
    DEFAULT_GPU_MEMORY,
    DEFAULT_MAX_MODEL_LEN,
)

# ============================================================================
# Version Configurations
# ============================================================================
UNIM_VERSION = "1.15.3"

# ============================================================================
# Docker Image Configurations
# ============================================================================
UNIM_IMAGE = f"nvcr.io/nim/nvidia/llm-nim:{UNIM_VERSION}"

# ============================================================================
# NIM Container Paths
# ============================================================================
NIM_CACHE_PATH = "/opt/nim/.cache"
NIM_LOCAL_MODEL_PATH = "/opt/models/local_model"


# ============================================================================
# Universal NIM Deployment
# ============================================================================
class UNIMModelDeployer(BaseModelDeployer):
    """Universal NIM deployment for HuggingFace Safetensors models."""
    
    def __init__(self, model, engine, cache_dir=None, port=DEFAULT_PORT,
                 gpu_memory=DEFAULT_GPU_MEMORY, container_name=None, tp_size=1,
                 max_model_len=None, extra_args=None, local_model_path=None,
                 max_input_length=None, max_output_length=None):
        """
        Initialize UNIM Model Deployer.
        
        Args:
            model: HuggingFace model path (e.g., "mistralai/Codestral-22B-v0.1") 
                   or local path. For HuggingFace, use "hf://" prefix or just the path.
            engine: Backend engine - "vllm", "trtllm", "sglang", or "python" (safetensors)
            cache_dir: Local cache directory for NIM cache
            port: Port to expose the service (default: 8000)
            gpu_memory: GPU memory utilization (not directly used by NIM, kept for compatibility)
            container_name: Custom container name
            tp_size: Tensor parallel size (number of GPUs)
            max_model_len: Maximum model length (deprecated, use max_input_length/max_output_length)
            extra_args: Additional arguments (not used by NIM)
            local_model_path: Path to local model directory (for offline deployment)
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.local_model_path = local_model_path
        self.max_input_length = max_input_length or max_model_len or DEFAULT_MAX_MODEL_LEN
        # max_output_length defaults to max_input_length if not specified
        self.max_output_length = max_output_length if max_output_length is not None else (max_model_len if max_model_len else None)
        
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
        """Generate container name from model and engine."""
        sanitized_model = re.sub(r'[^a-zA-Z0-9_-]', '-', self.model.split('/')[-1].lower())
        return f"{sanitized_model}-unim-{self.engine}"
    
    def _get_health_endpoint(self) -> str:
        """Get health check endpoint."""
        return "/v1/health/ready"
    
    def _get_user_id(self):
        """Get current user ID for container user mapping."""
        try:
            result = subprocess.run(
                ["id", "-u"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def _normalize_model_path(self):
        """
        Normalize model path for NIM.
        - If local_model_path is provided, use it
        - If model starts with 'hf://', use as-is
        - If model is a local path (starts with '/' or './'), use as-is
        - Otherwise, prepend 'hf://' for HuggingFace models
        """
        if self.local_model_path:
            # Local model deployment
            return "/opt/models/local_model"
        
        model_path = self.model
        if model_path.startswith("hf://"):
            return model_path
        elif model_path.startswith("/") or model_path.startswith("./"):
            # Local path
            return model_path
        else:
            # HuggingFace model - prepend hf://
            return f"hf://{model_path}"
    
    def _get_nim_model_profile(self):
        """Get NIM_MODEL_PROFILE based on engine."""
        engine_map = {
            "python": None,  # Default (safetensors)
            "vllm": "vllm",
            "sglang": "sglang",
            "trtllm": "tensorrt_llm",
        }
        return engine_map.get(self.engine)
    
    def _build_docker_command(self) -> str:
        """Build Docker command for NIM deployment."""
        nim_model_name = self._normalize_model_path()
        # NIM_SERVED_MODEL_NAME should be the full model path (without hf:// prefix)
        # This is what the API will expose and what clients should use
        nim_served_model_name = self.model.replace("hf://", "")
        
        # Build environment variables
        env_vars = [
            f"-e NIM_MODEL_NAME={nim_model_name}",
            f"-e NIM_SERVED_MODEL_NAME={nim_served_model_name}",
        ]
        
        # Add HuggingFace token if provided
        hf_token = os.getenv('HF_TOKEN', '')
        if hf_token:
            env_vars.append(f"-e HF_TOKEN={hf_token}")
        
        # Add backend profile if specified
        nim_profile = self._get_nim_model_profile()
        if nim_profile:
            env_vars.append(f"-e NIM_MODEL_PROFILE={nim_profile}")
        
        # Add tensor parallel size if > 1
        if self.tp_size > 1:
            env_vars.append(f"-e NIM_TENSOR_PARALLEL_SIZE={self.tp_size}")
        
        # Add max input/output lengths
        if self.max_input_length:
            env_vars.append(f"-e NIM_MAX_INPUT_LENGTH={self.max_input_length}")
        if self.max_output_length:
            env_vars.append(f"-e NIM_MAX_OUTPUT_LENGTH={self.max_output_length}")
        
        env_vars_str = " \\\n    ".join(env_vars)
        
        # Build volume mounts
        volumes = [
            f'-v "{self.cache_dir}:{NIM_CACHE_PATH}"',
        ]
        
        # Add local model mount if using local model
        if self.local_model_path:
            local_path = Path(self.local_model_path).expanduser().resolve()
            volumes.append(f'-v "{local_path}:{NIM_LOCAL_MODEL_PATH}"')
        
        volumes_str = " \\\n    ".join(volumes)
        
        # Get user ID for container user mapping
        user_id = self._get_user_id()
        user_flag = f"-u {user_id}" if user_id else ""
        
        # Build GPU specification
        if self.tp_size > 1:
            gpu_spec = "--gpus all"
        else:
            gpu_spec = '--gpus "device=0"'
        
        return f"""docker run -d \\
    --name {self.container_name} \\
    --runtime=nvidia \\
    {gpu_spec} \\
    --shm-size=16GB \\
    {env_vars_str} \\
    {volumes_str} \\
    {user_flag} \\
    -p {self.port}:8000 \\
    {UNIM_IMAGE}"""

