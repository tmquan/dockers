#!/usr/bin/env python3

"""
HuggingFace Direct Deployment - Direct HuggingFace model deployment with inference engines.
"""

import os
import re
from BaseModelDeployer import (
    BaseModelDeployer,
    DEFAULT_PORT,
    DEFAULT_GPU_MEMORY,
    DEFAULT_MAX_MODEL_LEN,
    CONTAINER_CACHE_PATH
)

# ============================================================================
# Version Configurations
# ============================================================================
VLLM_VERSION = "v0.12.0"
SGLANG_VERSION = "v0.5.6.post2"
TRTLLM_VERSION = "1.2.0rc4"

# ============================================================================
# Docker Image Configurations
# ============================================================================
VLLM_IMAGE = f"vllm/vllm-openai:{VLLM_VERSION}"
SGLANG_IMAGE = f"lmsysorg/sglang:{SGLANG_VERSION}"
TRTLLM_IMAGE = f"nvcr.io/nvidia/tensorrt-llm/release:{TRTLLM_VERSION}"


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

