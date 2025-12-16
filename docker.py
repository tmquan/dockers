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
import argparse
import re
from BaseModelDeployer import (
    BaseModelDeployer,
    DEFAULT_PORT,
    DEFAULT_GRPC_PORT,
    DEFAULT_METRICS_PORT,
    DEFAULT_OPENAI_PORT,
    DEFAULT_GPU_MEMORY,
    DEFAULT_METHOD,
    DEFAULT_ENGINE,
    DEFAULT_MAX_MODEL_LEN,
    SUPPORTED_METHODS,
    SUPPORTED_ENGINES,
)
from HFModelDeployer import HFModelDeployer
from NIMModelDeployer import NIMModelDeployer
from UNIMModelDeployer import UNIMModelDeployer
from TritonModelDeployer import TritonModelDeployer

# ============================================================================
# Factory Function
# ============================================================================
def create_deployer(method, model, engine, **kwargs):
    """Factory function to create the appropriate deployer."""
    method = method.lower()
    
    if method == "hf":
        return HFModelDeployer(model, engine, **kwargs)
    elif method == "nim":
        return NIMModelDeployer(model, engine, **kwargs)
    elif method == "unim":
        return UNIMModelDeployer(model, engine, **kwargs)
    elif method == "triton":
        return TritonModelDeployer(model, engine, **kwargs)
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
  - hf: Direct HuggingFace deployment (engines: vllm, trtllm, sglang)
  - nim: NVIDIA NIM (to be implemented) (engines: vllm)
  - unim: Universal NIM deployment (engines: vllm, trtllm, sglang, python/safetensors)
  - triton: Triton Inference Server (engines: vllm, trtllm)
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

