#!/usr/bin/env python3

"""
AlphaFold2 NIM Deployment Manager
This script manages the lifecycle of AlphaFold2 NIM Docker containers.

Usage:
    python docker_nim.py start [--api-key KEY] [--cache-dir PATH] [--port PORT] [--model MODEL]
    python docker_nim.py stop
    python docker_nim.py restart [--api-key KEY] [--model MODEL]
    python docker_nim.py status
    python docker_nim.py logs [-f]

    python docker_hf.py start [--api-key KEY] [--cache-dir PATH] [--port PORT] [--model MODEL]  
    python docker_hf.py stop
    python docker_hf.py restart [--api-key KEY] [--model MODEL]
    python docker_hf.py status
    python docker_hf.py logs [-f]
"""