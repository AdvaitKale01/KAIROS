"""
KAIROS Memory - Thread-Safe Utilities

Provides GPU lock and common utilities for the standalone KAIROS module.
"""
import threading
import logging

# Thread-safe GPU lock for sentence transformer operations
# Prevents Metal command buffer corruption on Apple Silicon
gpu_lock = threading.RLock()

# Configure logger for KAIROS
logger = logging.getLogger('kairos')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

__all__ = ['gpu_lock', 'logger']
