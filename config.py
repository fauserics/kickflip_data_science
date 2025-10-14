# config.py
import os

class MissingConfig(RuntimeError):
    pass

def get_cfg(name: str, default=None, required: bool = False):
    val = os.getenv(name, default)
    if required and not val:
        raise MissingConfig(f"Missing required config: {name}")
    return val
