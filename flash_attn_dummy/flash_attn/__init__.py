# Dummy flash_attn module to satisfy import checks
# The actual implementation will use eager attention instead

def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("flash_attn is not available, use attn_implementation='eager'")

def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("flash_attn is not available, use attn_implementation='eager'")

__version__ = "2.0.0"
