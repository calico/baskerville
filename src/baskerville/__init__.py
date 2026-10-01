import os

# Long runs fragment the CUDA cache into reserved blocks too small to reuse.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
