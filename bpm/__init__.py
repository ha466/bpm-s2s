import sys

# Prevent broken C++ dynamic operator libraries (like torchvision/torchcodec)
# from crashing transformers / datasets when running on Colab or mixed environments
for _mod in ("torchcodec", "torchvision"):
    try:
        __import__(_mod)
    except Exception:
        sys.modules[_mod] = None
