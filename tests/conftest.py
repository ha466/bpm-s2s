import os, sys

# Disable conflicting/broken optional dependencies if their C++ extensions fail to load
for _mod in ("torchcodec", "torchvision"):
    try:
        __import__(_mod)
    except Exception:
        sys.modules[_mod] = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
