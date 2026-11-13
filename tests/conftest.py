"""Pytest configuration and environment initialization for Windows."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows finds native DLLs for FAISS and OpenBLAS
faiss_libs = Path(sys.prefix) / "Lib" / "site-packages" / "faiss_cpu.libs"
if faiss_libs.is_dir() and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(str(faiss_libs))
    except Exception:
        pass

# Prevent OpenMP multiple runtime collision on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Safe import of native libraries
try:
    import faiss
    import torch
except Exception:
    pass
