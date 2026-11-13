"""Pytest configuration and environment initialization for Windows."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows finds native DLLs for FAISS and OpenBLAS
_dll_directory_handles: list[object] = []
faiss_libs = Path(sys.prefix) / "Lib" / "site-packages" / "faiss_cpu.libs"
if faiss_libs.is_dir():
    # Prepend to PATH for standard Windows dynamic loader
    os.environ["PATH"] = str(faiss_libs) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            # Must keep reference to avoid garbage collection closing the handle
            _dll_directory_handles.append(os.add_dll_directory(str(faiss_libs)))
        except Exception:
            pass

# Prevent OpenMP multiple runtime collision on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Explicitly initialize native C extensions before any GUI or threading libraries
import faiss
import torch
