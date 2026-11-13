"""Convenience entrypoint to launch the EvidenceGuard Streamlit application.

Configures workspace-local configuration and threading controls for Windows 11.

Usage:
    python run_app.py
    python run_app.py --server.port 8501
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure workspace-local configuration is used for Streamlit credentials
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("USERPROFILE", str(PROJECT_ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure project root is on sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows finds native DLLs for FAISS
_dll_directory_handles: list[object] = []
faiss_libs = Path(sys.prefix) / "Lib" / "site-packages" / "faiss_cpu.libs"
if faiss_libs.is_dir():
    os.environ["PATH"] = str(faiss_libs) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            _dll_directory_handles.append(os.add_dll_directory(str(faiss_libs)))
        except Exception:
            pass

from streamlit.web import cli as stcli

if __name__ == "__main__":
    app_path = str(PROJECT_ROOT / "app" / "main.py")
    sys.argv = ["streamlit", "run", app_path] + sys.argv[1:]
    sys.exit(stcli.main())
