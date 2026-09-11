"""
Auto-detect an NVIDIA GPU and, if present, swap in the CUDA-accelerated build
of llama-cpp-python so local model inference (including Agent mode's
multi-step ReAct loop) runs on GPU instead of silently falling back to CPU.

Safe no-op on machines without an NVIDIA GPU/driver: the CPU wheel already
installed from requirements.txt is left untouched. Idempotent: if CUDA
support is already active, does nothing.

Run manually after `pip install -r requirements.txt`:
    python scripts/setup_llama_gpu.py
or: npm run setup:gpu
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
# cu121 is the conservative choice: works with any driver new enough for
# CUDA 12.x (525.60+) and any GPU of compute capability 6.0+.
CUDA_TAG = "cu121"
# The cu121 wheel links against the CUDA 12 runtime but doesn't bundle it,
# and most machines have no system-wide CUDA Toolkit install at all (or, on
# a dev box with one already installed, it may be a non-matching major
# version). These redistributable packages are self-contained DLL-only
# wheels -- no toolkit or system install required. backend/llm_integration.py
# preloads the DLLs from them before importing llama_cpp.
CUDA_RUNTIME_PACKAGES = ["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"]


def _pinned_version() -> str:
    match = re.search(r"^llama-cpp-python==([\w.]+)", REQUIREMENTS.read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find pinned llama-cpp-python version in requirements.txt")
    return match.group(1)


def _has_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _cuda_active() -> bool:
    # Run in a fresh subprocess -- if llama_cpp is already imported in this
    # process (e.g. re-checking after a reinstall), the old shared library
    # stays loaded and a same-process check would report stale results.
    # Import through backend.llm_integration (not raw llama_cpp) so the CUDA
    # runtime DLL preload it does actually runs before llama_cpp loads --
    # otherwise this check can report a false negative even when everything
    # is installed correctly.
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "try:\n"
        "    from backend import llm_integration\n"
        "    import llama_cpp.llama_cpp as lcpp\n"
        "    sys.exit(0 if lcpp.llama_supports_gpu_offload() else 1)\n"
        "except Exception:\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    return result.returncode == 0


def main() -> int:
    if not _has_nvidia_gpu():
        print("[setup_llama_gpu] No NVIDIA GPU detected -- keeping CPU build of llama-cpp-python.")
        return 0

    print("[setup_llama_gpu] NVIDIA GPU detected.")

    if _cuda_active():
        print("[setup_llama_gpu] llama-cpp-python already has CUDA support enabled. Nothing to do.")
        return 0

    version = _pinned_version()
    index_url = f"https://abetlen.github.io/llama-cpp-python/whl/{CUDA_TAG}"
    print(f"[setup_llama_gpu] Installing CUDA build of llama-cpp-python=={version} ({CUDA_TAG})...")

    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        f"llama-cpp-python=={version}",
        "--extra-index-url", index_url,
        "--force-reinstall", "--no-deps",
    ])

    if result.returncode != 0:
        print(
            "[setup_llama_gpu] CUDA wheel install failed -- falling back to CPU build. "
            "The app still works; local LLM calls just run on CPU."
        )
        return 1

    print(f"[setup_llama_gpu] Installing CUDA 12 runtime redistributables ({', '.join(CUDA_RUNTIME_PACKAGES)})...")
    runtime_result = subprocess.run([sys.executable, "-m", "pip", "install", *CUDA_RUNTIME_PACKAGES])
    if runtime_result.returncode != 0:
        print(
            "[setup_llama_gpu] CUDA runtime package install failed -- local models will fall back to CPU "
            "until this is resolved (llama.dll needs cudart/cublas from these packages)."
        )
        return 1

    if _cuda_active():
        print("[setup_llama_gpu] CUDA acceleration enabled successfully.")
        return 0

    print(
        "[setup_llama_gpu] Installed the CUDA wheel but GPU offload still reports unavailable. "
        "Check your NVIDIA driver version (need 525.60+ for CUDA 12.1)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
