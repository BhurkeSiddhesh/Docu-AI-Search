"""
GPU Acceleration Diagnostic for llama-cpp-python

Prints the system info from llama-cpp to verify if BLAS, CUDA, or Metal 
acceleration is correctly initialized and available.
"""
try:
    import llama_cpp
    print(llama_cpp.get_system_info())
except Exception as e:
    print(e)
