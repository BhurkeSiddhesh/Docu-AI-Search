with open("backend/tests/test_benchmarks.py", "r") as f:
    content = f.read()

# Replace backend.benchmarks with scripts.benchmark_models in patch string since it's looking for benchmarks.py
content = content.replace("from backend.benchmarks import get_memory_usage", "from scripts.benchmark_models import get_memory_usage_mb")
content = content.replace("backend.benchmarks", "scripts.benchmark_models")
content = content.replace("get_memory_usage()", "get_memory_usage_mb()")

with open("backend/tests/test_benchmarks.py", "w") as f:
    f.write(content)
