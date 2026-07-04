# Model Benchmark Results

*Generated: 2026-07-04 18:27:29*

## System Info
- **RAM**: 31.7 GB total, 7.2 GB available
- **CPU Cores**: 20

## Results

| Model | Size | Load Time | TPS | Fact Score | Memory |
|-------|------|-----------|-----|------------|--------|
| gemma 2b it Q4_K_M | 1426MB | 0.8s | 7.8 | 44% | 1827MB |
| mistral 7b instruct v0 2 Q4_K_M | 4166MB | 1.1s | 3.2 | 76% | 4674MB |
| Mistral 7B Instruct v0 3 Q8_0 | 7346MB | 0.7s | 1.9 | 91% | 7612MB |
| phi 2 Q4_K_M | 1706MB | 0.7s | 5.8 | 66% | 2472MB |
| tinyllama 1 1b chat v1 0 Q4_K_M | 638MB | 0.3s | 12.6 | 67% | 785MB |

## Analysis

- **Fastest**: tinyllama 1 1b chat v1 0 Q4_K_M (12.6 tokens/sec)
- **Most Accurate**: Mistral 7B Instruct v0 3 Q8_0 (91% fact retention)
- **Most Efficient**: tinyllama 1 1b chat v1 0 Q4_K_M (785 MB)