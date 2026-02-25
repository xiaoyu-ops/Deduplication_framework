# Multimodal Deduplication Framework

A high-performance, modular framework for deduplicating large multimodal datasets (images, audio, and text).

## Key Innovation — Quality-Aware Semantic Deduplication (Q-SemDeDup)

Traditional semantic deduplication techniques (for example, SemDeDup) typically pick the clip closest to a cluster centroid. In practice this representative item can be low quality (low resolution, compressed, or small file size). Q-SemDeDup augments semantic similarity with a quality-aware score so the retained exemplar is both representative and high quality.

Score formula:

$$
Score = \alpha \cdot Sim(x, C) + (1-\alpha) \cdot Norm(Quality(x))
$$

Where:
- $Sim(x, C)$ is semantic similarity to the cluster centroid (representativeness).
- $Quality(x)$ is a quality proxy (file size, resolution, bitrate, etc.).
- $\alpha$ (default 0.7) controls the trade-off between semantic fidelity and media quality.

This approach ensures the dataset retains the highest-utility item from each duplicate cluster, improving downstream training and evaluation quality.

## Features

- Automated, zero-shot semantic deduplication without requiring precomputed indices.
- Falls back to a lightweight folder-based strategy when memory is limited.
- High-throughput batch processing with multi-worker parallelism (optimized throughput on typical hardware).
- Multimodal support: Image (CLIP embeddings), Audio (fingerprinting / spectral features), and Text (MinHash / n-gram) pipelines.

## Quickstart

1. Configure your pipeline by editing `configs/my_pipeline.yaml` to point at your dataset and desired stages.

2. Run the pipeline:

```bash
python -m pipelines.multimodal_runner --config configs/my_pipeline.yaml
```

## Example: Image 10k Subset Benchmark

| Method | Precision | Recall | Speed | Note |
|---|---:|---:|---:|---|
| Ours (Q-SemDeDup) | 85.2% | 69%–90%* | 141 imgs/s | Includes quality-aware selection |
| SemDeDup (original) | 93.7% | 96.2% | 27.9 imgs/s | Precomputed indices required |
| SimCLR (baseline) | 18.2% | 99.6% | 45.0 imgs/s | Low precision |

*See `docs/README.md` for full evaluation details and dataset setup.

---
For full documentation and examples, see [docs/README.md](docs/README.md).
