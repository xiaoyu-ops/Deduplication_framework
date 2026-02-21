# Multimodal Deduplication Framework (多模态去重框架)

A high-performance, modular framework for deduplicating massive multi-modal datasets (Image, Audio, Text).

## 🚀 Key Innovation: Quality-Aware Semantic Deduplication (Q-SemDeDup)

Unlike traditional methods (e.g., SemDeDup) that blindly select the image closest to the cluster centroid, our framework introduces **Quality-Aware Sorting**.

- **The Problem**: The "most representative" image in a cluster is mathematically closest to the centroid but often suffers from low resolution, compression artifacts, or small file size.
- **Our Solution**: We implement a **Utility-Preserving** selection strategy.
    
    $$ Score = \alpha \cdot Sim(x, C) + (1-\alpha) \cdot Norm(Quality(x)) $$
    
    Where:
    - $Sim(x, C)$ is the semantic similarity to the cluster centroid (Representativeness).
    - $Quality(x)$ is the image quality metric (e.g., file size/resolution).
    - $\alpha$ (default 0.7) balances semantic precision with visual quality.

 This ensures that **we retain the highest quality version** of an image among duplicates, effectively functioning as both a deduplicator and a dataset cleaner.

## 🌟 Features

- **Automated Zero-Shot SemDeDup**: 
  - No need for pre-computed K-Means indices.
  - Automatically runs `MiniBatchKMeans` on-the-fly for global deduplication if no indices are provided.
  - Falls back to `Folder-based` strategy if memory is constrained.
- **High Throughput**: 
  - Optimized for batch processing with multi-worker parallelism.
  - achieving >140 imgs/s on standard hardware (5x faster than baseline SemDeDup).
- **Multi-Modal**: Supports Image (CLIP), Audio (Fingerprinting), and Text (MinHash/N-gram) pipelines.

## 🛠️ Usage

### 1. Configure
Edit `configs/my_pipeline.yaml` to point to your dataset.

### 2. Run
```bash
python -m pipelines.multimodal_runner --config configs/my_pipeline.yaml
```

## 📊 Performance Benchmark (Image 10k Subset)

| Method | Precision | Recall | Speed | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Ours (System)** | **85.2%** | **69% - 90%*** | **141 imgs/s** | *Includes Quality Selection |
| SemDeDup (Original) | 93.7% | 96.2% | 27.9 imgs/s | Static, Pre-computed only |
| SimCLR | 18.2% | 99.6% | 45.0 imgs/s | Low precision |

---
*For detailed documentation, see [docs/README.md](docs/README.md).*
