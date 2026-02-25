# MMdedup: A Parallel and Pipelined Multimodal Data Deduplication Framework for MLLM Training

**MMdedup** is a highly efficient, end-to-end framework designed to clean the "digital swamp" of raw, heterogeneous web data for Multimodal Large Language Model (MLLM) training. It integrates a robust automated classification front-end with modality-aware deep deduplication pipelines for text, images, and audio.

## 🌟 Key Contributions

**"Classification-before-Clean" Architecture**: A parallel and pipelined design that transforms unstructured, mixed-modality harvests into high-quality training corpora.

**Adversarial-Robust Sorter**: A lightweight heuristic classifier that achieves **100% accuracy** on files with misleading extensions or JSON-wrapped references, maintaining a high throughput of 4.25K files/s.


* **Modality-Specific Deep Cleaning**:
**Image**: Utilizes **CLIP-ViT-B-16 embeddings** and a divide-and-conquer clustering strategy to detect semantic near-duplicates.
**Audio**: Implements **Spectral Fingerprinting** and MinHash-LSH to remove acoustically similar recordings.
**Text**: Employs a **Multi-Granularity N-gram** method to capture syntactic and lexical variations with 14% faster processing than standard MinHash.

**Significant Efficiency Gains**: Achieves **3.6x higher throughput** in image deduplication compared to state-of-the-art SemDeDup while improving downstream model accuracy.



## 📊 Experimental Results

Our evaluation on nearly **1TB** of data shows that MMdedup effectively balances data reduction and model performance:

Modality,Deduplication Rate,Downstream Test Acc.,Storage Saving
Image (ImageNet-Exp),~67.25%,69.58%,46% - 90%
Audio (ESC-Exp),~77.48%,92.01%,46% - 90%
Text (Amazon-Exp),~90.03%,83.14%,46% - 90%

