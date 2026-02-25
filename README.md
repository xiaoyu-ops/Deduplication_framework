# MMdedup: A Parallel and Pipelined Multimodal Data Deduplication Framework for MLLM Training

**MMdedup** is a highly efficient, end-to-end framework designed to clean the "digital swamp" of raw, heterogeneous web data for Multimodal Large Language Model (MLLM) training. It integrates a robust automated classification front-end with modality-aware deep deduplication pipelines for text, images, and audio.

## 🌟 Key Contributions

* 
**"Classification-before-Clean" Architecture**: A parallel and pipelined design that transforms unstructured, mixed-modality harvests into high-quality training corpora.


* 
**Adversarial-Robust Sorter**: A lightweight heuristic classifier that achieves **100% accuracy** on files with misleading extensions or JSON-wrapped references, maintaining a high throughput of 4.25K files/s.


* **Modality-Specific Deep Cleaning**:
* 
**Image**: Utilizes **CLIP-ViT-B-16 embeddings** and a divide-and-conquer clustering strategy to detect semantic near-duplicates.


* 
**Audio**: Implements **Spectral Fingerprinting** and MinHash-LSH to remove acoustically similar recordings.


* 
**Text**: Employs a **Multi-Granularity N-gram** method to capture syntactic and lexical variations with 14% faster processing than standard MinHash.




* 
**Significant Efficiency Gains**: Achieves **3.6x higher throughput** in image deduplication compared to state-of-the-art SemDeDup while improving downstream model accuracy.



## 📊 Experimental Results

Our evaluation on nearly **1TB** of data shows that MMdedup effectively balances data reduction and model performance:

| Modality | Deduplication Rate | Downstream Test Acc. | Storage Saving |
| --- | --- | --- | --- |
| **Image** (ImageNet-Exp) | ~67.25% 

 | 69.58% 

 | 46% - 90% 

 |
| **Audio** (ESC-Exp) | ~77.48% 

 | <br>**92.01%** 

 | 46% - 90% 

 |
| **Text** (Amazon-Exp) | ~90.03% 

 | 83.14% 

 | 46% - 90% 

 |

> 
> **Note**: Near-duplicate detection alone contributes over **20 percentage points** of additional deduplication beyond exact binary matching.
> 
> 

## 🛠️ System Architecture

MMdedup operates in two primary phases:

1. 
**Phase 1 (Sorter)**: Uses magic number sniffing, printability analysis, and semantic parsing to route files into modality streams.


2. 
**Phase 2 (Deduplication)**: Each stream undergoes a lightweight SHA-256 exact match followed by an algorithmic near-duplicate removal pass.



## 🚀 Getting Started

### Prerequisites

* Python 3.10+ 


* PyTorch 2.0+ 


* NVIDIA GPU (RTX 3090/4090 recommended for CLIP inference) 



### Installation

```bash
git clone https://github.com/your-username/MMdedup.git
cd MMdedup
pip install -r requirements.txt

```

*Dependencies: `open_clip_torch`, `faiss-gpu`, `librosa`, `datasketch`, `scikit-learn*`.

### Basic Usage

```python
# Run the full pipeline
python main.py --input_dir ./raw_data --output_dir ./clean_data

# Or run specific modules
python sorter.py --input ./raw_data
python image_dedup.py --input ./images --threshold 0.1

```

