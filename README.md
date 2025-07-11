# MedSigLIP in FiftyOne

**MedSigLIP** is a large-scale medical vision-language model developed by Google Health. It is designed to encode medical images and associated text into a shared embedding space, enabling advanced applications in healthcare AI.

This repository provides a FiftyOne integration for Google's MedSigLIP embedding models, enabling powerful text-image similarity search capabilities in your FiftyOne datasets.

# ℹ️  Important! Be sure to request access to the model!

This is a gated model, so you will need to fill out the form on the model card: https://huggingface.co/google/medsiglip-448

Approval should be instantaneous.

You'll also have to set your Hugging Face in your enviornment:

```bash
export HF_TOKEN="your_token"
```

Or sign-in to Hugging Face via the CLI:

```bash
huggingface-cli login
```

### About the model

- **Architecture:** Two-tower encoder, each with 400 million parameters: one for images (vision transformer) and one for text (text transformer).
- **Input Support:** 
  - Images: 448x448 resolution
  - Text: Up to 64 tokens
- **Training Data:** Trained on a diverse mix of de-identified medical images and text pairs (e.g., chest X-rays, dermatology, ophthalmology, pathology, CT/MRI slices) plus natural image-text pairs.
- **Primary Use Cases:**
  - Medical image interpretation
  - Data-efficient and zero-shot classification
  - Semantic image retrieval
- **Performance:** Demonstrates strong zero-shot and linear probe performance across multiple medical imaging domains, outperforming or matching specialized models on key benchmarks.
- **Recommended For:** Healthcare AI developers seeking robust, general-purpose medical image and text embeddings, especially for classification and retrieval tasks (not for text generation).

### Example Applications

- Zero-shot classification of medical images
- Semantic search in medical image databases
- Embedding generation for downstream machine learning tasks

## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harpreetsahota204/medsiglip/blob/main/using_medsiglip_model.ipynb)

### Installation


Install the requirements: `pip install fiftyone huggingface-hub accelerate sentencepiece protobuf`

### Download a sample dataset

You can use the [SLAKE dataset](https://huggingface.co/datasets/Voxel51/SLAKE) as a running example. This is how to download it from the Hugging Face hub:

```python
import fiftyone as fo

from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub(
    "Voxel51/SLAKE",
    name="SLAKE",
    overwrite=True,
    max_samples=10
    )
```

Next, you need to register and download the model:

```python
import fiftyone.zoo as foz

# Register this custom model source
foz.register_zoo_model_source("https://github.com/harpreetsahota204/medsiglip")

# Download your preferred SigLIP2 variant
# Note that you will need to acknowledge the license if you haven't yet of MedSiglip on HuggingFace if you haven't yet
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/medsiglip",
    model_name="google/medsiglip-448",
)
```

### Loading the Model

```python
import fiftyone.zoo as foz

model = foz.load_zoo_model(
    "google/medsiglip-448"
)
```

### Computing Image Embeddings

```python
dataset.compute_embeddings(
    model=model,
    embeddings_field="medsiglip_embeddings",
)
```

### Visualizing Embeddings

```python
import fiftyone.brain as fob

results = fob.compute_visualization(
    dataset,
    embeddings="medsiglip_embeddings",
    method="umap",
    brain_key="medsiglip_viz",
    num_dims=2,
)

# View in the App
session = fo.launch_app(dataset)
```

### Text-Image Similarity Search

```python
import fiftyone.brain as fob

# Build a similarity index
text_img_index = fob.compute_similarity(
    dataset,
    model=m"google/medsiglip-448",
    brain_key="medsiglip_similarity",
)

# Search by text query
similar_images = text_img_index.sort_by_similarity("a photo of a chest x-ray")

# View results
session = fo.launch_app(similar_images)
```

## License

This model is released with Health AI Developer Foundations Terms of Use. Refer to the [official license](https://developers.google.com/health-ai-developer-foundations/terms) for details.

## Citation

```bibtex
@article{sellergren2025medgemma,
  title={MedGemma Technical Report},
  author={Sellergren, Andrew and Kazemzadeh, Sahar and Jaroensri, Tiam and Kiraly, Atilla and Traverse, Madeleine and Kohlberger, Timo and Xu, Shawn and Jamil, Fayaz and Hughes, Cían and Lau, Charles and others},
  journal={arXiv preprint arXiv:2507.05201},
  year={2025}
}
```
