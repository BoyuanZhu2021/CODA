# Model Weights

This directory stores trained model weights and pre-downloaded HuggingFace model files. These are **not included** in the repository due to size.

## Stage 1 Classifier

The MLP classifier (`mlp_classifier.pt`) is trained on sentence embeddings from the FakeSV/FakeTT datasets.

### Train from scratch

```bash
python classifiers/train_classifier.py --dataset combined
```

This will save:
- `models/mlp_classifier.pt` -- MLP classifier (primary, used by CODA)
- `models/logreg_classifier.pkl` -- Logistic regression baseline

### Skip Stage 1

If you don't have training data, CODA still works -- it will skip Stage 1 and route all samples through the LLM agents:

```bash
python main.py --video clip.mp4 --no-stage1
```

## Embedding Models

CODA uses multilingual sentence embeddings for Stage 1. On first run, these are downloaded automatically from HuggingFace:

- **paraphrase-multilingual-MiniLM-L12-v2** -- 384-dim multilingual embeddings
- **xlm-roberta-base** -- for optional transformer fine-tuning

To pre-download manually:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
model.save("models/paraphrase-multilingual-MiniLM-L12-v2")
```

Or use the provided download script:

```bash
python download_configs.py
```
