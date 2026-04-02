# CODA: Cognitive Orchestration of Detection Agents for Multimodal Misinformation

Official implementation for the paper:

> **CODA: Cognitive Orchestration of Detection Agents for Multimodal Misinformation**
> Boyuan Zhu, Zhen Wang, Haiqiang Fei, Hong Li, Hongsong Zhu
> *ACM Multimedia 2026 (MM '26)*

## Overview

CODA is an adaptive multi-agent framework for detecting misinformation in short videos. Inspired by dual-process theory in cognitive science, CODA coordinates specialized LLM-based agents via a Directed Acyclic Graph (DAG) to adaptively determine analysis depth based on content complexity.

**Key idea:** Simple cases are resolved by a fast lightweight classifier; complex or ambiguous cases trigger deep multi-agent verification with claim extraction, language-aware web search, and evidence synthesis.

```
                         Input Video
                             |
                    [Audio + Frames + Text]
                             |
                             v
    +------------------------------------------------+
    |  STAGE 1: Intuitive Toolset (No API Cost)      |
    |  Sentence-BERT embeddings + MLP classifier     |
    |  High confidence --> Final prediction           |
    |  Low confidence  --> Stage 2                    |
    +------------------------------------------------+
                             | Low Confidence
                             v
    +------------------------------------------------+
    |  STAGE 2: Claim Extraction Agent (LLM)         |
    |  - Detect content language                     |
    |  - Extract verifiable claims                   |
    |  - Identify red flags & manipulation patterns  |
    |  - If claims need verification --> Stage 3     |
    +------------------------------------------------+
                             | Verifiable Claims
                             v
    +------------------------------------------------+
    |  STAGE 3: Verification & Judge Agents (LLM)    |
    |  - Language-aware web search (queries in the   |
    |    content's original language)                 |
    |  - Evidence synthesis with conflict resolution |
    |  - Final verdict with reasoning chain          |
    +------------------------------------------------+
```

## Quick Start

### 1. Installation

```bash
git clone https://github.com/BoyuanZhu2021/CODA.git
cd CODA
pip install -r requirements.txt
```

**System requirement:** [ffmpeg](https://ffmpeg.org/download.html) must be installed and on your PATH (used for audio extraction from video).

### 2. Set up API keys

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key (required):

```
OPENAI_API_KEY=sk-proj-your-key-here
```

### 3. Analyze a video

```bash
python main.py --video path/to/video.mp4
```

With an optional social media caption:

```bash
python main.py --video video.mp4 --description "#ufo #alien caught on camera!"
```

Save result as JSON:

```bash
python main.py --video video.mp4 --output result.json
```

**Example output:**

```
============================================================
  CODA - Multimodal Misinformation Detection
============================================================
  Video: ufo_romania.mp4

[1/4] Extracting audio ...
[2/4] Transcribing audio (Whisper) ...
       Audio: Brothers, what is this thing? This plane was ...
[3/4] Analyzing video frames (GPT-4o vision) ...
       Visual: The video frames depict a dramatic scene inv...
[4/4] Running CODA cascade pipeline ...

============================================================
  VERDICT:    FAKE
  Confidence: 78.5%
  Stage used: 3
  Language:   ro
  Time:       28.3s

  Reasoning:
    - Video contains UFO conspiracy claims with no credible sources
    - Emotional manipulation through dramatic framing and emojis
    - Web search found no corroborating reports from Romanian media
============================================================
```

## Reproducing Paper Experiments

### Dataset Setup

CODA is evaluated on two benchmarks. These datasets are from third-party sources and **not included** in this repository.

| Dataset | Language | Videos | Source |
|---------|----------|--------|--------|
| **FakeSV** | Chinese | 5,495 | [Qi et al., AAAI 2023](https://github.com/ICTMCG/FakeSV) |
| **FakeTT** | Multilingual | 1,992 | [Shang et al., ACM MM 2021](https://github.com/ICTMCG/FakeTT) |

See [`video_transcriptions/README.md`](video_transcriptions/README.md) for detailed instructions on obtaining and preparing the datasets.

### Train the Stage 1 Classifier

```bash
python classifiers/train_classifier.py --dataset combined
```

This trains the MLP classifier on sentence embeddings and saves it to `models/mlp_classifier.pt`.

### Run Full CODA Pipeline

```bash
python experiments/run_experiment.py --mode full
```

### Run Zero-Shot LLM Baselines

```bash
python experiments/llm_zeroshot_baseline.py --model gpt-4o-mini --dataset fakett
python experiments/llm_zeroshot_baseline.py --model deepseek-v3.2 --dataset fakesv
```

### Run Ablation Study

```bash
python experiments/ablation_study.py --dataset fakett --variants no_websearch no_langaware
```

### LLM Backbone Robustness

```bash
python experiments/llm_backbone_coda.py --backend openai --model gpt-4o-mini --dataset fakett
python experiments/llm_backbone_coda.py --backend anthropic --model claude-sonnet-4-20250514 --dataset fakesv
python experiments/llm_backbone_coda.py --backend siliconflow --model deepseek --dataset fakett
```

### Threshold Sensitivity

```bash
python experiments/threshold_sensitivity.py --dataset fakett
```

## Project Structure

```
CODA/
├── main.py                      # End-to-end: video file --> verdict
├── config.py                    # API keys (via env vars), model settings
├── requirements.txt
├── .env.example                 # Template for API keys
│
├── agents/
│   ├── claim_extractor.py       # Stage 2: LLM claim extraction
│   ├── verification_agent.py    # Stage 3: language-aware web search
│   └── judge_agent.py           # Final verdict with evidence synthesis
│
├── pipeline/
│   ├── cascade_pipeline.py      # 3-stage cascade orchestration
│   └── evaluation.py            # Metrics (accuracy, F1, confusion matrix)
│
├── classifiers/
│   ├── models.py                # MLP / LogReg / Transformer classifiers
│   └── train_classifier.py      # Training script with comparison
│
├── data/
│   └── loaders.py               # Dataset loading and preprocessing
│
├── utils/
│   ├── llm_client.py            # Unified LLM client (OpenAI/Anthropic/SiliconFlow)
│   ├── prompts.py               # LLM prompt templates
│   ├── text_processing.py       # Text cleaning, language detection
│   └── logger.py                # Structured logging
│
├── experiments/                  # Scripts to reproduce paper results
│   ├── run_experiment.py         # Main experiment runner
│   ├── llm_zeroshot_baseline.py  # Zero-shot LLM baselines
│   ├── llm_backbone_coda.py      # Backbone robustness experiments
│   ├── ablation_study.py         # Component ablation
│   └── threshold_sensitivity.py  # Threshold sweep
│
├── video_transcriptions/         # Place dataset JSONs here (see README inside)
└── models/                       # Trained weights stored here (see README inside)
```

## Configuration

All API keys are loaded from environment variables (via `.env` file). Edit `config.py` to change model settings:

```python
LLM_BACKEND = "openai"           # "openai", "anthropic", or "siliconflow"
MODEL_NAME = "gpt-4o-mini"       # Model for claim extraction & judgment
CONFIDENCE_THRESHOLD = 0.75      # Stage 1 threshold (higher = more samples to LLM)
MLP_HIDDEN_LAYERS = [256, 128, 64]
```

## Citation

```bibtex
@inproceedings{zhu2026coda,
  title={{CODA}: Cognitive Orchestration of Detection Agents for Multimodal Misinformation},
  author={Zhu, Boyuan and Wang, Zhen and Fei, Haiqiang and Li, Hong and Zhu, Hongsong},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
