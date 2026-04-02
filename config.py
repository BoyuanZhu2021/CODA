"""Configuration for Fake Video Detection System"""

import os
from dotenv import load_dotenv
load_dotenv()

# =================================================================
# API CONFIGURATION
# =================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"

# LLM Backend: "openai" or "anthropic"
LLM_BACKEND = "openai"

# Model names by backend
MODEL_NAME = "gpt-4o-mini"  #OpenAI: gpt-4o-mini, gpt-4o, gpt-5, o1-preview
CLAUDE_MODEL = "claude-sonnet-4-20250514"  # Anthropic: claude-sonnet-4-20250514, claude-3-5-sonnet-20241022

# Web Search Configuration
USE_OPENAI_WEB_SEARCH = True
OPENAI_WEB_SEARCH_MODEL = "gpt-4o-mini"

# =================================================================
# MODEL CONFIGURATION
# =================================================================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Multilingual embedding model for Stage 1 classifier (LogReg, MLP)
EMBEDDING_MODEL = os.path.join(_BASE_DIR, "models", "paraphrase-multilingual-MiniLM-L12-v2")

# Transformer model for fine-tuning (more accurate but slower to train)
TRANSFORMER_MODEL = os.path.join(_BASE_DIR, "models", "xlm-roberta-base")

# Alternative: Use Hugging Face model names (requires download)
# EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# TRANSFORMER_MODEL = "xlm-roberta-base"

# =================================================================
# CLASSIFIER CONFIGURATION
# =================================================================
# Path to trained MLP classifier for Stage 1 (Intuitive Toolset)
CLASSIFIER_PATH = os.path.join(_BASE_DIR, "models", "mlp_classifier.pt")

# Confidence threshold for Stage 1 - samples below this go to Stage 2
CONFIDENCE_THRESHOLD = 0.75

# MLP Architecture
MLP_HIDDEN_LAYERS = [256, 128, 64]
MLP_DROPOUT = 0.3
MLP_EPOCHS = 30  # More epochs for better convergence
MLP_LEARNING_RATE = 0.001
MLP_EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs

# Transformer fine-tuning settings
TRANSFORMER_EPOCHS = 7  # More epochs for transformer
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_BATCH_SIZE = 8  # Smaller batch for GPU memory

# General training settings
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
TRAIN_TEST_SPLIT = 0.2
RANDOM_SEED = 42  # Fixed seed for reproducibility

# =================================================================
# DATA PATHS
# =================================================================
FAKESV_PATH = "video_transcriptions/fakesv_1082videos.json"
FAKETT_PATH = "video_transcriptions/fakett_1089videos.json"

# =================================================================
# DEVICE CONFIGURATION
# =================================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

