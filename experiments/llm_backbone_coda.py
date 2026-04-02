"""
LLM Backbone Robustness - Full CODA Pipeline (Section 5.1)

Runs the complete CODA pipeline with different LLM backends.
Supports OpenAI, Anthropic (Claude), and SiliconFlow (DeepSeek, Llama) models.
This tests how robust CODA is when using different LLMs for:
- Claim Extractor Agent
- Verification Agent (web search - OpenAI only)
- Judge Agent
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and modify config BEFORE importing pipeline
import config
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL

# SiliconFlow model mappings (use their exact model IDs)
SILICONFLOW_MODELS = {
    'deepseek': 'deepseek-ai/DeepSeek-V3',
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
    'deepseek-r1': 'deepseek-ai/DeepSeek-R1',
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'qwen': 'Qwen/Qwen2.5-7B-Instruct',
}

def run_coda_with_model(
    model_name: str,
    backend: str = 'openai',
    dataset_name: str = 'fakett',
    sample_limit: Optional[int] = None
):
    """Run full CODA pipeline with specified LLM backend."""
    
    print(f"\n{'='*60}")
    print(f"FULL CODA Pipeline with: {model_name}")
    print(f"Backend: {backend}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Override the config based on backend
    original_backend = config.LLM_BACKEND
    original_model = config.MODEL_NAME
    original_claude = getattr(config, 'CLAUDE_MODEL', None)
    original_api_key = getattr(config, 'OPENAI_API_KEY', None)
    original_base_url = getattr(config, 'OPENAI_BASE_URL', None)
    
    config.LLM_BACKEND = backend
    
    if backend == 'anthropic':
        config.CLAUDE_MODEL = model_name
        print(f"Using Anthropic Claude: {model_name}")
        print("Note: Web search uses OpenAI (Claude doesn't support web search)")
    elif backend == 'siliconflow':
        # Map short names to full model names
        full_model = SILICONFLOW_MODELS.get(model_name.lower(), model_name)
        config.MODEL_NAME = full_model
        config.SILICONFLOW_API_KEY = SILICONFLOW_API_KEY
        config.SILICONFLOW_BASE_URL = SILICONFLOW_BASE_URL
        print(f"Using SiliconFlow: {full_model}")
        print("Note: Web search uses OpenAI (SiliconFlow doesn't support web search)")
    else:
        config.MODEL_NAME = model_name
        if 'gpt-5' in model_name.lower():
            print("Note: Using GPT-5 (temperature=1 required)")
    
    # Now import pipeline (will use updated config)
    from pipeline.cascade_pipeline import CascadePipeline
    from data.loaders import load_dataset
    from classifiers.models import MLPClassifier
    from config import CLASSIFIER_PATH, CONFIDENCE_THRESHOLD
    
    # Load classifier for Stage 1
    classifier = None
    if os.path.exists(CLASSIFIER_PATH):
        classifier = MLPClassifier()
        classifier.load(CLASSIFIER_PATH)
        print(f"Loaded classifier from {CLASSIFIER_PATH}")
    else:
        print("WARNING: No classifier found, skipping Stage 1")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    _, test_samples = dataset.get_train_test_split()
    
    if sample_limit:
        test_samples = test_samples[:sample_limit]
    
    print(f"Test samples: {len(test_samples)}")
    
    # Create and run pipeline
    pipeline = CascadePipeline(
        classifier=classifier,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        use_web_search=True,
        verbose=False,
        enable_logging=True
    )
    
    print(f"\nProcessing with full CODA pipeline (web search ENABLED)...")
    results = pipeline.process_batch(test_samples, show_progress=True, dataset_name=dataset_name)
    
    # Calculate metrics
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / len(results) * 100
    
    # Stage distribution
    stage_dist = {1: 0, 2: 0, 3: 0}
    for r in results:
        stage_dist[r.stage_used] += 1
    
    # Per-class metrics
    y_true = [1 if r.ground_truth == 'fake' else 0 for r in results]
    y_pred = [r.prediction_label for r in results]
    
    tp_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    precision_fake = tp_fake / (tp_fake + fp_fake) * 100 if (tp_fake + fp_fake) > 0 else 0
    recall_fake = tp_fake / (tp_fake + fn_fake) * 100 if (tp_fake + fn_fake) > 0 else 0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0
    
    tp_real = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn_real = fp_fake
    fp_real = fn_fake
    precision_real = tp_real / (tp_real + fp_real) * 100 if (tp_real + fp_real) > 0 else 0
    recall_real = tp_real / (tp_real + fn_real) * 100 if (tp_real + fn_real) > 0 else 0
    f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0
    
    macro_f1 = (f1_fake + f1_real) / 2
    
    # Restore original config
    config.LLM_BACKEND = original_backend
    config.MODEL_NAME = original_model
    if original_claude:
        config.CLAUDE_MODEL = original_claude
    if original_api_key:
        config.OPENAI_API_KEY = original_api_key
    if original_base_url:
        config.OPENAI_BASE_URL = original_base_url
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name} ({backend}) on {dataset_name} (FULL CODA)")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Macro F1: {macro_f1:.2f}%")
    print(f"Fake - P: {precision_fake:.2f}%, R: {recall_fake:.2f}%, F1: {f1_fake:.2f}%")
    print(f"Real - P: {precision_real:.2f}%, R: {recall_real:.2f}%, F1: {f1_real:.2f}%")
    print(f"Stage Distribution: S1={stage_dist[1]} ({stage_dist[1]/len(results)*100:.1f}%), "
          f"S2={stage_dist[2]} ({stage_dist[2]/len(results)*100:.1f}%), "
          f"S3={stage_dist[3]} ({stage_dist[3]/len(results)*100:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/coda_{model_name.replace('.', '_')}_{dataset_name}_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'model': model_name,
        'backend': backend,
        'dataset': dataset_name,
        'pipeline': 'FULL_CODA_WITH_WEBSEARCH',
        'samples': len(results),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision_fake': precision_fake,
        'recall_fake': recall_fake,
        'f1_fake': f1_fake,
        'precision_real': precision_real,
        'recall_real': recall_real,
        'f1_real': f1_real,
        'stage_distribution': stage_dist
    }
    
    os.makedirs('logs', exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {log_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Run Full CODA with different LLM backends")
    parser.add_argument('--backend', type=str, default='openai',
                        choices=['openai', 'anthropic', 'siliconflow'],
                        help='LLM backend to use')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model name (OpenAI: gpt-4o-mini, gpt-5.2; Anthropic: claude-sonnet-4-20250514; SiliconFlow: deepseek, llama)')
    parser.add_argument('--dataset', type=str, default='fakett',
                        choices=['fakett', 'fakesv'],
                        help='Dataset to test')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples')
    
    args = parser.parse_args()
    
    run_coda_with_model(args.model, args.backend, args.dataset, args.limit)


if __name__ == "__main__":
    main()

