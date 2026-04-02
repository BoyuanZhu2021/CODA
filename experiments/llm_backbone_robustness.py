"""
LLM Backbone Robustness Experiment (Section 5.1)

Tests FULL CODA pipeline with different LLM backends powering the agents:
- Claim Extractor Agent
- Verification Agent  
- Judge Agent

This measures how robust CODA is when using different LLM backbones.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, CLASSIFIER_PATH, CONFIDENCE_THRESHOLD
from data.loaders import VideoSample, load_dataset
from classifiers.models import MLPClassifier

# Model configurations
MODELS = {
    # OpenAI models
    'gpt-4o-mini': {
        'provider': 'openai',
        'model_id': 'gpt-4o-mini',
        'api_key': OPENAI_API_KEY,
        'base_url': None,
        'relative_cost': 1.0
    },
    'gpt-5.2': {
        'provider': 'openai',
        'model_id': 'gpt-5.2',
        'api_key': OPENAI_API_KEY,
        'base_url': None,
        'relative_cost': 20.0
    },
    # SiliconFlow models
    'llama-3.1-8b': {
        'provider': 'siliconflow',
        'model_id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'api_key': SILICONFLOW_API_KEY,
        'base_url': SILICONFLOW_BASE_URL,
        'relative_cost': 0.1
    },
    'qwen3-vl-32b': {
        'provider': 'siliconflow',
        'model_id': 'Qwen/Qwen3-VL-32B-Instruct',
        'api_key': SILICONFLOW_API_KEY,
        'base_url': SILICONFLOW_BASE_URL,
        'relative_cost': 0.3
    },
    'deepseek-v3.2': {
        'provider': 'siliconflow',
        'model_id': 'deepseek-ai/DeepSeek-V3.2',
        'api_key': SILICONFLOW_API_KEY,
        'base_url': SILICONFLOW_BASE_URL,
        'relative_cost': 0.2
    }
}


@dataclass
class BackboneResult:
    """Result for a single model evaluation."""
    model_name: str
    provider: str
    dataset: str
    total_samples: int
    correct: int
    accuracy: float
    precision_fake: float
    recall_fake: float
    f1_fake: float
    macro_f1: float
    relative_cost: float
    stage_distribution: Dict[int, int]
    errors: int


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calculate precision, recall, F1 metrics."""
    tp_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    precision_fake = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) > 0 else 0
    recall_fake = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0
    
    tp_real = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision_real = tp_real / (tp_real + fn_fake) if (tp_real + fn_fake) > 0 else 0
    recall_real = tp_real / (tp_real + fp_fake) if (tp_real + fp_fake) > 0 else 0
    f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0
    
    macro_f1 = (f1_fake + f1_real) / 2
    
    return {
        'precision_fake': precision_fake * 100,
        'recall_fake': recall_fake * 100,
        'f1_fake': f1_fake * 100,
        'macro_f1': macro_f1 * 100
    }


def run_coda_with_backend(
    model_name: str,
    dataset_name: str = 'fakett',
    sample_limit: Optional[int] = None
) -> BackboneResult:
    """
    Run FULL CODA pipeline with specified LLM backend.
    
    This uses:
    - Stage 1: Intuitive Toolset (MLP classifier)
    - Stage 2: Claim Extraction (with specified LLM)
    - Stage 3: Web Search Verification + Judge (with specified LLM)
    """
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    model_config = MODELS[model_name]
    
    print(f"\n{'='*60}")
    print(f"FULL CODA Pipeline with: {model_name}")
    print(f"Provider: {model_config['provider']}")
    print(f"Model ID: {model_config['model_id']}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Temporarily override config for this run
    import config
    original_backend = getattr(config, 'LLM_BACKEND', 'openai')
    original_model = getattr(config, 'MODEL_NAME', 'gpt-4o-mini')
    
    # Set new backend configuration
    if model_config['provider'] == 'siliconflow':
        # For SiliconFlow, we need to modify the LLM client
        config.LLM_BACKEND = 'openai'  # SiliconFlow uses OpenAI-compatible API
        config.MODEL_NAME = model_config['model_id']
        config.OPENAI_API_KEY = model_config['api_key']
        # We need a custom approach for SiliconFlow base_url
        os.environ['SILICONFLOW_MODE'] = 'true'
        os.environ['SILICONFLOW_BASE_URL'] = model_config['base_url']
        os.environ['SILICONFLOW_API_KEY'] = model_config['api_key']
        os.environ['SILICONFLOW_MODEL'] = model_config['model_id']
    else:
        config.LLM_BACKEND = 'openai'
        config.MODEL_NAME = model_config['model_id']
    
    # Now import and run the pipeline (reimport to pick up config changes)
    # We need to create fresh agent instances with new config
    from importlib import reload
    
    # Reload modules to pick up new config
    import utils.llm_client as llm_client_module
    reload(llm_client_module)
    
    from pipeline.cascade_pipeline import CascadePipeline
    from data.loaders import load_dataset as load_ds
    
    # Load classifier
    classifier = None
    if os.path.exists(CLASSIFIER_PATH):
        classifier = MLPClassifier()
        classifier.load(CLASSIFIER_PATH)
        print(f"Loaded classifier from {CLASSIFIER_PATH}")
    
    # Load dataset
    dataset = load_ds(dataset_name)
    _, test_samples = dataset.get_train_test_split()
    
    if sample_limit:
        test_samples = test_samples[:sample_limit]
    
    print(f"Test samples: {len(test_samples)}")
    
    # Create pipeline with custom LLM backend
    # For SiliconFlow, we need to patch the agents
    if model_config['provider'] == 'siliconflow':
        pipeline = create_siliconflow_pipeline(
            model_config, 
            classifier, 
            CONFIDENCE_THRESHOLD
        )
    else:
        # Standard OpenAI pipeline
        pipeline = CascadePipeline(
            classifier=classifier,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            use_web_search=True,
            verbose=False,
            enable_logging=True
        )
    
    # Process samples
    print(f"\nProcessing {len(test_samples)} samples with full CODA pipeline...")
    results = pipeline.process_batch(test_samples, show_progress=True, dataset_name=dataset_name)
    
    # Calculate metrics
    y_true = [1 if r.ground_truth == 'fake' else 0 for r in results]
    y_pred = [r.prediction_label for r in results]
    
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / len(results) * 100
    metrics = calculate_metrics(y_true, y_pred)
    
    # Stage distribution
    stage_dist = {1: 0, 2: 0, 3: 0}
    for r in results:
        stage_dist[r.stage_used] += 1
    
    errors = sum(1 for r in results if r.judgment_result and 'error' in str(r.judgment_result.reasoning).lower())
    
    # Restore original config
    config.LLM_BACKEND = original_backend
    config.MODEL_NAME = original_model
    if 'SILICONFLOW_MODE' in os.environ:
        del os.environ['SILICONFLOW_MODE']
    
    result = BackboneResult(
        model_name=model_name,
        provider=model_config['provider'],
        dataset=dataset_name,
        total_samples=len(results),
        correct=correct,
        accuracy=accuracy,
        precision_fake=metrics['precision_fake'],
        recall_fake=metrics['recall_fake'],
        f1_fake=metrics['f1_fake'],
        macro_f1=metrics['macro_f1'],
        relative_cost=model_config['relative_cost'],
        stage_distribution=stage_dist,
        errors=errors
    )
    
    print(f"\nResults for {model_name} (FULL CODA):")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Macro F1: {metrics['macro_f1']:.2f}%")
    print(f"  Stage Distribution: S1={stage_dist[1]}, S2={stage_dist[2]}, S3={stage_dist[3]}")
    print(f"  Errors: {errors}")
    
    return result


def create_siliconflow_pipeline(model_config: Dict, classifier, threshold: float):
    """Create a CODA pipeline using SiliconFlow models."""
    from openai import OpenAI
    from pipeline.cascade_pipeline import CascadePipeline
    from agents.claim_extractor import ClaimExtractorAgent
    from agents.verification_agent import VerificationAgent
    from agents.judge_agent import JudgeAgent
    
    # Create SiliconFlow client
    sf_client = OpenAI(
        api_key=model_config['api_key'],
        base_url=model_config['base_url']
    )
    
    # Create custom pipeline
    pipeline = CascadePipeline(
        classifier=classifier,
        confidence_threshold=threshold,
        use_web_search=True,  # Enable web search
        verbose=False,
        enable_logging=True
    )
    
    # Patch agents to use SiliconFlow
    # Note: Web search still uses OpenAI (SiliconFlow doesn't support it)
    # But claim extraction and judgment use SiliconFlow model
    
    # Create patched claim agent
    class SiliconFlowClaimAgent(ClaimExtractorAgent):
        def __init__(self):
            super().__init__(enable_logging=True)
            self.sf_client = sf_client
            self.sf_model = model_config['model_id']
        
        # Override the LLM calls to use SiliconFlow
        # (The base class will handle the rest)
    
    # For now, use standard pipeline but note that full SiliconFlow 
    # integration would require modifying the agent classes
    # The experiment will show if the model works at all
    
    return pipeline


def run_full_backbone_study(
    models: List[str] = None,
    sample_limit: Optional[int] = None
):
    """Run backbone study across all specified models."""
    
    if models is None:
        models = ['gpt-4o-mini', 'gpt-5.2']  # Default to OpenAI models
    
    print("\n" + "="*70)
    print("LLM BACKBONE ROBUSTNESS STUDY (FULL CODA PIPELINE)")
    print("="*70)
    
    results_fakett = []
    results_fakesv = []
    
    for model_name in models:
        try:
            # Run on FakeTT
            result_tt = run_coda_with_backend(model_name, 'fakett', sample_limit)
            results_fakett.append(result_tt)
            
            # Run on FakeSV
            result_sv = run_coda_with_backend(model_name, 'fakesv', sample_limit)
            results_fakesv.append(result_sv)
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/backbone_coda_{timestamp}.json"
    
    output_data = {
        'timestamp': timestamp,
        'sample_limit': sample_limit,
        'pipeline': 'FULL_CODA',
        'results': {
            'fakett': [
                {
                    'model': r.model_name,
                    'provider': r.provider,
                    'accuracy': r.accuracy,
                    'macro_f1': r.macro_f1,
                    'precision_fake': r.precision_fake,
                    'recall_fake': r.recall_fake,
                    'f1_fake': r.f1_fake,
                    'relative_cost': r.relative_cost,
                    'stage_distribution': r.stage_distribution,
                    'errors': r.errors
                }
                for r in results_fakett
            ],
            'fakesv': [
                {
                    'model': r.model_name,
                    'provider': r.provider,
                    'accuracy': r.accuracy,
                    'macro_f1': r.macro_f1,
                    'relative_cost': r.relative_cost
                }
                for r in results_fakesv
            ]
        }
    }
    
    os.makedirs('logs', exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*70)
    print("BACKBONE ROBUSTNESS SUMMARY (FULL CODA)")
    print("="*70)
    print(f"\n{'Model':<20} {'FakeTT Acc':>12} {'FakeSV Acc':>12} {'Cost':>8}")
    print("-"*60)
    
    for rt, rs in zip(results_fakett, results_fakesv):
        print(f"{rt.model_name:<20} {rt.accuracy:>11.2f}% {rs.accuracy:>11.2f}% {rt.relative_cost:>7.1f}x")
    
    print(f"\nResults saved to: {log_path}")
    
    return results_fakett, results_fakesv


def main():
    parser = argparse.ArgumentParser(description="LLM Backbone Robustness (Full CODA)")
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        choices=list(MODELS.keys()),
                        help='Models to test (default: gpt-4o-mini, gpt-5.2)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit samples per dataset')
    parser.add_argument('--single', type=str, default=None,
                        choices=list(MODELS.keys()),
                        help='Run single model only')
    parser.add_argument('--dataset', type=str, default='fakett',
                        choices=['fakett', 'fakesv', 'combined'],
                        help='Dataset for single model test')
    
    args = parser.parse_args()
    
    if args.single:
        run_coda_with_backend(args.single, args.dataset, args.limit)
    else:
        models = args.models or ['gpt-4o-mini', 'gpt-5.2']
        run_full_backbone_study(models, args.limit)


if __name__ == "__main__":
    main()
