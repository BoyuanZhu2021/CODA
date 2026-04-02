"""
Zero-shot LLM Baseline Experiment (Section 5.1)

Tests different LLMs in zero-shot classification mode.
Supports both OpenAI and SiliconFlow (OpenAI-compatible) APIs.

This shows LLM capability WITHOUT CODA's architecture.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL
from data.loaders import load_dataset
from utils.text_processing import extract_text_content

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
    # SiliconFlow models (use SiliconFlow base URL)
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

# Zero-shot prompt
SYSTEM_PROMPT = """You are an expert fact-checker analyzing short video content for misinformation.
Your task is to determine if the video content is FAKE (contains misinformation) or REAL (authentic/factual).

Analyze the content carefully, considering:
1. Factual accuracy of claims made
2. Presence of manipulation tactics (emotional appeals, misleading framing)
3. Source credibility indicators
4. Logical consistency

Respond in JSON format:
{
    "verdict": "fake" or "real",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""

USER_PROMPT = """Analyze this video content and determine if it's fake or real:

{content}

Respond with JSON only."""


def create_client(model_config: Dict) -> OpenAI:
    """Create OpenAI-compatible client."""
    kwargs = {'api_key': model_config['api_key']}
    if model_config['base_url']:
        kwargs['base_url'] = model_config['base_url']
    return OpenAI(**kwargs)


def classify_sample(client: OpenAI, model_id: str, content: str, provider: str) -> Dict[str, Any]:
    """Classify a single sample using zero-shot LLM."""
    try:
        kwargs = {
            'model': model_id,
            'messages': [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(content=content[:3000])}
            ],
            'temperature': 0.1,
            'max_tokens': 500
        }
        
        # Handle GPT-5 constraints
        if 'gpt-5' in model_id.lower():
            kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
            kwargs['temperature'] = 1
        
        response = client.chat.completions.create(**kwargs)
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        result = json.loads(response_text)
        return {
            'verdict': result.get('verdict', 'real').lower(),
            'confidence': result.get('confidence', 0.5),
            'error': None
        }
        
    except json.JSONDecodeError:
        # Try to extract verdict from text
        if 'fake' in response_text.lower():
            return {'verdict': 'fake', 'confidence': 0.6, 'error': None}
        return {'verdict': 'real', 'confidence': 0.5, 'error': 'JSON parse error'}
    except Exception as e:
        return {'verdict': 'real', 'confidence': 0.5, 'error': str(e)}


def run_zeroshot_experiment(
    model_name: str,
    dataset_name: str = 'fakett',
    sample_limit: Optional[int] = None
):
    """Run zero-shot classification experiment."""
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    model_config = MODELS[model_name]
    
    print(f"\n{'='*60}")
    print(f"Zero-shot Baseline: {model_name}")
    print(f"Provider: {model_config['provider']}")
    print(f"Model ID: {model_config['model_id']}")
    print(f"Base URL: {model_config['base_url'] or 'OpenAI default'}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Create client with correct base URL
    client = create_client(model_config)
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    _, test_samples = dataset.get_train_test_split()
    
    if sample_limit:
        test_samples = test_samples[:sample_limit]
    
    print(f"Samples: {len(test_samples)}")
    
    # Process samples
    y_true = []
    y_pred = []
    errors = 0
    
    from tqdm import tqdm
    for sample in tqdm(test_samples, desc=f"Zero-shot {model_name}"):
        content = extract_text_content(sample.raw_data) if hasattr(sample, 'raw_data') else sample.combined_text
        
        result = classify_sample(client, model_config['model_id'], content, model_config['provider'])
        
        ground_truth = 1 if sample.ground_truth == 'fake' else 0
        prediction = 1 if result['verdict'] == 'fake' else 0
        
        y_true.append(ground_truth)
        y_pred.append(prediction)
        
        if result['error']:
            errors += 1
    
    # Calculate metrics
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) * 100
    
    # F1 metrics
    tp_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    precision_fake = tp_fake / (tp_fake + fp_fake) * 100 if (tp_fake + fp_fake) > 0 else 0
    recall_fake = tp_fake / (tp_fake + fn_fake) * 100 if (tp_fake + fn_fake) > 0 else 0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0
    
    tp_real = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision_real = tp_real / (tp_real + fn_fake) * 100 if (tp_real + fn_fake) > 0 else 0
    recall_real = tp_real / (tp_real + fp_fake) * 100 if (tp_real + fp_fake) > 0 else 0
    f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0
    
    macro_f1 = (f1_fake + f1_real) / 2
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name} Zero-shot on {dataset_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Macro F1: {macro_f1:.2f}%")
    print(f"Errors: {errors}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/zeroshot_{model_name.replace('.', '_')}_{dataset_name}_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'model': model_name,
        'provider': model_config['provider'],
        'dataset': dataset_name,
        'mode': 'ZERO_SHOT',
        'samples': len(y_true),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision_fake': precision_fake,
        'recall_fake': recall_fake,
        'f1_fake': f1_fake,
        'errors': errors
    }
    
    os.makedirs('logs', exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {log_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Zero-shot LLM Baseline")
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODELS.keys()),
                        help='Model to test')
    parser.add_argument('--dataset', type=str, default='fakett',
                        choices=['fakett', 'fakesv'],
                        help='Dataset to test')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples')
    
    args = parser.parse_args()
    
    run_zeroshot_experiment(args.model, args.dataset, args.limit)


if __name__ == "__main__":
    main()

