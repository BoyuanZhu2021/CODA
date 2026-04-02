"""
Zero-Shot LLM Baseline for Misinformation Detection.

This script evaluates LLMs (GPT-5, Claude, etc.) in a zero-shot setting
WITHOUT the CODA pipeline - just direct classification.

Usage:
    # GPT-4o-mini baseline
    python experiments/llm_baseline.py --backend openai --model gpt-4o-mini --samples 200

    # GPT-5 baseline  
    python experiments/llm_baseline.py --backend openai --model gpt-5 --samples 200

    # Claude Sonnet 4 baseline
    python experiments/llm_baseline.py --backend anthropic --model claude-sonnet-4-20250514 --samples 200

    # Full dataset evaluation
    python experiments/llm_baseline.py --backend openai --model gpt-4o-mini --samples -1
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FAKESV_PATH, FAKETT_PATH, RANDOM_SEED
from utils.llm_client import UnifiedLLMClient
from utils.text_processing import extract_text_content

# Zero-shot classification prompt
ZERO_SHOT_SYSTEM = """You are an expert fact-checker analyzing short video content for misinformation.

Your task: Determine if the video content is FAKE (misinformation) or REAL (authentic).

Analyze the content carefully for:
- Factual accuracy of claims
- Signs of manipulation or sensationalism
- Logical consistency
- Source credibility indicators

Respond ONLY with valid JSON in this exact format:
{
    "verdict": "fake" or "real",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation"
}"""

ZERO_SHOT_USER = """Analyze this video content and determine if it's FAKE or REAL:

=== VIDEO CONTENT ===
{content}
=== END CONTENT ===

Provide your classification as JSON."""


def load_datasets(
    fakesv_path: str = FAKESV_PATH,
    fakett_path: str = FAKETT_PATH,
    source: str = "both"
) -> List[Dict[str, Any]]:
    """Load dataset samples."""
    samples = []
    
    def extract_videos(data: Dict, source_name: str) -> List[Dict]:
        """Extract video list from dataset JSON."""
        videos = []
        
        # Check if data has 'videos' key (nested format)
        if isinstance(data, dict) and 'videos' in data:
            video_list = data['videos']
        elif isinstance(data, list):
            video_list = data
        elif isinstance(data, dict):
            # Old format: {video_id: video_data}
            video_list = []
            for vid, vdata in data.items():
                if isinstance(vdata, dict):
                    vdata['video_id'] = vid
                    video_list.append(vdata)
        else:
            video_list = []
        
        for item in video_list:
            if isinstance(item, dict):
                item['source'] = source_name
                videos.append(item)
        
        return videos
    
    if source in ["both", "sv"]:
        with open(fakesv_path, 'r', encoding='utf-8') as f:
            fakesv_data = json.load(f)
            samples.extend(extract_videos(fakesv_data, 'fakesv'))
    
    if source in ["both", "tt"]:
        with open(fakett_path, 'r', encoding='utf-8') as f:
            fakett_data = json.load(f)
            samples.extend(extract_videos(fakett_data, 'fakett'))
    
    return samples


def evaluate_sample(
    client: UnifiedLLMClient,
    sample: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate a single sample with zero-shot LLM."""
    # Extract text content
    content = extract_text_content(sample)
    
    # Truncate if too long
    if len(content) > 6000:
        content = content[:6000] + "... [truncated]"
    
    # Get LLM prediction
    try:
        result = client.chat_json(
            system_prompt=ZERO_SHOT_SYSTEM,
            user_prompt=ZERO_SHOT_USER.format(content=content),
            max_tokens=500
        )
        
        verdict = result.get('verdict', 'uncertain').lower()
        confidence = result.get('confidence', 0.5)
        reasoning = result.get('reasoning', '')
        
        # Normalize verdict
        if verdict not in ['fake', 'real']:
            verdict = 'fake' if 'fake' in verdict.lower() else 'real'
        
        return {
            'success': True,
            'verdict': verdict,
            'confidence': confidence,
            'reasoning': reasoning
        }
        
    except Exception as e:
        return {
            'success': False,
            'verdict': 'uncertain',
            'confidence': 0.5,
            'reasoning': f'Error: {str(e)}'
        }


def compute_metrics(
    predictions: List[str],
    labels: List[str]
) -> Dict[str, float]:
    """Compute accuracy and F1 scores."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    # Convert to binary: fake=1, real=0
    pred_binary = [1 if p == 'fake' else 0 for p in predictions]
    label_binary = [1 if l == 'fake' else 0 for l in labels]
    
    return {
        'accuracy': accuracy_score(label_binary, pred_binary) * 100,
        'macro_f1': f1_score(label_binary, pred_binary, average='macro') * 100,
        'fake_precision': precision_score(label_binary, pred_binary, pos_label=1) * 100,
        'fake_recall': recall_score(label_binary, pred_binary, pos_label=1) * 100,
        'fake_f1': f1_score(label_binary, pred_binary, pos_label=1) * 100,
        'real_precision': precision_score(label_binary, pred_binary, pos_label=0) * 100,
        'real_recall': recall_score(label_binary, pred_binary, pos_label=0) * 100,
        'real_f1': f1_score(label_binary, pred_binary, pos_label=0) * 100,
    }


def run_baseline(
    backend: str,
    model: str,
    num_samples: int = 200,
    source: str = "both",
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """Run zero-shot baseline evaluation."""
    
    print(f"\n{'='*60}")
    print(f"Zero-Shot LLM Baseline Evaluation")
    print(f"{'='*60}")
    print(f"Backend: {backend}")
    print(f"Model: {model}")
    print(f"Source: {source}")
    print(f"Samples: {num_samples if num_samples > 0 else 'ALL'}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Initialize client
    client = UnifiedLLMClient(backend=backend, model_name=model)
    print(f"✓ LLM client initialized: {client.info}")
    
    # Load data
    samples = load_datasets(source=source)
    print(f"✓ Loaded {len(samples)} samples")
    
    # Random sample if needed
    if num_samples > 0 and num_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, num_samples)
        print(f"✓ Randomly selected {num_samples} samples (seed={seed})")
    
    # Separate by source for per-dataset metrics
    fakett_samples = [s for s in samples if s.get('source') == 'fakett']
    fakesv_samples = [s for s in samples if s.get('source') == 'fakesv']
    
    print(f"  - FakeTT: {len(fakett_samples)} samples")
    print(f"  - FakeSV: {len(fakesv_samples)} samples")
    
    # Run evaluation
    results = {
        'fakett': {'predictions': [], 'labels': []},
        'fakesv': {'predictions': [], 'labels': []},
        'combined': {'predictions': [], 'labels': []}
    }
    
    errors = 0
    
    print(f"\nRunning zero-shot evaluation...")
    for sample in tqdm(samples, desc="Evaluating"):
        # Get ground truth (try multiple field names)
        label = sample.get('ground_truth', 
                    sample.get('annotation', 
                        sample.get('label', 'unknown')))
        if isinstance(label, int):
            label = 'fake' if label == 1 else 'real'
        label = str(label).lower()
        
        if label not in ['fake', 'real']:
            continue
        
        # Get prediction
        result = evaluate_sample(client, sample)
        
        if not result['success']:
            errors += 1
        
        prediction = result['verdict']
        source_key = sample.get('source', 'combined')
        
        # Store results
        results[source_key]['predictions'].append(prediction)
        results[source_key]['labels'].append(label)
        results['combined']['predictions'].append(prediction)
        results['combined']['labels'].append(label)
    
    # Compute metrics
    print(f"\n{'='*60}")
    print(f"Results: {model} (Zero-Shot)")
    print(f"{'='*60}")
    
    metrics = {}
    
    for source_name, data in results.items():
        if len(data['predictions']) == 0:
            continue
            
        m = compute_metrics(data['predictions'], data['labels'])
        metrics[source_name] = m
        
        print(f"\n--- {source_name.upper()} ({len(data['predictions'])} samples) ---")
        print(f"Accuracy:     {m['accuracy']:.2f}%")
        print(f"Macro F1:     {m['macro_f1']:.2f}%")
        print(f"Fake P/R/F1:  {m['fake_precision']:.2f}% / {m['fake_recall']:.2f}% / {m['fake_f1']:.2f}%")
        print(f"Real P/R/F1:  {m['real_precision']:.2f}% / {m['real_recall']:.2f}% / {m['real_f1']:.2f}%")
    
    if errors > 0:
        print(f"\n⚠ {errors} API errors encountered")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'config': {
            'backend': backend,
            'model': model,
            'num_samples': num_samples,
            'source': source,
            'seed': seed,
            'timestamp': timestamp
        },
        'metrics': metrics,
        'errors': errors
    }
    
    output_path = f"logs/baseline_{backend}_{model.replace('/', '_')}_{timestamp}.json"
    os.makedirs('logs', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print table-ready format
    print(f"\n{'='*60}")
    print("TABLE-READY FORMAT (for paper):")
    print(f"{'='*60}")
    
    if 'fakett' in metrics:
        m = metrics['fakett']
        print(f"FakeTT | {model} (zero-shot) | {m['accuracy']:.2f} | {m['macro_f1']:.2f}")
    
    if 'fakesv' in metrics:
        m = metrics['fakesv']
        print(f"FakeSV | {model} (zero-shot) | {m['accuracy']:.2f} | {m['macro_f1']:.2f}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot LLM Baseline Evaluation')
    parser.add_argument('--backend', type=str, default='openai',
                        choices=['openai', 'anthropic'],
                        help='LLM backend (openai or anthropic)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model name (e.g., gpt-4o-mini, gpt-5, claude-sonnet-4-20250514)')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of samples to evaluate (-1 for all)')
    parser.add_argument('--source', type=str, default='both',
                        choices=['both', 'sv', 'tt'],
                        help='Dataset source (both, sv, tt)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for sampling')
    
    args = parser.parse_args()
    
    run_baseline(
        backend=args.backend,
        model=args.model,
        num_samples=args.samples,
        source=args.source,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

