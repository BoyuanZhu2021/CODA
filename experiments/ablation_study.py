"""
Ablation Study Experiments for CODA

Tests the contribution of each component by removing one at a time:
1. w/o Intuitive Toolset (LLM-only): Skip Stage 1 classifier
2. w/o Web Search: Disable web verification  
3. w/o Conflict Resolution: Skip conflict analysis in judgment
4. w/o Language-Aware Query: Use English-only search queries
5. w/o Manipulation Detector: Skip red flag detection
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIDENCE_THRESHOLD, CLASSIFIER_PATH
from data.loaders import VideoSample, load_dataset
from classifiers.models import MLPClassifier
from agents.claim_extractor import ClaimExtractorAgent, ClaimExtractionResult
from agents.verification_agent import VerificationAgent, VerificationResult
from agents.judge_agent import JudgeAgent, JudgmentResult


@dataclass
class AblationResult:
    """Result for a single sample in ablation study."""
    video_id: str
    ground_truth: str
    prediction: str
    is_correct: bool
    confidence: float
    stage_used: int
    processing_time: float


@dataclass
class AblationStudyResult:
    """Aggregated results for an ablation variant."""
    variant_name: str
    dataset: str
    total_samples: int
    correct: int
    accuracy: float
    precision_fake: float
    recall_fake: float
    f1_fake: float
    precision_real: float
    recall_real: float
    f1_real: float
    macro_f1: float
    total_time: float
    results: List[AblationResult] = field(default_factory=list)


class AblationPipeline:
    """
    Modified pipeline for ablation experiments.
    Supports disabling individual components.
    """
    
    def __init__(
        self,
        use_intuitive_toolset: bool = True,
        use_web_search: bool = True,
        use_conflict_resolution: bool = True,
        use_language_aware_query: bool = True,
        use_manipulation_detector: bool = True,
        classifier_path: str = CLASSIFIER_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        verbose: bool = False
    ):
        self.use_intuitive_toolset = use_intuitive_toolset
        self.use_web_search = use_web_search
        self.use_conflict_resolution = use_conflict_resolution
        self.use_language_aware_query = use_language_aware_query
        self.use_manipulation_detector = use_manipulation_detector
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
        # Load classifier if using intuitive toolset
        self.classifier = None
        if use_intuitive_toolset and classifier_path and os.path.exists(classifier_path):
            self.classifier = MLPClassifier()
            self.classifier.load(classifier_path)
            if verbose:
                print(f"Loaded classifier from {classifier_path}")
        
        # Initialize agents (lazy)
        self._claim_agent = None
        self._verification_agent = None
        self._judge_agent = None
    
    @property
    def claim_agent(self) -> ClaimExtractorAgent:
        if self._claim_agent is None:
            self._claim_agent = ClaimExtractorAgent(enable_logging=False)
        return self._claim_agent
    
    @property
    def verification_agent(self) -> VerificationAgent:
        if self._verification_agent is None:
            self._verification_agent = VerificationAgent(enable_logging=False)
        return self._verification_agent
    
    @property
    def judge_agent(self) -> JudgeAgent:
        if self._judge_agent is None:
            self._judge_agent = JudgeAgent(enable_logging=False)
        return self._judge_agent
    
    def process_sample(self, sample: VideoSample) -> AblationResult:
        """Process a single sample through the ablation pipeline."""
        start_time = time.time()
        video_id = sample.video_id
        ground_truth = sample.ground_truth
        
        stage_used = 1
        prediction = None
        confidence = 0.5
        
        # Stage 1: Intuitive Toolset
        if self.use_intuitive_toolset and self.classifier is not None:
            proba = self.classifier.predict_proba([sample.combined_text])[0]
            stage1_pred = "fake" if np.argmax(proba) == 1 else "real"
            stage1_conf = float(np.max(proba))
            
            if stage1_conf >= self.confidence_threshold:
                prediction = stage1_pred
                confidence = stage1_conf
                processing_time = time.time() - start_time
                return AblationResult(
                    video_id=video_id,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    is_correct=(prediction == ground_truth),
                    confidence=confidence,
                    stage_used=1,
                    processing_time=processing_time
                )
        
        # Stage 2: Claim Extraction
        stage_used = 2
        claim_result = self._extract_claims_ablation(sample)
        
        # Check if web search needed
        needs_web = claim_result.needs_web_search if self.use_web_search else False
        
        if not needs_web:
            # Make judgment without web search
            judgment = self._make_judgment_ablation(claim_result, None)
            processing_time = time.time() - start_time
            return AblationResult(
                video_id=video_id,
                ground_truth=ground_truth,
                prediction=judgment.verdict,
                is_correct=(judgment.verdict == ground_truth),
                confidence=judgment.confidence,
                stage_used=2,
                processing_time=processing_time
            )
        
        # Stage 3: Web Search Verification
        stage_used = 3
        verification_result = self._verify_claims_ablation(claim_result)
        
        # Final judgment
        judgment = self._make_judgment_ablation(claim_result, verification_result)
        
        processing_time = time.time() - start_time
        return AblationResult(
            video_id=video_id,
            ground_truth=ground_truth,
            prediction=judgment.verdict,
            is_correct=(judgment.verdict == ground_truth),
            confidence=judgment.confidence,
            stage_used=3,
            processing_time=processing_time
        )
    
    def _extract_claims_ablation(self, sample: VideoSample) -> ClaimExtractionResult:
        """Extract claims with ablation options."""
        result = self.claim_agent.extract_claims(sample.raw_data, sample.video_id)
        
        # Ablation: Remove manipulation detector (red flags)
        if not self.use_manipulation_detector:
            result.red_flags = []
        
        return result
    
    def _verify_claims_ablation(self, claim_result: ClaimExtractionResult) -> VerificationResult:
        """Verify claims with ablation options."""
        # Ablation: Use English-only queries instead of language-aware
        if not self.use_language_aware_query:
            # Override detected language to force English queries
            original_lang = claim_result.detected_language
            original_name = claim_result.language_name
            claim_result.detected_language = "en"
            claim_result.language_name = "English"
            
            result = self.verification_agent.verify_claims(claim_result)
            
            # Restore original language
            claim_result.detected_language = original_lang
            claim_result.language_name = original_name
            return result
        
        return self.verification_agent.verify_claims(claim_result)
    
    def _make_judgment_ablation(
        self, 
        claim_result: ClaimExtractionResult,
        verification_result: Optional[VerificationResult]
    ) -> JudgmentResult:
        """Make judgment with ablation options."""
        # Ablation: Skip conflict resolution (use quick judgment)
        if not self.use_conflict_resolution:
            return self.judge_agent.quick_judgment_from_claims(claim_result)
        
        if verification_result:
            return self.judge_agent.make_judgment(claim_result, verification_result)
        else:
            return self.judge_agent.quick_judgment_from_claims(claim_result)
    
    def process_batch(
        self, 
        samples: List[VideoSample],
        show_progress: bool = True
    ) -> List[AblationResult]:
        """Process a batch of samples."""
        results = []
        
        iterator = samples
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(samples, desc="Processing")
        
        for sample in iterator:
            try:
                result = self.process_sample(sample)
                results.append(result)
                
                if show_progress and hasattr(iterator, 'set_postfix'):
                    correct = sum(1 for r in results if r.is_correct)
                    iterator.set_postfix({'acc': f'{correct/len(results):.1%}'})
            except Exception as e:
                print(f"Error processing {sample.video_id}: {e}")
                # Add failed result
                results.append(AblationResult(
                    video_id=sample.video_id,
                    ground_truth=sample.ground_truth,
                    prediction="fake",  # Default
                    is_correct=False,
                    confidence=0.5,
                    stage_used=0,
                    processing_time=0.0
                ))
        
        return results


def calculate_metrics(results: List[AblationResult]) -> Dict[str, float]:
    """Calculate precision, recall, F1 for each class."""
    y_true = [1 if r.ground_truth == "fake" else 0 for r in results]
    y_pred = [1 if r.prediction == "fake" else 0 for r in results]
    
    # Fake class metrics (label=1)
    tp_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn_fake = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    precision_fake = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) > 0 else 0
    recall_fake = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0
    f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0
    
    # Real class metrics (label=0)
    tp_real = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp_real = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fn_real = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    
    precision_real = tp_real / (tp_real + fp_real) if (tp_real + fp_real) > 0 else 0
    recall_real = tp_real / (tp_real + fn_real) if (tp_real + fn_real) > 0 else 0
    f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0
    
    macro_f1 = (f1_fake + f1_real) / 2
    
    return {
        'precision_fake': precision_fake * 100,
        'recall_fake': recall_fake * 100,
        'f1_fake': f1_fake * 100,
        'precision_real': precision_real * 100,
        'recall_real': recall_real * 100,
        'f1_real': f1_real * 100,
        'macro_f1': macro_f1 * 100
    }


def run_ablation_variant(
    variant_name: str,
    dataset_name: str,
    samples: List[VideoSample],
    **kwargs
) -> AblationStudyResult:
    """Run a single ablation variant."""
    print(f"\n{'='*60}")
    print(f"Running: {variant_name}")
    print(f"Dataset: {dataset_name}, Samples: {len(samples)}")
    print(f"Config: {kwargs}")
    print(f"{'='*60}")
    
    pipeline = AblationPipeline(**kwargs)
    results = pipeline.process_batch(samples)
    
    # Calculate metrics
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / len(results) * 100 if results else 0
    total_time = sum(r.processing_time for r in results)
    
    metrics = calculate_metrics(results)
    
    study_result = AblationStudyResult(
        variant_name=variant_name,
        dataset=dataset_name,
        total_samples=len(results),
        correct=correct,
        accuracy=accuracy,
        precision_fake=metrics['precision_fake'],
        recall_fake=metrics['recall_fake'],
        f1_fake=metrics['f1_fake'],
        precision_real=metrics['precision_real'],
        recall_real=metrics['recall_real'],
        f1_real=metrics['f1_real'],
        macro_f1=metrics['macro_f1'],
        total_time=total_time,
        results=results
    )
    
    print(f"\nResults for {variant_name}:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Macro F1: {metrics['macro_f1']:.2f}%")
    print(f"  Total Time: {total_time:.1f}s")
    
    return study_result


def run_full_ablation_study(
    dataset_name: str = "combined",
    sample_limit: Optional[int] = None,
    variants: Optional[List[str]] = None
):
    """
    Run complete ablation study with all variants.
    
    Variants:
    - full: Full CODA (baseline)
    - no_intuitive: w/o Intuitive Toolset (LLM-only)
    - no_websearch: w/o Web Search
    - no_conflict: w/o Conflict Resolution
    - no_langaware: w/o Language-Aware Query Generation
    - no_manipulation: w/o Manipulation Detector
    """
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    _, test_samples = dataset.get_train_test_split()
    
    if sample_limit:
        test_samples = test_samples[:sample_limit]
    
    print(f"Test samples: {len(test_samples)}")
    
    # Define ablation variants
    all_variants = {
        'full': {
            'use_intuitive_toolset': True,
            'use_web_search': True,
            'use_conflict_resolution': True,
            'use_language_aware_query': True,
            'use_manipulation_detector': True
        },
        'no_intuitive': {
            'use_intuitive_toolset': False,
            'use_web_search': True,
            'use_conflict_resolution': True,
            'use_language_aware_query': True,
            'use_manipulation_detector': True
        },
        'no_websearch': {
            'use_intuitive_toolset': True,
            'use_web_search': False,
            'use_conflict_resolution': True,
            'use_language_aware_query': True,
            'use_manipulation_detector': True
        },
        'no_conflict': {
            'use_intuitive_toolset': True,
            'use_web_search': True,
            'use_conflict_resolution': False,
            'use_language_aware_query': True,
            'use_manipulation_detector': True
        },
        'no_langaware': {
            'use_intuitive_toolset': True,
            'use_web_search': True,
            'use_conflict_resolution': True,
            'use_language_aware_query': False,
            'use_manipulation_detector': True
        },
        'no_manipulation': {
            'use_intuitive_toolset': True,
            'use_web_search': True,
            'use_conflict_resolution': True,
            'use_language_aware_query': True,
            'use_manipulation_detector': False
        }
    }
    
    # Filter variants if specified
    if variants:
        all_variants = {k: v for k, v in all_variants.items() if k in variants}
    
    # Run each variant
    study_results = []
    for variant_name, config in all_variants.items():
        result = run_ablation_variant(
            variant_name=variant_name,
            dataset_name=dataset_name,
            samples=test_samples,
            **config
        )
        study_results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"logs/ablation_{dataset_name}_{timestamp}.json"
    
    # Convert to JSON-serializable format
    output_data = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'total_samples': len(test_samples),
        'variants': []
    }
    
    for result in study_results:
        variant_data = {
            'variant_name': result.variant_name,
            'accuracy': result.accuracy,
            'macro_f1': result.macro_f1,
            'precision_fake': result.precision_fake,
            'recall_fake': result.recall_fake,
            'f1_fake': result.f1_fake,
            'precision_real': result.precision_real,
            'recall_real': result.recall_real,
            'f1_real': result.f1_real,
            'total_time': result.total_time,
            'sample_results': [
                {
                    'video_id': r.video_id,
                    'ground_truth': r.ground_truth,
                    'prediction': r.prediction,
                    'is_correct': r.is_correct,
                    'stage_used': r.stage_used
                }
                for r in result.results
            ]
        }
        output_data['variants'].append(variant_data)
    
    os.makedirs('logs', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Variant':<25} {'Accuracy':>10} {'Macro F1':>10} {'Δ Acc':>10}")
    print("-" * 60)
    
    baseline_acc = study_results[0].accuracy if study_results else 0
    for result in study_results:
        delta = result.accuracy - baseline_acc
        delta_str = f"{delta:+.2f}%" if result.variant_name != 'full' else "-"
        print(f"{result.variant_name:<25} {result.accuracy:>9.2f}% {result.macro_f1:>9.2f}% {delta_str:>10}")
    
    print(f"\nResults saved to: {output_path}")
    
    return study_results


def main():
    parser = argparse.ArgumentParser(description="Run ablation study for CODA")
    parser.add_argument('--dataset', type=str, default='combined',
                        choices=['fakesv', 'fakett', 'combined'],
                        help='Dataset to use')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples (for quick testing)')
    parser.add_argument('--variants', type=str, nargs='+', default=None,
                        choices=['full', 'no_intuitive', 'no_websearch', 
                                'no_conflict', 'no_langaware', 'no_manipulation'],
                        help='Specific variants to run (default: all)')
    
    args = parser.parse_args()
    
    run_full_ablation_study(
        dataset_name=args.dataset,
        sample_limit=args.limit,
        variants=args.variants
    )


if __name__ == "__main__":
    main()

