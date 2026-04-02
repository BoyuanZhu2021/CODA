"""
Cascading Pipeline for Fake Video Detection

This pipeline implements a 3-stage approach:
1. Stage 1: Lightweight classifier for high-confidence predictions
2. Stage 2: LLM claim extraction for uncertain samples
3. Stage 3: Web search verification for verifiable claims

The cascade approach saves API costs by only using LLM/search when necessary.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIDENCE_THRESHOLD
from data.loaders import VideoSample, load_dataset
from classifiers.models import EmbeddingClassifier, MLPClassifier
from agents.claim_extractor import ClaimExtractorAgent, ClaimExtractionResult
from agents.verification_agent import VerificationAgent, VerificationResult
from agents.judge_agent import JudgeAgent, JudgmentResult
from utils.logger import PipelineLogger, get_pipeline_logger


@dataclass
class PipelineResult:
    """Result from the complete pipeline for a single video."""
    video_id: str
    ground_truth: str
    prediction: str  # "fake" or "real"
    prediction_label: int  # 1 = fake, 0 = real
    confidence: float
    stage_used: int  # 1, 2, or 3
    stage1_prediction: Optional[str] = None
    stage1_confidence: Optional[float] = None
    claim_result: Optional[ClaimExtractionResult] = None
    verification_result: Optional[VerificationResult] = None
    judgment_result: Optional[JudgmentResult] = None
    processing_time: float = 0.0
    
    @property
    def is_correct(self) -> bool:
        """Check if prediction matches ground truth."""
        return self.prediction == self.ground_truth
    
    @property
    def ground_truth_label(self) -> int:
        """Get ground truth as label."""
        return 1 if self.ground_truth == "fake" else 0


class CascadePipeline:
    """
    Three-stage cascade pipeline for fake video detection.
    
    Stage 1: Fast classifier (no API cost)
        - High confidence predictions go directly to output
        - Low confidence samples proceed to Stage 2
    
    Stage 2: LLM Claim Analysis (API cost per sample)
        - Extract and analyze claims
        - Identify if web search is needed
        - Make judgment if no verification needed
    
    Stage 3: Web Search Verification (API cost + search)
        - Verify claims using language-aware web search
        - Final judgment with all evidence
    """
    
    def __init__(
        self,
        classifier: Optional[Any] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        use_web_search: bool = True,
        verbose: bool = True,
        enable_logging: bool = True
    ):
        """
        Initialize the cascade pipeline.
        
        Args:
            classifier: Pre-trained Stage 1 classifier (or None to skip Stage 1)
            confidence_threshold: Threshold for Stage 1 to proceed to Stage 2
            use_web_search: Whether to use web search in Stage 3
            verbose: Whether to print progress
            enable_logging: Whether to enable detailed logging
        """
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.use_web_search = use_web_search
        self.verbose = verbose
        self.enable_logging = enable_logging
        
        # Initialize loggers
        self.pipeline_logger = PipelineLogger() if enable_logging else None
        self.logger = get_pipeline_logger()
        
        # Initialize agents (lazy loading)
        self._claim_agent = None
        self._verification_agent = None
        self._judge_agent = None
    
    @property
    def claim_agent(self) -> ClaimExtractorAgent:
        if self._claim_agent is None:
            self._claim_agent = ClaimExtractorAgent()
        return self._claim_agent
    
    @property
    def verification_agent(self) -> VerificationAgent:
        if self._verification_agent is None:
            self._verification_agent = VerificationAgent()
        return self._verification_agent
    
    @property
    def judge_agent(self) -> JudgeAgent:
        if self._judge_agent is None:
            self._judge_agent = JudgeAgent()
        return self._judge_agent
    
    def process_sample(self, sample: VideoSample) -> PipelineResult:
        """
        Process a single video sample through the cascade pipeline.
        
        Args:
            sample: VideoSample to process
        
        Returns:
            PipelineResult with prediction and metadata
        """
        start_time = time.time()
        
        video_id = sample.video_id
        ground_truth = sample.ground_truth
        
        self.logger.debug(f"[{video_id}] Processing sample...")
        
        # =========================================
        # STAGE 1: Classifier
        # =========================================
        stage1_prediction = None
        stage1_confidence = None
        needs_web_search = None
        
        if self.classifier is not None:
            proba = self.classifier.predict_proba([sample.combined_text])[0]
            stage1_prediction = "fake" if np.argmax(proba) == 1 else "real"
            stage1_confidence = float(np.max(proba))
            
            self.logger.debug(f"[{video_id}] Stage 1: {stage1_prediction} (conf: {stage1_confidence:.3f})")
            
            # If high confidence, return Stage 1 result
            if stage1_confidence >= self.confidence_threshold:
                result = PipelineResult(
                    video_id=video_id,
                    ground_truth=ground_truth,
                    prediction=stage1_prediction,
                    prediction_label=1 if stage1_prediction == "fake" else 0,
                    confidence=stage1_confidence,
                    stage_used=1,
                    stage1_prediction=stage1_prediction,
                    stage1_confidence=stage1_confidence,
                    processing_time=time.time() - start_time
                )
                self._log_result(result)
                return result
        
        # =========================================
        # STAGE 2: Claim Extraction
        # =========================================
        self.logger.debug(f"[{video_id}] Proceeding to Stage 2 (claim extraction)...")
        claim_result = self.claim_agent.extract_claims(sample.raw_data, video_id)
        needs_web_search = claim_result.needs_web_search
        
        # Check if web search is needed
        if not self.use_web_search or not claim_result.needs_web_search:
            # Make judgment without web search
            judgment = self.judge_agent.quick_judgment_from_claims(claim_result)
            
            result = PipelineResult(
                video_id=video_id,
                ground_truth=ground_truth,
                prediction=judgment.verdict,
                prediction_label=judgment.prediction_label,
                confidence=judgment.confidence,
                stage_used=2,
                stage1_prediction=stage1_prediction,
                stage1_confidence=stage1_confidence,
                claim_result=claim_result,
                judgment_result=judgment,
                processing_time=time.time() - start_time
            )
            self._log_result(result, needs_web_search)
            return result
        
        # =========================================
        # STAGE 3: Web Search Verification
        # =========================================
        self.logger.debug(f"[{video_id}] Proceeding to Stage 3 (web verification)...")
        verification_result = self.verification_agent.verify_claims(claim_result)
        
        # Make final judgment with all evidence
        classifier_info = None
        if stage1_prediction is not None:
            classifier_info = {
                'prediction': stage1_prediction,
                'confidence': stage1_confidence
            }
        
        judgment = self.judge_agent.make_judgment(
            claim_result, 
            verification_result,
            classifier_info
        )
        
        result = PipelineResult(
            video_id=video_id,
            ground_truth=ground_truth,
            prediction=judgment.verdict,
            prediction_label=judgment.prediction_label,
            confidence=judgment.confidence,
            stage_used=3,
            stage1_prediction=stage1_prediction,
            stage1_confidence=stage1_confidence,
            claim_result=claim_result,
            verification_result=verification_result,
            judgment_result=judgment,
            processing_time=time.time() - start_time
        )
        self._log_result(result, needs_web_search)
        return result
    
    def _log_result(self, result: PipelineResult, needs_web_search: bool = None):
        """Log a pipeline result."""
        if self.pipeline_logger:
            self.pipeline_logger.log_sample(
                video_id=result.video_id,
                ground_truth=result.ground_truth,
                stage_used=result.stage_used,
                prediction=result.prediction,
                confidence=result.confidence,
                processing_time=result.processing_time,
                stage1_prediction=result.stage1_prediction,
                stage1_confidence=result.stage1_confidence,
                needs_web_search=needs_web_search
            )
    
    def process_batch(
        self, 
        samples: List[VideoSample],
        show_progress: bool = True,
        dataset_name: str = "unknown"
    ) -> List[PipelineResult]:
        """
        Process a batch of video samples.
        
        Args:
            samples: List of VideoSample objects
            show_progress: Whether to show progress bar
            dataset_name: Name of the dataset being processed
        
        Returns:
            List of PipelineResult objects
        """
        # Log batch start
        if self.pipeline_logger:
            self.pipeline_logger.start_batch(len(samples), dataset_name)
        
        self.logger.info(f"Starting batch processing: {len(samples)} samples from {dataset_name}")
        
        results = []
        correct_count = 0
        total_count = 0
        
        iterator = samples
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(samples, desc="Processing samples")
        
        for sample in iterator:
            result = self.process_sample(sample)
            results.append(result)
            
            # Track runtime accuracy
            total_count += 1
            if result.is_correct:
                correct_count += 1
            runtime_accuracy = correct_count / total_count
            
            # Update progress bar with runtime accuracy
            if show_progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'acc': f'{runtime_accuracy:.1%}',
                    'correct': f'{correct_count}/{total_count}'
                })
            
            if self.verbose and not show_progress:
                status = "✓" if result.is_correct else "✗"
                print(f"  {status} {result.video_id}: Stage {result.stage_used} -> "
                      f"{result.prediction} (truth: {result.ground_truth}) | "
                      f"Running Acc: {runtime_accuracy:.1%}")
            
            # Periodic GPU cache clearing to prevent memory buildup (every 50 samples)
            if total_count % 50 == 0:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Print final runtime accuracy
        final_accuracy = correct_count / total_count if total_count > 0 else 0
        self.logger.info(f"Batch complete: {correct_count}/{total_count} correct ({final_accuracy:.2%})")
        
        # Log batch end
        if self.pipeline_logger:
            self.pipeline_logger.end_batch()
            self.pipeline_logger.save_results()
        
        return results
    
    def get_statistics(self, results: List[PipelineResult]) -> Dict[str, Any]:
        """
        Calculate statistics from pipeline results.
        
        Args:
            results: List of PipelineResult objects
        
        Returns:
            Dictionary with statistics
        """
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        
        # Stage distribution
        stage_counts = {1: 0, 2: 0, 3: 0}
        stage_correct = {1: 0, 2: 0, 3: 0}
        
        for r in results:
            stage_counts[r.stage_used] += 1
            if r.is_correct:
                stage_correct[r.stage_used] += 1
        
        # Calculate per-stage accuracy
        stage_accuracy = {}
        for stage in [1, 2, 3]:
            if stage_counts[stage] > 0:
                stage_accuracy[stage] = stage_correct[stage] / stage_counts[stage]
            else:
                stage_accuracy[stage] = 0.0
        
        # Processing time
        total_time = sum(r.processing_time for r in results)
        avg_time = total_time / total if total > 0 else 0
        
        # API usage estimate
        api_calls = sum(1 for r in results if r.stage_used >= 2)
        web_searches = sum(1 for r in results if r.stage_used == 3)
        
        return {
            'total_samples': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
            'stage_distribution': stage_counts,
            'stage_coverage': {k: v/total for k, v in stage_counts.items()},
            'stage_accuracy': stage_accuracy,
            'total_time_seconds': total_time,
            'avg_time_per_sample': avg_time,
            'api_calls': api_calls,
            'web_searches': web_searches,
            'api_usage_rate': api_calls / total if total > 0 else 0
        }
    
    def print_statistics(self, results: List[PipelineResult]):
        """Print formatted statistics."""
        stats = self.get_statistics(results)
        
        print("\n" + "="*60)
        print("PIPELINE STATISTICS")
        print("="*60)
        
        print(f"\nOverall Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total_samples']})")
        
        print("\nStage Distribution:")
        for stage in [1, 2, 3]:
            count = stats['stage_distribution'][stage]
            coverage = stats['stage_coverage'][stage]
            acc = stats['stage_accuracy'][stage]
            print(f"  Stage {stage}: {count} samples ({coverage:.1%}), Accuracy: {acc:.2%}")
        
        print(f"\nAPI Usage:")
        print(f"  LLM API calls: {stats['api_calls']} ({stats['api_usage_rate']:.1%} of samples)")
        print(f"  Web searches: {stats['web_searches']}")
        
        print(f"\nProcessing Time:")
        print(f"  Total: {stats['total_time_seconds']:.1f}s")
        print(f"  Average per sample: {stats['avg_time_per_sample']:.2f}s")


def run_pipeline(
    dataset_name: str = "combined",
    classifier_path: Optional[str] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    use_web_search: bool = True,
    limit: Optional[int] = None
) -> Tuple[List[PipelineResult], Dict[str, Any]]:
    """
    Run the complete cascade pipeline on a dataset.
    
    Args:
        dataset_name: Dataset to use ('fakesv', 'fakett', 'combined')
        classifier_path: Path to pre-trained classifier (or None)
        confidence_threshold: Stage 1 confidence threshold
        use_web_search: Whether to enable web search
        limit: Limit number of samples (for testing)
    
    Returns:
        Tuple of (results list, statistics dict)
    """
    print("\n" + "#"*60)
    print("# CASCADE PIPELINE EXECUTION")
    print("#"*60)
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    _, test_samples = dataset.get_train_test_split()
    
    if limit:
        test_samples = test_samples[:limit]
    
    print(f"\nDataset: {dataset_name}")
    print(f"Test samples: {len(test_samples)}")
    
    # Load classifier if provided
    classifier = None
    if classifier_path and os.path.exists(classifier_path):
        print(f"Loading classifier from: {classifier_path}")
        if classifier_path.endswith('.pkl'):
            classifier = EmbeddingClassifier()
            classifier.load(classifier_path)
        elif classifier_path.endswith('.pt'):
            classifier = MLPClassifier()
            classifier.load(classifier_path)
        print("Classifier loaded successfully")
    
    # Initialize pipeline
    pipeline = CascadePipeline(
        classifier=classifier,
        confidence_threshold=confidence_threshold,
        use_web_search=use_web_search,
        verbose=True
    )
    
    # Process samples
    print("\nProcessing samples...")
    results = pipeline.process_batch(test_samples)
    
    # Print statistics
    pipeline.print_statistics(results)
    
    return results, pipeline.get_statistics(results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cascade pipeline")
    parser.add_argument('--dataset', type=str, default='combined',
                        choices=['fakesv', 'fakett', 'combined'])
    parser.add_argument('--classifier', type=str, default=None,
                        help='Path to pre-trained classifier')
    parser.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD,
                        help='Confidence threshold for Stage 1')
    parser.add_argument('--no-web-search', action='store_true',
                        help='Disable web search')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples')
    
    args = parser.parse_args()
    
    results, stats = run_pipeline(
        dataset_name=args.dataset,
        classifier_path=args.classifier,
        confidence_threshold=args.threshold,
        use_web_search=not args.no_web_search,
        limit=args.limit
    )

