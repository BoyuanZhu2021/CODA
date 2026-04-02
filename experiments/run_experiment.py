"""
Main experiment script for the Agentic Fake Video Detection System.

This script runs the complete pipeline including:
1. Training Stage 1 classifiers (optional)
2. Running the cascade pipeline
3. Evaluating results and generating reports

Usage:
    python run_experiment.py --mode full
    python run_experiment.py --mode evaluate --classifier models/mlp_classifier.pt
    python run_experiment.py --mode quick --limit 50
    
Random Sampling Mode:
    # Basic (50 videos from both datasets):
    python run_experiment.py --random-sample 50
    
    # With classifier:
    python run_experiment.py --random-sample 100 --classifier models/mlp_classifier.pt
    
    # FakeTT only (multilingual):
    python run_experiment.py --random-sample 100 --source tt --classifier models/mlp_classifier.pt
    
    # FakeSV only (Chinese):
    python run_experiment.py --random-sample 100 --source sv --classifier models/mlp_classifier.pt
    
    # Use FULL dataset (not just test set) - allows more samples:
    python run_experiment.py --random-sample 500 --use-full-dataset --classifier models/mlp_classifier.pt
    
    # Custom random seed:
    python run_experiment.py --random-sample 50 --seed 123

Note: Without --use-full-dataset, max samples = ~355 (test sets only: FakeSV ~177 + FakeTT ~178)
      With --use-full-dataset, max samples = 1773 (FakeSV 883 + FakeTT 890)
"""

import os
import sys
import argparse
import json
import random
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIDENCE_THRESHOLD, RANDOM_SEED
from data.loaders import load_dataset, FakeVideoDataset
from classifiers.train_classifier import run_classifier_comparison
from pipeline.cascade_pipeline import CascadePipeline, run_pipeline
from pipeline.evaluation import (
    evaluate_pipeline, 
    print_evaluation_report,
    compare_configurations,
    save_results,
    analyze_errors
)


def get_random_samples_from_datasets(
    n_samples: int,
    random_seed: int = RANDOM_SEED,
    dataset_source: str = "both",  # "both", "sv", "tt"
    use_full_dataset: bool = False  # If True, sample from all data, not just test set
) -> List:
    """
    Get random samples from FakeSV and/or FakeTT datasets.
    
    Args:
        n_samples: Total number of samples to get
        random_seed: Random seed for reproducibility
        dataset_source: "both", "sv" (FakeSV only), or "tt" (FakeTT only)
        use_full_dataset: If True, sample from all data (not just test set)
    
    Returns:
        List of VideoSample objects with dataset source tagged
    """
    random.seed(random_seed)
    
    fakesv_samples = []
    fakett_samples = []
    
    # Load datasets based on source
    if dataset_source in ["both", "sv"]:
        fakesv_dataset = FakeVideoDataset("fakesv")
        if use_full_dataset:
            fakesv_pool = fakesv_dataset.samples
        else:
            _, fakesv_pool = fakesv_dataset.get_train_test_split()
        # Tag samples with dataset source
        for s in fakesv_pool:
            s.dataset_source = "fakesv"
        print(f"FakeSV pool: {len(fakesv_pool)} samples available")
    else:
        fakesv_pool = []
    
    if dataset_source in ["both", "tt"]:
        fakett_dataset = FakeVideoDataset("fakett")
        if use_full_dataset:
            fakett_pool = fakett_dataset.samples
        else:
            _, fakett_pool = fakett_dataset.get_train_test_split()
        # Tag samples with dataset source
        for s in fakett_pool:
            s.dataset_source = "fakett"
        print(f"FakeTT pool: {len(fakett_pool)} samples available")
    else:
        fakett_pool = []
    
    # Calculate how many to sample from each
    if dataset_source == "sv":
        n_fakesv = min(n_samples, len(fakesv_pool))
        n_fakett = 0
    elif dataset_source == "tt":
        n_fakesv = 0
        n_fakett = min(n_samples, len(fakett_pool))
    else:  # both
        n_per_dataset = n_samples // 2
        n_fakesv = min(n_per_dataset, len(fakesv_pool))
        n_fakett = min(n_samples - n_fakesv, len(fakett_pool))
        # If one dataset is smaller, take more from the other
        if n_fakesv < n_per_dataset:
            n_fakett = min(n_samples - n_fakesv, len(fakett_pool))
        elif n_fakett < n_per_dataset:
            n_fakesv = min(n_samples - n_fakett, len(fakesv_pool))
    
    # Sample from each
    if n_fakesv > 0:
        fakesv_samples = random.sample(fakesv_pool, n_fakesv)
    if n_fakett > 0:
        fakett_samples = random.sample(fakett_pool, n_fakett)
    
    # Combine and shuffle
    combined = fakesv_samples + fakett_samples
    random.shuffle(combined)
    
    print(f"\nRandom sampling: {n_fakesv} from FakeSV, {n_fakett} from FakeTT")
    print(f"Total samples: {len(combined)}")
    if use_full_dataset:
        print("(Using FULL dataset, not just test set)")
    
    # Print class distribution
    if combined:
        fake_count = sum(1 for s in combined if s.ground_truth == 'fake')
        real_count = len(combined) - fake_count
        print(f"Class distribution: {fake_count} fake ({100*fake_count/len(combined):.1f}%), "
              f"{real_count} real ({100*real_count/len(combined):.1f}%)")
    
    return combined, n_fakesv, n_fakett


def run_full_experiment(
    dataset_name: str = "combined",
    train_classifier: bool = True,
    include_transformer: bool = False,
    use_web_search: bool = True,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    test_limit: Optional[int] = None,
    save_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run the complete experiment pipeline.
    
    Args:
        dataset_name: Dataset to use
        train_classifier: Whether to train Stage 1 classifier
        include_transformer: Whether to include transformer in classifier comparison
        use_web_search: Whether to enable web search in Stage 3
        confidence_threshold: Threshold for Stage 1
        test_limit: Limit test samples (for quick testing)
        save_dir: Directory to save results
    
    Returns:
        Dictionary with all experiment results
    """
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  AGENTIC FAKE VIDEO DETECTION - FULL EXPERIMENT".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), save_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    experiment_results = {
        'timestamp': timestamp,
        'config': {
            'dataset': dataset_name,
            'train_classifier': train_classifier,
            'include_transformer': include_transformer,
            'use_web_search': use_web_search,
            'confidence_threshold': confidence_threshold,
            'test_limit': test_limit
        }
    }
    
    # =========================================
    # PHASE 1: Train Classifiers (Optional)
    # =========================================
    classifier_path = None
    
    if train_classifier:
        print("\n" + "="*60)
        print(" PHASE 1: TRAINING STAGE 1 CLASSIFIERS")
        print("="*60)
        
        classifier_results = run_classifier_comparison(
            dataset_name=dataset_name,
            include_transformer=include_transformer,
            save_models=True
        )
        
        experiment_results['classifier_comparison'] = classifier_results
        
        # Use best classifier for pipeline
        best_model = classifier_results.get('best_model', 'mlp')
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        if best_model == 'logreg':
            classifier_path = os.path.join(models_dir, 'logreg_classifier.pkl')
        elif best_model == 'mlp':
            classifier_path = os.path.join(models_dir, 'mlp_classifier.pt')
        elif best_model == 'transformer':
            classifier_path = os.path.join(models_dir, 'transformer_classifier')
        
        print(f"\nUsing {best_model} classifier for pipeline")
    
    # =========================================
    # PHASE 2: Run Cascade Pipeline
    # =========================================
    print("\n" + "="*60)
    print(" PHASE 2: RUNNING CASCADE PIPELINE")
    print("="*60)
    
    pipeline_results, pipeline_stats = run_pipeline(
        dataset_name=dataset_name,
        classifier_path=classifier_path,
        confidence_threshold=confidence_threshold,
        use_web_search=use_web_search,
        limit=test_limit
    )
    
    experiment_results['pipeline_stats'] = pipeline_stats
    
    # =========================================
    # PHASE 3: Evaluation
    # =========================================
    print("\n" + "="*60)
    print(" PHASE 3: DETAILED EVALUATION")
    print("="*60)
    
    metrics = print_evaluation_report(pipeline_results, "Cascade Pipeline Evaluation")
    experiment_results['evaluation_metrics'] = metrics
    
    # Error analysis
    error_analysis = analyze_errors(pipeline_results, n_samples=10)
    experiment_results['error_analysis'] = error_analysis
    
    # =========================================
    # Save Results
    # =========================================
    save_results(
        pipeline_results, 
        metrics,
        os.path.join(results_dir, 'pipeline_results.json')
    )
    
    # Save full experiment results
    with open(os.path.join(results_dir, 'experiment_results.json'), 'w') as f:
        # Convert non-serializable items
        serializable_results = json.loads(json.dumps(experiment_results, default=str))
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✅ Experiment complete! Results saved to: {results_dir}")
    
    # Final summary
    print("\n" + "="*60)
    print(" EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\n  Dataset:           {dataset_name}")
    print(f"  Total Samples:     {pipeline_stats['total_samples']}")
    print(f"  Overall Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  F1 Score:          {metrics['f1_score']:.4f}")
    print(f"  API Usage:         {pipeline_stats['api_usage_rate']:.1%} of samples")
    
    if metrics['accuracy'] >= 0.80:
        print(f"\n  🎉 TARGET ACHIEVED! Accuracy >= 80%")
    else:
        print(f"\n  ⚠️  Accuracy below 80% target. Consider:")
        print(f"      - Adjusting confidence threshold")
        print(f"      - Fine-tuning the classifier with more data")
        print(f"      - Improving claim extraction prompts")
    
    return experiment_results


def run_quick_test(
    dataset_name: str = "combined",
    limit: int = 20,
    use_classifier: bool = False
) -> Dict[str, Any]:
    """
    Run a quick test with a small number of samples.
    Useful for testing the pipeline without full training.
    
    Args:
        dataset_name: Dataset to use
        limit: Number of samples to test
        use_classifier: Whether to use existing classifier
    
    Returns:
        Test results
    """
    print("\n" + "#"*60)
    print("# QUICK TEST MODE")
    print("#"*60)
    
    # Check for existing classifier
    classifier_path = None
    if use_classifier:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        for filename in ['mlp_classifier.pt', 'logreg_classifier.pkl']:
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                classifier_path = path
                break
    
    # Run pipeline
    results, stats = run_pipeline(
        dataset_name=dataset_name,
        classifier_path=classifier_path,
        use_web_search=True,
        limit=limit
    )
    
    # Print evaluation
    metrics = print_evaluation_report(results, "Quick Test Results")
    
    return {
        'stats': stats,
        'metrics': metrics
    }


def run_random_sample_experiment(
    n_samples: int,
    classifier_path: Optional[str] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    use_web_search: bool = True,
    random_seed: int = RANDOM_SEED,
    dataset_source: str = "both",  # "both", "sv", "tt"
    use_full_dataset: bool = False  # Sample from all data, not just test set
) -> Dict[str, Any]:
    """
    Run experiment on random samples from datasets.
    
    Args:
        n_samples: Total number of samples to test
        classifier_path: Path to pre-trained classifier (optional)
        confidence_threshold: Threshold for Stage 1
        use_web_search: Whether to enable web search
        random_seed: Random seed for reproducibility
        dataset_source: "both", "sv" (FakeSV only), or "tt" (FakeTT only)
        use_full_dataset: If True, sample from all data, not just test set
    
    Returns:
        Experiment results
    """
    source_name = {"both": "FakeSV + FakeTT", "sv": "FakeSV Only", "tt": "FakeTT Only"}[dataset_source]
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + f"  RANDOM SAMPLE EXPERIMENT - {n_samples} Videos".center(68) + "#")
    print("#" + f"  Dataset: {source_name}".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Get random samples from datasets
    test_samples, n_fakesv, n_fakett = get_random_samples_from_datasets(
        n_samples, random_seed, dataset_source, use_full_dataset
    )
    
    if not test_samples:
        print("ERROR: No samples available!")
        return {}
    
    # Set up classifier if provided
    classifier = None
    if classifier_path:
        # Handle relative paths - try both relative to cwd and relative to project root
        if os.path.exists(classifier_path):
            resolved_path = classifier_path
        else:
            # Try relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            resolved_path = os.path.join(project_root, classifier_path)
        
        if os.path.exists(resolved_path):
            from classifiers.models import MLPClassifier, EmbeddingClassifier
            print(f"\nLoading classifier from: {resolved_path}")
            
            if 'mlp' in resolved_path.lower():
                classifier = MLPClassifier()
            else:
                classifier = EmbeddingClassifier()
            classifier.load(resolved_path)
            print(f"✓ Classifier loaded successfully!")
            
            # Verify classifier works by testing
            test_text = "This is a test video about news"
            test_proba = classifier.predict_proba([test_text])[0]
            print(f"  Classifier test: proba = {test_proba}, max_conf = {max(test_proba):.3f}")
        else:
            print(f"\n⚠ WARNING: Classifier file not found at:")
            print(f"  - {classifier_path}")
            print(f"  - {resolved_path}")
            print("  Using LLM for all samples")
    else:
        print("\nNo classifier provided - using LLM for all samples")
    
    # Create pipeline
    pipeline = CascadePipeline(
        classifier=classifier,
        confidence_threshold=confidence_threshold,
        use_web_search=use_web_search,
        verbose=True
    )
    
    print(f"\nPipeline Configuration:")
    print(f"  - Stage 1 Classifier: {'Enabled' if classifier else 'Disabled'}")
    print(f"  - Confidence Threshold: {confidence_threshold}")
    print(f"  - Web Search: {'Enabled' if use_web_search else 'Disabled'}")
    print(f"  - Random Seed: {random_seed}")
    print(f"  - Dataset Source: {source_name}")
    print(f"  - Use Full Dataset: {use_full_dataset}")
    
    # Process samples
    print("\n" + "="*60)
    print(" PROCESSING SAMPLES")
    print("="*60)
    
    results = pipeline.process_batch(test_samples)
    
    # Stage distribution summary
    stage_counts = {1: 0, 2: 0, 3: 0}
    for r in results:
        stage_counts[r.stage_used] += 1
    
    print("\n" + "="*60)
    print(" STAGE DISTRIBUTION")
    print("="*60)
    print(f"  Stage 1 (Classifier only): {stage_counts[1]} samples ({100*stage_counts[1]/len(results):.1f}%)")
    print(f"  Stage 2 (LLM, no search):  {stage_counts[2]} samples ({100*stage_counts[2]/len(results):.1f}%)")
    print(f"  Stage 3 (LLM + web search):{stage_counts[3]} samples ({100*stage_counts[3]/len(results):.1f}%)")
    
    if stage_counts[1] == 0 and classifier is not None:
        print("\n⚠ WARNING: No samples used Stage 1 classifier!")
        print("  This may indicate classifier confidence is below threshold for all samples.")
        print(f"  Current threshold: {confidence_threshold}")
        print("  Try lowering the threshold with --threshold 0.6")
    
    # Evaluate - Overall
    print("\n" + "="*60)
    print(" OVERALL RESULTS")
    print("="*60)
    
    print_evaluation_report(results)
    metrics = evaluate_pipeline(results)
    
    # Per-dataset evaluation
    per_dataset_metrics = {}
    
    if dataset_source == "both" and n_fakesv > 0 and n_fakett > 0:
        # Separate results by dataset
        fakesv_results = [r for r, s in zip(results, test_samples) 
                         if hasattr(s, 'dataset_source') and s.dataset_source == "fakesv"]
        fakett_results = [r for r, s in zip(results, test_samples) 
                         if hasattr(s, 'dataset_source') and s.dataset_source == "fakett"]
        
        if fakesv_results:
            print("\n" + "="*60)
            print(" FAKESV RESULTS (Chinese Content)")
            print("="*60)
            print_evaluation_report(fakesv_results, "FakeSV Performance")
            per_dataset_metrics['fakesv'] = evaluate_pipeline(fakesv_results)
        
        if fakett_results:
            print("\n" + "="*60)
            print(" FAKETT RESULTS (Multilingual Content)")
            print("="*60)
            print_evaluation_report(fakett_results, "FakeTT Performance")
            per_dataset_metrics['fakett'] = evaluate_pipeline(fakett_results)
        
        # Print comparison
        print("\n" + "="*60)
        print(" DATASET COMPARISON")
        print("="*60)
        print(f"  {'Dataset':<15} {'Samples':<10} {'Accuracy':<12} {'F1':<10}")
        print(f"  {'-'*47}")
        if 'fakesv' in per_dataset_metrics:
            sv_m = per_dataset_metrics['fakesv']
            print(f"  {'FakeSV':<15} {len(fakesv_results):<10} {sv_m['accuracy']:.4f}       {sv_m['f1_score']:.4f}")
        if 'fakett' in per_dataset_metrics:
            tt_m = per_dataset_metrics['fakett']
            print(f"  {'FakeTT':<15} {len(fakett_results):<10} {tt_m['accuracy']:.4f}       {tt_m['f1_score']:.4f}")
        print(f"  {'OVERALL':<15} {len(results):<10} {metrics['accuracy']:.4f}       {metrics['f1_score']:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    result_path = os.path.join(results_dir, f'random_sample_{len(test_samples)}_{timestamp}.json')
    
    # Use correct save_results signature
    save_results(results, metrics, result_path)
    
    # Also save extended info
    extended_path = os.path.join(results_dir, f'random_sample_{len(test_samples)}_{timestamp}_extended.json')
    with open(extended_path, 'w') as f:
        json.dump({
            'n_samples_requested': n_samples,
            'n_samples_actual': len(test_samples),
            'n_fakesv': n_fakesv,
            'n_fakett': n_fakett,
            'random_seed': random_seed,
            'dataset_source': dataset_source,
            'use_full_dataset': use_full_dataset,
            'classifier_used': classifier_path is not None,
            'use_web_search': use_web_search,
            'confidence_threshold': confidence_threshold,
            'overall_metrics': metrics,
            'per_dataset_metrics': per_dataset_metrics,
            'sample_ids': [s.video_id for s in test_samples]
        }, f, indent=2)
    
    print(f"\nExtended results saved to: {extended_path}")
    
    return {
        'metrics': metrics,
        'per_dataset_metrics': per_dataset_metrics,
        'results': results,
        'samples': test_samples
    }


def run_ablation_study(
    dataset_name: str = "combined",
    test_limit: Optional[int] = 100
) -> Dict[str, Any]:
    """
    Run ablation study comparing different pipeline configurations.
    
    Args:
        dataset_name: Dataset to use
        test_limit: Limit test samples
    
    Returns:
        Comparison results
    """
    print("\n" + "#"*60)
    print("# ABLATION STUDY")
    print("#"*60)
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    _, test_samples = dataset.get_train_test_split()
    
    if test_limit:
        test_samples = test_samples[:test_limit]
    
    configurations = []
    
    # Config 1: LLM only (no classifier)
    print("\n--- Configuration 1: LLM Only (No Stage 1 Classifier) ---")
    pipeline1 = CascadePipeline(classifier=None, use_web_search=True, verbose=False)
    results1 = pipeline1.process_batch(test_samples)
    configurations.append(("LLM Only", results1))
    
    # Config 2: LLM without web search
    print("\n--- Configuration 2: LLM Without Web Search ---")
    pipeline2 = CascadePipeline(classifier=None, use_web_search=False, verbose=False)
    results2 = pipeline2.process_batch(test_samples)
    configurations.append(("LLM No Search", results2))
    
    # Config 3: With classifier (if available)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    classifier_path = os.path.join(models_dir, 'mlp_classifier.pt')
    
    if os.path.exists(classifier_path):
        print("\n--- Configuration 3: Full Pipeline (Classifier + LLM + Search) ---")
        from classifiers.models import MLPClassifier
        classifier = MLPClassifier()
        classifier.load(classifier_path)
        
        pipeline3 = CascadePipeline(
            classifier=classifier,
            confidence_threshold=0.75,
            use_web_search=True,
            verbose=False
        )
        results3 = pipeline3.process_batch(test_samples)
        configurations.append(("Full Pipeline (0.75)", results3))
        
        # Config 4: Higher threshold
        print("\n--- Configuration 4: Full Pipeline (Higher Threshold 0.85) ---")
        pipeline4 = CascadePipeline(
            classifier=classifier,
            confidence_threshold=0.85,
            use_web_search=True,
            verbose=False
        )
        results4 = pipeline4.process_batch(test_samples)
        configurations.append(("Full Pipeline (0.85)", results4))
    
    # Compare all configurations
    comparison = compare_configurations(configurations)
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run Agentic Fake Video Detection Experiments"
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='full',
        choices=['full', 'quick', 'ablation', 'evaluate'],
        help='Experiment mode'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='combined',
        choices=['fakesv', 'fakett', 'combined'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--limit', 
        type=int, 
        default=None,
        help='Limit number of test samples'
    )
    parser.add_argument(
        '--no-train', 
        action='store_true',
        help='Skip classifier training'
    )
    parser.add_argument(
        '--no-web-search', 
        action='store_true',
        help='Disable web search'
    )
    parser.add_argument(
        '--include-transformer', 
        action='store_true',
        help='Include transformer in classifier comparison'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=CONFIDENCE_THRESHOLD,
        help='Confidence threshold for Stage 1'
    )
    parser.add_argument(
        '--classifier', 
        type=str, 
        default=None,
        help='Path to pre-trained classifier'
    )
    parser.add_argument(
        '--random-sample',
        type=int,
        default=None,
        help='Randomly sample N videos from datasets'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help='Random seed for sampling'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='both',
        choices=['both', 'sv', 'tt'],
        help='Dataset source: both (FakeSV+FakeTT), sv (FakeSV only), tt (FakeTT only)'
    )
    parser.add_argument(
        '--use-full-dataset',
        action='store_true',
        help='Sample from all data, not just test set (allows more samples)'
    )
    
    args = parser.parse_args()
    
    # Handle random sampling mode
    if args.random_sample:
        run_random_sample_experiment(
            n_samples=args.random_sample,
            classifier_path=args.classifier,
            confidence_threshold=args.threshold,
            use_web_search=not args.no_web_search,
            random_seed=args.seed,
            dataset_source=args.source,
            use_full_dataset=args.use_full_dataset
        )
        return
    
    if args.mode == 'full':
        run_full_experiment(
            dataset_name=args.dataset,
            train_classifier=not args.no_train,
            include_transformer=args.include_transformer,
            use_web_search=not args.no_web_search,
            confidence_threshold=args.threshold,
            test_limit=args.limit
        )
    
    elif args.mode == 'quick':
        run_quick_test(
            dataset_name=args.dataset,
            limit=args.limit or 20,
            use_classifier=args.classifier is not None
        )
    
    elif args.mode == 'ablation':
        run_ablation_study(
            dataset_name=args.dataset,
            test_limit=args.limit or 100
        )
    
    elif args.mode == 'evaluate':
        if args.classifier is None:
            print("Error: --classifier path required for evaluate mode")
            return
        
        run_pipeline(
            dataset_name=args.dataset,
            classifier_path=args.classifier,
            confidence_threshold=args.threshold,
            use_web_search=not args.no_web_search,
            limit=args.limit
        )


if __name__ == "__main__":
    main()

