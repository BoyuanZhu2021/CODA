"""Evaluation utilities for the fake video detection pipeline."""

import os
import sys
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.cascade_pipeline import PipelineResult


def evaluate_pipeline(results: List[PipelineResult]) -> Dict[str, Any]:
    """
    Comprehensive evaluation of pipeline results.
    
    Args:
        results: List of PipelineResult objects
    
    Returns:
        Dictionary with all evaluation metrics
    """
    y_true = [r.ground_truth_label for r in results]
    y_pred = [r.prediction_label for r in results]
    confidences = [r.confidence for r in results]
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # AUC if we have varied confidences
    try:
        # Use confidence for fake class (label=1)
        fake_confidences = [c if p == 1 else 1-c for c, p in zip(confidences, y_pred)]
        metrics['roc_auc'] = roc_auc_score(y_true, fake_confidences)
    except ValueError:
        metrics['roc_auc'] = None
    
    # Per-stage metrics
    stage_metrics = {}
    for stage in [1, 2, 3]:
        stage_results = [r for r in results if r.stage_used == stage]
        if stage_results:
            stage_y_true = [r.ground_truth_label for r in stage_results]
            stage_y_pred = [r.prediction_label for r in stage_results]
            
            stage_metrics[stage] = {
                'count': len(stage_results),
                'coverage': len(stage_results) / len(results),
                'accuracy': accuracy_score(stage_y_true, stage_y_pred),
                'precision': precision_score(stage_y_true, stage_y_pred, zero_division=0),
                'recall': recall_score(stage_y_true, stage_y_pred, zero_division=0),
                'f1_score': f1_score(stage_y_true, stage_y_pred, zero_division=0)
            }
    
    metrics['per_stage'] = stage_metrics
    
    # Class-specific metrics
    fake_results = [r for r in results if r.ground_truth == "fake"]
    real_results = [r for r in results if r.ground_truth == "real"]
    
    metrics['per_class'] = {
        'fake': {
            'total': len(fake_results),
            'correct': sum(1 for r in fake_results if r.is_correct),
            'accuracy': sum(1 for r in fake_results if r.is_correct) / len(fake_results) if fake_results else 0
        },
        'real': {
            'total': len(real_results),
            'correct': sum(1 for r in real_results if r.is_correct),
            'accuracy': sum(1 for r in real_results if r.is_correct) / len(real_results) if real_results else 0
        }
    }
    
    # Confidence analysis
    correct_confidences = [r.confidence for r in results if r.is_correct]
    incorrect_confidences = [r.confidence for r in results if not r.is_correct]
    
    metrics['confidence_analysis'] = {
        'mean_overall': np.mean(confidences),
        'mean_correct': np.mean(correct_confidences) if correct_confidences else 0,
        'mean_incorrect': np.mean(incorrect_confidences) if incorrect_confidences else 0
    }
    
    # Error analysis
    errors = [r for r in results if not r.is_correct]
    false_positives = [r for r in errors if r.prediction == "fake" and r.ground_truth == "real"]
    false_negatives = [r for r in errors if r.prediction == "real" and r.ground_truth == "fake"]
    
    metrics['error_analysis'] = {
        'total_errors': len(errors),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'error_rate': len(errors) / len(results) if results else 0
    }
    
    return metrics


def print_evaluation_report(results: List[PipelineResult], title: str = "Pipeline Evaluation"):
    """
    Print a formatted evaluation report.
    
    Args:
        results: List of PipelineResult objects
        title: Report title
    """
    metrics = evaluate_pipeline(results)
    
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)
    
    # Overall metrics
    print("\n📊 OVERALL METRICS")
    print("-"*40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if metrics['roc_auc']:
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    print("\n📋 CONFUSION MATRIX")
    print("-"*40)
    print("              Predicted")
    print("              Real    Fake")
    print(f"  Actual Real  {cm[0][0]:<6}  {cm[0][1]:<6}")
    print(f"  Actual Fake  {cm[1][0]:<6}  {cm[1][1]:<6}")
    
    # Per-stage performance
    print("\n🔄 PER-STAGE PERFORMANCE")
    print("-"*40)
    print(f"  {'Stage':<8} {'Count':<8} {'Coverage':<10} {'Accuracy':<10} {'F1':<10}")
    print(f"  {'-'*46}")
    
    for stage, stage_data in metrics['per_stage'].items():
        print(f"  Stage {stage:<2} {stage_data['count']:<8} "
              f"{stage_data['coverage']:.1%}     "
              f"{stage_data['accuracy']:.4f}     "
              f"{stage_data['f1_score']:.4f}")
    
    # Per-class performance
    print("\n📁 PER-CLASS PERFORMANCE")
    print("-"*40)
    for class_name, class_data in metrics['per_class'].items():
        print(f"  {class_name.upper()}: {class_data['correct']}/{class_data['total']} "
              f"({class_data['accuracy']:.2%})")
    
    # Confidence analysis
    print("\n🎯 CONFIDENCE ANALYSIS")
    print("-"*40)
    ca = metrics['confidence_analysis']
    print(f"  Mean confidence (overall):   {ca['mean_overall']:.4f}")
    print(f"  Mean confidence (correct):   {ca['mean_correct']:.4f}")
    print(f"  Mean confidence (incorrect): {ca['mean_incorrect']:.4f}")
    
    # Error analysis
    print("\n❌ ERROR ANALYSIS")
    print("-"*40)
    ea = metrics['error_analysis']
    print(f"  Total errors:      {ea['total_errors']} ({ea['error_rate']:.2%})")
    print(f"  False positives:   {ea['false_positives']} (real predicted as fake)")
    print(f"  False negatives:   {ea['false_negatives']} (fake predicted as real)")
    
    print("\n" + "="*70)
    
    return metrics


def compare_configurations(
    results_list: List[Tuple[str, List[PipelineResult]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple pipeline configurations.
    
    Args:
        results_list: List of (config_name, results) tuples
    
    Returns:
        Dictionary mapping config names to metrics
    """
    comparison = {}
    
    print("\n" + "="*70)
    print(" CONFIGURATION COMPARISON")
    print("="*70)
    
    print(f"\n{'Config':<25} {'Accuracy':<12} {'F1':<12} {'API Usage':<12}")
    print("-"*61)
    
    for config_name, results in results_list:
        metrics = evaluate_pipeline(results)
        
        # Calculate API usage
        api_calls = sum(1 for r in results if r.stage_used >= 2)
        api_rate = api_calls / len(results) if results else 0
        
        print(f"{config_name:<25} {metrics['accuracy']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {api_rate:<12.2%}")
        
        comparison[config_name] = metrics
    
    # Find best configuration
    best_config = max(comparison.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best configuration: {best_config[0]} (Accuracy: {best_config[1]['accuracy']:.4f})")
    
    return comparison


def save_results(
    results: List[PipelineResult], 
    metrics: Dict[str, Any],
    filepath: str
):
    """
    Save results and metrics to JSON file.
    
    Args:
        results: List of PipelineResult objects
        metrics: Evaluation metrics
        filepath: Output file path
    """
    output = {
        'metrics': metrics,
        'results': [
            {
                'video_id': r.video_id,
                'ground_truth': r.ground_truth,
                'prediction': r.prediction,
                'confidence': r.confidence,
                'stage_used': r.stage_used,
                'is_correct': r.is_correct,
                'processing_time': r.processing_time
            }
            for r in results
        ]
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")


def analyze_errors(
    results: List[PipelineResult],
    n_samples: int = 10
) -> List[Dict[str, Any]]:
    """
    Analyze error cases in detail.
    
    Args:
        results: List of PipelineResult objects
        n_samples: Number of error samples to analyze
    
    Returns:
        List of error analysis dictionaries
    """
    errors = [r for r in results if not r.is_correct]
    
    print(f"\n📝 DETAILED ERROR ANALYSIS (showing {min(n_samples, len(errors))} of {len(errors)})")
    print("-"*60)
    
    analysis = []
    
    for i, error in enumerate(errors[:n_samples]):
        print(f"\n[{i+1}] Video: {error.video_id}")
        print(f"    Ground Truth: {error.ground_truth}")
        print(f"    Prediction: {error.prediction} (confidence: {error.confidence:.2f})")
        print(f"    Stage Used: {error.stage_used}")
        
        if error.claim_result:
            print(f"    Initial Assessment: {error.claim_result.initial_assessment}")
            print(f"    Is Debunking: {error.claim_result.is_debunking_video}")
            if error.claim_result.red_flags:
                print(f"    Red Flags: {error.claim_result.red_flags[:2]}")
        
        analysis.append({
            'video_id': error.video_id,
            'ground_truth': error.ground_truth,
            'prediction': error.prediction,
            'confidence': error.confidence,
            'stage_used': error.stage_used,
            'initial_assessment': error.claim_result.initial_assessment if error.claim_result else None
        })
    
    return analysis

