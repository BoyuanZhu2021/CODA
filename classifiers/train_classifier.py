"""Training script for Stage 1 classifiers with model comparison."""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import load_dataset, VideoSample
from classifiers.models import EmbeddingClassifier, MLPClassifier, TransformerClassifier
from config import CONFIDENCE_THRESHOLD


def evaluate_classifier(
    y_true: List[int], 
    y_pred: np.ndarray,
    confidences: np.ndarray = None
) -> Dict[str, Any]:
    """Evaluate classifier performance."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if confidences is not None:
        metrics['mean_confidence'] = float(np.mean(confidences))
        metrics['min_confidence'] = float(np.min(confidences))
        metrics['max_confidence'] = float(np.max(confidences))
        
        # Calculate coverage at different thresholds
        for threshold in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
            high_conf_mask = confidences >= threshold
            coverage = np.mean(high_conf_mask)
            if coverage > 0:
                high_conf_acc = accuracy_score(
                    np.array(y_true)[high_conf_mask],
                    y_pred[high_conf_mask]
                )
                metrics[f'coverage_at_{threshold}'] = float(coverage)
                metrics[f'accuracy_at_{threshold}'] = float(high_conf_acc)
    
    return metrics


def train_and_evaluate_logreg(
    train_samples: List[VideoSample],
    test_samples: List[VideoSample]
) -> Tuple[EmbeddingClassifier, Dict[str, Any]]:
    """Train and evaluate Logistic Regression classifier."""
    print("\n" + "="*60)
    print("Training Embedding + Logistic Regression Classifier")
    print("="*60)
    
    train_texts = [s.combined_text for s in train_samples]
    train_labels = [s.label for s in train_samples]
    test_texts = [s.combined_text for s in test_samples]
    test_labels = [s.label for s in test_samples]
    
    start_time = time.time()
    
    classifier = EmbeddingClassifier()
    train_metrics = classifier.fit(train_texts, train_labels)
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_preds = classifier.predict(test_texts)
    test_confidences = classifier.get_confidence(test_texts)
    
    test_metrics = evaluate_classifier(test_labels, test_preds, test_confidences)
    
    results = {
        'model': 'EmbeddingClassifier (LogReg)',
        'training_time_seconds': training_time,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    print(f"\nTraining time: {training_time:.2f}s")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Mean Confidence: {test_metrics['mean_confidence']:.4f}")
    
    return classifier, results


def train_and_evaluate_mlp(
    train_samples: List[VideoSample],
    test_samples: List[VideoSample],
    val_samples: List[VideoSample] = None
) -> Tuple[MLPClassifier, Dict[str, Any]]:
    """Train and evaluate MLP classifier."""
    print("\n" + "="*60)
    print("Training Embedding + MLP Classifier")
    print("="*60)
    
    train_texts = [s.combined_text for s in train_samples]
    train_labels = [s.label for s in train_samples]
    test_texts = [s.combined_text for s in test_samples]
    test_labels = [s.label for s in test_samples]
    
    val_texts = [s.combined_text for s in val_samples] if val_samples else None
    val_labels = [s.label for s in val_samples] if val_samples else None
    
    start_time = time.time()
    
    classifier = MLPClassifier()  # Uses MLP_EPOCHS from config
    history = classifier.fit(train_texts, train_labels, val_texts, val_labels)
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_preds = classifier.predict(test_texts)
    test_confidences = classifier.get_confidence(test_texts)
    
    test_metrics = evaluate_classifier(test_labels, test_preds, test_confidences)
    
    results = {
        'model': 'MLPClassifier',
        'training_time_seconds': training_time,
        'training_history': {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_acc': history.get('val_acc', [])
        },
        'test_metrics': test_metrics
    }
    
    print(f"\nTraining time: {training_time:.2f}s")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Mean Confidence: {test_metrics['mean_confidence']:.4f}")
    
    return classifier, results


def train_and_evaluate_transformer(
    train_samples: List[VideoSample],
    test_samples: List[VideoSample],
    val_samples: List[VideoSample] = None,
    model_name: str = "xlm-roberta-base"
) -> Tuple[TransformerClassifier, Dict[str, Any]]:
    """Train and evaluate fine-tuned transformer classifier."""
    print("\n" + "="*60)
    print(f"Training Fine-tuned Transformer ({model_name})")
    print("="*60)
    
    train_texts = [s.combined_text for s in train_samples]
    train_labels = [s.label for s in train_samples]
    test_texts = [s.combined_text for s in test_samples]
    test_labels = [s.label for s in test_samples]
    
    val_texts = [s.combined_text for s in val_samples] if val_samples else None
    val_labels = [s.label for s in val_samples] if val_samples else None
    
    start_time = time.time()
    
    classifier = TransformerClassifier(model_name=model_name)  # Uses TRANSFORMER_EPOCHS from config
    history = classifier.fit(train_texts, train_labels, val_texts, val_labels)
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_preds = classifier.predict(test_texts)
    test_confidences = classifier.get_confidence(test_texts)
    
    test_metrics = evaluate_classifier(test_labels, test_preds, test_confidences)
    
    results = {
        'model': f'TransformerClassifier ({model_name})',
        'training_time_seconds': training_time,
        'training_history': {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_acc': history.get('val_acc', [])
        },
        'test_metrics': test_metrics
    }
    
    print(f"\nTraining time: {training_time:.2f}s")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Mean Confidence: {test_metrics['mean_confidence']:.4f}")
    
    return classifier, results


def run_classifier_comparison(
    dataset_name: str = "combined",
    include_transformer: bool = True,
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Run comparison of all classifier architectures.
    
    Args:
        dataset_name: 'fakesv', 'fakett', or 'combined'
        include_transformer: Whether to include fine-tuned transformer (slower)
        save_models: Whether to save trained models
    
    Returns:
        Dictionary with all results
    """
    print("\n" + "#"*60)
    print(f"# CLASSIFIER COMPARISON - Dataset: {dataset_name}")
    print("#"*60)
    
    # Load data
    dataset = load_dataset(dataset_name)
    train_samples, test_samples = dataset.get_train_test_split()
    
    # Further split training into train/val for early stopping
    from sklearn.model_selection import train_test_split
    train_samples, val_samples = train_test_split(
        train_samples, 
        test_size=0.1, 
        random_state=42,
        stratify=[s.label for s in train_samples]
    )
    
    print(f"\nData splits:")
    print(f"  Training: {len(train_samples)}")
    print(f"  Validation: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")
    
    all_results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'data_splits': {
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples)
        },
        'models': {}
    }
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Train and evaluate each model
    # 1. Logistic Regression (baseline)
    logreg_clf, logreg_results = train_and_evaluate_logreg(train_samples, test_samples)
    all_results['models']['logreg'] = logreg_results
    
    if save_models:
        logreg_clf.save(os.path.join(models_dir, 'logreg_classifier.pkl'))
    
    # 2. MLP
    mlp_clf, mlp_results = train_and_evaluate_mlp(
        train_samples, test_samples, val_samples
    )
    all_results['models']['mlp'] = mlp_results
    
    if save_models:
        mlp_clf.save(os.path.join(models_dir, 'mlp_classifier.pt'))
    
    # 3. Fine-tuned Transformer (optional - slower)
    if include_transformer:
        transformer_clf, transformer_results = train_and_evaluate_transformer(
            train_samples, test_samples, val_samples,
            model_name="xlm-roberta-base"
        )
        all_results['models']['transformer'] = transformer_results
        
        if save_models:
            transformer_clf.save(os.path.join(models_dir, 'transformer_classifier'))
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<40} {'Accuracy':<12} {'F1':<12} {'Time (s)':<12}")
    print("-"*76)
    
    for model_name, results in all_results['models'].items():
        acc = results['test_metrics']['accuracy']
        f1 = results['test_metrics']['f1_score']
        train_time = results['training_time_seconds']
        print(f"{results['model']:<40} {acc:<12.4f} {f1:<12.4f} {train_time:<12.2f}")
    
    # Find best model based on accuracy
    best_model = max(
        all_results['models'].items(),
        key=lambda x: x[1]['test_metrics']['accuracy']
    )
    all_results['best_model'] = best_model[0]
    
    print(f"\nBest model: {best_model[1]['model']}")
    print(f"Best accuracy: {best_model[1]['test_metrics']['accuracy']:.4f}")
    
    # Coverage analysis at confidence threshold
    print(f"\n--- Coverage Analysis at confidence >= {CONFIDENCE_THRESHOLD} ---")
    for model_name, results in all_results['models'].items():
        threshold_key = f'coverage_at_{CONFIDENCE_THRESHOLD}'
        acc_key = f'accuracy_at_{CONFIDENCE_THRESHOLD}'
        
        if threshold_key in results['test_metrics']:
            coverage = results['test_metrics'][threshold_key]
            acc = results['test_metrics'][acc_key]
            print(f"{results['model']}: Coverage={coverage:.2%}, Accuracy={acc:.4f}")
    
    # Save results
    results_path = os.path.join(models_dir, 'classifier_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and compare classifiers")
    parser.add_argument('--dataset', type=str, default='combined',
                        choices=['fakesv', 'fakett', 'combined'],
                        help='Dataset to use')
    parser.add_argument('--no-transformer', action='store_true',
                        help='Skip transformer training (faster)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save models')
    
    args = parser.parse_args()
    
    results = run_classifier_comparison(
        dataset_name=args.dataset,
        include_transformer=not args.no_transformer,
        save_models=not args.no_save
    )

