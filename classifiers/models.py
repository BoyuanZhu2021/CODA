"""Classifier models for Stage 1: Deep Learning Classification."""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_MODEL, TRANSFORMER_MODEL, DEVICE, 
    MLP_HIDDEN_LAYERS, MLP_DROPOUT, BATCH_SIZE, RANDOM_SEED,
    MLP_EPOCHS, MLP_LEARNING_RATE, MLP_EARLY_STOPPING_PATIENCE,
    TRANSFORMER_EPOCHS, TRANSFORMER_LEARNING_RATE, TRANSFORMER_BATCH_SIZE
)

# Set random seeds for reproducibility
def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
from utils.logger import TrainingLogger, get_training_logger


# Global cache for embedding models to avoid reloading
_embedding_model_cache: Dict[str, SentenceTransformer] = {}


def get_cached_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Get a cached embedding model to avoid reloading.
    
    Args:
        model_name: Name or path of the sentence transformer model
    
    Returns:
        Cached SentenceTransformer model
    """
    global _embedding_model_cache
    
    if model_name not in _embedding_model_cache:
        print(f"Loading embedding model: {model_name}...")
        _embedding_model_cache[model_name] = SentenceTransformer(model_name, device=DEVICE)
        print(f"Embedding model loaded on {DEVICE}")
    
    return _embedding_model_cache[model_name]


def clear_embedding_cache():
    """Clear the embedding model cache to free GPU memory."""
    global _embedding_model_cache
    
    for model in _embedding_model_cache.values():
        del model
    
    _embedding_model_cache.clear()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Embedding model cache cleared")


def get_embeddings(
    texts: List[str], 
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using sentence transformers.
    Uses cached model to avoid memory issues.
    
    Args:
        texts: List of text strings
        model_name: Name of the sentence transformer model
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar
    
    Returns:
        numpy array of embeddings (n_samples, embedding_dim)
    """
    # Use cached model instead of creating new one each time
    model = get_cached_embedding_model(model_name)
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    return embeddings


class EmbeddingClassifier:
    """
    Logistic Regression classifier on top of sentence embeddings.
    Fast baseline with ~75-78% expected accuracy.
    """
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Train the classifier."""
        logger = TrainingLogger("EmbeddingClassifier_LogReg")
        
        logger.logger.info("Generating embeddings for training...")
        embeddings = get_embeddings(texts, self.embedding_model)
        
        logger.logger.info("Training Logistic Regression classifier...")
        self.classifier.fit(embeddings, labels)
        self.is_fitted = True
        
        # Training accuracy
        train_preds = self.classifier.predict(embeddings)
        train_acc = accuracy_score(labels, train_preds)
        train_f1 = f1_score(labels, train_preds)
        
        # Log final metrics
        logger.log_epoch(1, 1, 0.0, train_acc)
        logger.log_final_metrics({
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'embedding_model': self.embedding_model,
            'num_samples': len(texts)
        })
        
        return {'train_accuracy': train_acc, 'train_f1': train_f1}
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        embeddings = get_embeddings(texts, self.embedding_model, show_progress=False)
        return self.classifier.predict(embeddings)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        embeddings = get_embeddings(texts, self.embedding_model, show_progress=False)
        return self.classifier.predict_proba(embeddings)
    
    def get_confidence(self, texts: List[str]) -> np.ndarray:
        """Get confidence scores (max probability) for predictions."""
        proba = self.predict_proba(texts)
        return np.max(proba, axis=1)
    
    def save(self, filepath: str):
        """Save the classifier."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'embedding_model': self.embedding_model
            }, f)
    
    def load(self, filepath: str):
        """Load the classifier."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.embedding_model = data['embedding_model']
            self.is_fitted = True


class MLPHead(nn.Module):
    """MLP classification head."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_layers: List[int] = MLP_HIDDEN_LAYERS,
        dropout: float = MLP_DROPOUT
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPClassifier:
    """
    MLP classifier on top of frozen sentence embeddings.
    Expected accuracy: ~78-82%.
    
    Now includes:
    - Fixed random seed for reproducibility
    - Early stopping to prevent overfitting
    - More epochs for better convergence
    """
    
    def __init__(
        self, 
        embedding_model: str = EMBEDDING_MODEL,
        hidden_layers: List[int] = MLP_HIDDEN_LAYERS,
        dropout: float = MLP_DROPOUT,
        learning_rate: float = MLP_LEARNING_RATE,
        num_epochs: int = MLP_EPOCHS,
        batch_size: int = BATCH_SIZE,
        early_stopping_patience: int = MLP_EARLY_STOPPING_PATIENCE,
        random_seed: int = RANDOM_SEED
    ):
        self.embedding_model = embedding_model
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed
        
        self.encoder = None
        self.mlp = None
        self.is_fitted = False
        
        # Set seed for reproducibility
        set_seed(self.random_seed)
    
    def fit(
        self, 
        texts: List[str], 
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Train the MLP classifier with early stopping."""
        # Set seed for reproducibility
        set_seed(self.random_seed)
        
        logger = TrainingLogger("MLPClassifier")
        
        logger.logger.info("Generating embeddings for training...")
        embeddings = get_embeddings(texts, self.embedding_model)
        embedding_dim = embeddings.shape[1]
        
        logger.logger.info(f"Embedding dimension: {embedding_dim}")
        logger.logger.info(f"Hidden layers: {self.hidden_layers}")
        logger.logger.info(f"Dropout: {self.dropout}")
        logger.logger.info(f"Learning rate: {self.learning_rate}")
        logger.logger.info(f"Batch size: {self.batch_size}")
        logger.logger.info(f"Max epochs: {self.num_epochs}")
        logger.logger.info(f"Early stopping patience: {self.early_stopping_patience}")
        logger.logger.info(f"Random seed: {self.random_seed}")
        
        # Initialize MLP
        self.mlp = MLPHead(embedding_dim, self.hidden_layers, self.dropout).to(DEVICE)
        
        # Prepare data
        X = torch.FloatTensor(embeddings).to(DEVICE)
        y = torch.LongTensor(labels).to(DEVICE)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Class weights for imbalanced data
        class_counts = np.bincount(labels)
        class_weights = torch.FloatTensor(len(labels) / (2 * class_counts)).to(DEVICE)
        logger.logger.info(f"Class weights: {class_weights.cpu().numpy()}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        
        # Early stopping variables
        best_val_acc = 0
        best_model_state = None
        epochs_without_improvement = 0
        
        logger.logger.info("Starting MLP training...")
        for epoch in range(self.num_epochs):
            self.mlp.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.mlp(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            train_acc = correct / total
            avg_loss = total_loss / len(dataloader)
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            val_acc = None
            if val_texts is not None and val_labels is not None:
                val_preds = self.predict(val_texts)
                val_acc = accuracy_score(val_labels, val_preds)
                history['val_acc'].append(val_acc)
                scheduler.step(1 - val_acc)
                
                # Early stopping check
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.mlp.state_dict().copy()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            else:
                scheduler.step(avg_loss)
            
            # Log every epoch
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_epoch(
                epoch + 1, 
                self.num_epochs, 
                avg_loss, 
                train_acc, 
                val_acc,
                current_lr
            )
            
            # Early stopping
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.logger.info(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {self.early_stopping_patience} epochs)")
                break
        
        # Restore best model if we have validation data
        if best_model_state is not None:
            self.mlp.load_state_dict(best_model_state)
            logger.logger.info(f"Restored best model with validation accuracy: {best_val_acc:.4f}")
        
        self.is_fitted = True
        
        # Log final metrics
        actual_epochs = len(history['train_acc'])
        final_metrics = {
            'final_train_accuracy': history['train_acc'][-1],
            'final_train_loss': history['train_loss'][-1],
            'num_epochs': actual_epochs,
            'max_epochs': self.num_epochs,
            'num_samples': len(texts),
            'early_stopped': actual_epochs < self.num_epochs
        }
        if history['val_acc']:
            final_metrics['final_val_accuracy'] = history['val_acc'][-1]
            final_metrics['best_val_accuracy'] = max(history['val_acc'])
        
        logger.log_final_metrics(final_metrics)
        logger.save_history()
        
        return history
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        self.mlp.eval()
        embeddings = get_embeddings(texts, self.embedding_model, show_progress=False)
        X = torch.FloatTensor(embeddings).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.mlp(X)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        self.mlp.eval()
        embeddings = get_embeddings(texts, self.embedding_model, show_progress=False)
        X = torch.FloatTensor(embeddings).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.mlp(X)
            proba = torch.softmax(outputs, dim=1)
        
        return proba.cpu().numpy()
    
    def get_confidence(self, texts: List[str]) -> np.ndarray:
        """Get confidence scores for predictions."""
        proba = self.predict_proba(texts)
        return np.max(proba, axis=1)
    
    def save(self, filepath: str):
        """Save the classifier."""
        torch.save({
            'mlp_state_dict': self.mlp.state_dict(),
            'embedding_model': self.embedding_model,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout
        }, filepath)
    
    def load(self, filepath: str):
        """Load the classifier."""
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.embedding_model = checkpoint['embedding_model']
        self.hidden_layers = checkpoint['hidden_layers']
        self.dropout = checkpoint['dropout']
        
        # Get embedding dimension
        temp_model = SentenceTransformer(self.embedding_model)
        embedding_dim = temp_model.get_sentence_embedding_dimension()
        
        self.mlp = MLPHead(embedding_dim, self.hidden_layers, self.dropout).to(DEVICE)
        self.mlp.load_state_dict(checkpoint['mlp_state_dict'])
        self.is_fitted = True


class TransformerClassifier:
    """
    Fine-tuned transformer classifier for end-to-end training.
    Expected accuracy: ~82-86%.
    
    Uses XLM-RoBERTa-base which is a multilingual transformer model
    pre-trained on 100 languages including Chinese, English, Vietnamese, etc.
    
    Fine-tuning updates ALL model weights (125M parameters) to learn
    patterns specific to fake video detection.
    """
    
    def __init__(
        self,
        model_name: str = TRANSFORMER_MODEL,  # Use local path from config
        max_length: int = 512,
        learning_rate: float = TRANSFORMER_LEARNING_RATE,
        num_epochs: int = TRANSFORMER_EPOCHS,
        batch_size: int = TRANSFORMER_BATCH_SIZE,
        random_seed: int = RANDOM_SEED
    ):
        self.model_name = model_name
        self.random_seed = random_seed
        
        # Set seed for reproducibility
        set_seed(self.random_seed)
        
        print(f"TransformerClassifier initialized with model: {model_name}")
        print(f"  Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}, Seed: {random_seed}")
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.tokenizer = None
        self.model = None
        self.is_fitted = False
    
    def fit(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Fine-tune the transformer model."""
        from transformers import (
            AutoTokenizer, 
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup
        )
        
        logger = TrainingLogger(f"TransformerClassifier_{self.model_name.replace('/', '_')}")
        
        logger.logger.info(f"Loading transformer model: {self.model_name}...")
        logger.logger.info(f"Max length: {self.max_length}")
        logger.logger.info(f"Learning rate: {self.learning_rate}")
        logger.logger.info(f"Batch size: {self.batch_size}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2
        ).to(DEVICE)
        
        # Tokenize
        logger.logger.info("Tokenizing training data...")
        train_encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.LongTensor(labels)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Class weights
        class_counts = np.bincount(labels)
        class_weights = torch.FloatTensor(len(labels) / (2 * class_counts)).to(DEVICE)
        logger.logger.info(f"Class weights: {class_weights.cpu().numpy()}")
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        
        logger.logger.info("Starting transformer fine-tuning...")
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in pbar:
                input_ids, attention_mask, batch_labels = [b.to(DEVICE) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, batch_labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item()})
            
            train_acc = correct / total
            avg_loss = total_loss / len(train_loader)
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            val_acc = None
            if val_texts is not None and val_labels is not None:
                val_preds = self.predict(val_texts)
                val_acc = accuracy_score(val_labels, val_preds)
                history['val_acc'].append(val_acc)
            
            # Log epoch
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_epoch(epoch + 1, self.num_epochs, avg_loss, train_acc, val_acc, current_lr)
        
        self.is_fitted = True
        
        # Log final metrics
        final_metrics = {
            'final_train_accuracy': history['train_acc'][-1],
            'final_train_loss': history['train_loss'][-1],
            'model_name': self.model_name,
            'num_epochs': self.num_epochs,
            'num_samples': len(texts)
        }
        if history['val_acc']:
            final_metrics['final_val_accuracy'] = history['val_acc'][-1]
            final_metrics['best_val_accuracy'] = max(history['val_acc'])
        
        logger.log_final_metrics(final_metrics)
        logger.save_history()
        
        return history
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for texts."""
        self.model.eval()
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        predictions = []
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = [b.to(DEVICE) for b in batch]
                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, 1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        self.model.eval()
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        probabilities = []
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = [b.to(DEVICE) for b in batch]
                outputs = self.model(input_ids, attention_mask=attention_mask)
                proba = torch.softmax(outputs.logits, dim=1)
                probabilities.extend(proba.cpu().numpy())
        
        return np.array(probabilities)
    
    def get_confidence(self, texts: List[str]) -> np.ndarray:
        """Get confidence scores for predictions."""
        proba = self.predict_proba(texts)
        return np.max(proba, axis=1)
    
    def save(self, dirpath: str):
        """Save the model and tokenizer."""
        os.makedirs(dirpath, exist_ok=True)
        self.model.save_pretrained(dirpath)
        self.tokenizer.save_pretrained(dirpath)
    
    def load(self, dirpath: str):
        """Load the model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.tokenizer = AutoTokenizer.from_pretrained(dirpath)
        self.model = AutoModelForSequenceClassification.from_pretrained(dirpath).to(DEVICE)
        self.is_fitted = True

