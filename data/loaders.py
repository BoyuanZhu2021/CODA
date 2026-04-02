"""Data loading and preprocessing for fake video detection datasets."""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processing import (
    extract_text_content, 
    detect_content_language,
    is_debunking_content
)
from config import FAKESV_PATH, FAKETT_PATH, RANDOM_SEED, TRAIN_TEST_SPLIT


@dataclass
class VideoSample:
    """Represents a single video sample with all relevant features."""
    video_id: str
    filename: str
    ground_truth: str  # 'fake' or 'real'
    combined_text: str  # All text content combined
    language: str  # Detected language code
    original_annotation: str
    is_debunking: bool
    raw_data: Dict[str, Any]  # Original data for reference
    
    @property
    def label(self) -> int:
        """Convert ground truth to binary label (1 = fake, 0 = real)."""
        return 1 if self.ground_truth == 'fake' else 0


class FakeVideoDataset:
    """Dataset class for fake video detection."""
    
    def __init__(self, dataset_name: str = "combined"):
        """
        Initialize dataset.
        
        Args:
            dataset_name: 'fakesv', 'fakett', or 'combined'
        """
        self.dataset_name = dataset_name
        self.samples: List[VideoSample] = []
        self._load_data()
    
    def _load_data(self):
        """Load data from JSON files."""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if self.dataset_name in ['fakesv', 'combined']:
            fakesv_path = os.path.join(base_path, FAKESV_PATH)
            self._load_json(fakesv_path, 'FakeSV')
        
        if self.dataset_name in ['fakett', 'combined']:
            fakett_path = os.path.join(base_path, FAKETT_PATH)
            self._load_json(fakett_path, 'FakeTT')
        
        print(f"Loaded {len(self.samples)} samples from {self.dataset_name}")
        self._print_stats()
    
    def _load_json(self, filepath: str, source: str):
        """Load samples from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        videos = data.get('videos', [])
        
        for video in videos:
            sample = VideoSample(
                video_id=video.get('video_id', ''),
                filename=video.get('filename', ''),
                ground_truth=video.get('ground_truth', ''),
                combined_text=extract_text_content(video),
                language=detect_content_language(video),
                original_annotation=video.get('original_annotation', ''),
                is_debunking=is_debunking_content(video),
                raw_data=video
            )
            self.samples.append(sample)
    
    def _print_stats(self):
        """Print dataset statistics."""
        fake_count = sum(1 for s in self.samples if s.ground_truth == 'fake')
        real_count = sum(1 for s in self.samples if s.ground_truth == 'real')
        
        print(f"  - Fake: {fake_count} ({100*fake_count/len(self.samples):.1f}%)")
        print(f"  - Real: {real_count} ({100*real_count/len(self.samples):.1f}%)")
        
        # Language distribution
        lang_counts = {}
        for s in self.samples:
            lang_counts[s.language] = lang_counts.get(s.language, 0) + 1
        
        print("  - Languages:", dict(sorted(lang_counts.items(), key=lambda x: -x[1])[:5]))
    
    def get_texts_and_labels(self) -> Tuple[List[str], List[int]]:
        """Get all texts and labels for training."""
        texts = [s.combined_text for s in self.samples]
        labels = [s.label for s in self.samples]
        return texts, labels
    
    def get_train_test_split(
        self, 
        test_size: float = TRAIN_TEST_SPLIT,
        random_state: int = RANDOM_SEED
    ) -> Tuple[List[VideoSample], List[VideoSample]]:
        """Split dataset into train and test sets."""
        labels = [s.label for s in self.samples]
        
        train_samples, test_samples = train_test_split(
            self.samples,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        return train_samples, test_samples
    
    def get_samples_by_language(self, lang_code: str) -> List[VideoSample]:
        """Get samples filtered by language."""
        return [s for s in self.samples if s.language == lang_code]
    
    def get_sample_by_id(self, video_id: str) -> Optional[VideoSample]:
        """Get a sample by video ID."""
        for s in self.samples:
            if s.video_id == video_id:
                return s
        return None


class DataCollator:
    """Collates samples for batch processing."""
    
    def __init__(self, tokenizer=None, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, samples: List[VideoSample]) -> Dict[str, Any]:
        """Collate samples into a batch."""
        texts = [s.combined_text for s in samples]
        labels = [s.label for s in samples]
        
        if self.tokenizer:
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': np.array(labels)
            }
        
        return {
            'texts': texts,
            'labels': np.array(labels)
        }


def load_dataset(dataset_name: str = "combined") -> FakeVideoDataset:
    """
    Convenience function to load a dataset.
    
    Args:
        dataset_name: 'fakesv', 'fakett', or 'combined'
    
    Returns:
        FakeVideoDataset instance
    """
    return FakeVideoDataset(dataset_name)


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    
    # Load combined dataset
    dataset = load_dataset("combined")
    
    # Get train/test split
    train_samples, test_samples = dataset.get_train_test_split()
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Show a sample
    print("\n--- Sample Video ---")
    sample = dataset.samples[0]
    print(f"ID: {sample.video_id}")
    print(f"Ground Truth: {sample.ground_truth}")
    print(f"Language: {sample.language}")
    print(f"Is Debunking: {sample.is_debunking}")
    print(f"Text (first 500 chars): {sample.combined_text[:500]}...")

