"""
Logging utility for the Fake Video Detection System.

Provides structured logging for:
- Training progress and metrics
- LLM decision processes
- Pipeline flow and stage transitions
- Verification results
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

# Create logs directory
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)


def setup_logger(
    name: str = "fake_video_detector",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (auto-generated if None)
        level: Logging level
        console_output: Whether to also output to console
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOGS_DIR, f"{name}_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


# Global loggers
_main_logger = None
_training_logger = None
_llm_logger = None
_pipeline_logger = None


def get_main_logger() -> logging.Logger:
    """Get the main application logger."""
    global _main_logger
    if _main_logger is None:
        _main_logger = setup_logger("main")
    return _main_logger


def get_training_logger() -> logging.Logger:
    """Get the training-specific logger."""
    global _training_logger
    if _training_logger is None:
        _training_logger = setup_logger("training")
    return _training_logger


def get_llm_logger() -> logging.Logger:
    """Get the LLM decision logger."""
    global _llm_logger
    if _llm_logger is None:
        _llm_logger = setup_logger("llm_decisions")
    return _llm_logger


def get_pipeline_logger() -> logging.Logger:
    """Get the pipeline flow logger."""
    global _pipeline_logger
    if _pipeline_logger is None:
        _pipeline_logger = setup_logger("pipeline")
    return _pipeline_logger


@dataclass
class TrainingLog:
    """Structured log for training events."""
    model_name: str
    epoch: int
    total_epochs: int
    train_loss: float
    train_accuracy: float
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_log_message(self) -> str:
        msg = f"[{self.model_name}] Epoch {self.epoch}/{self.total_epochs} | "
        msg += f"Loss: {self.train_loss:.4f} | Train Acc: {self.train_accuracy:.4f}"
        if self.val_accuracy is not None:
            msg += f" | Val Acc: {self.val_accuracy:.4f}"
        return msg


@dataclass
class LLMDecisionLog:
    """Structured log for LLM decision events."""
    video_id: str
    stage: int  # 2 or 3
    action: str  # claim_extraction, search_query_generation, verification, judgment
    input_summary: str
    output_summary: str
    detected_language: Optional[str] = None
    claims_extracted: Optional[int] = None
    red_flags: Optional[List[str]] = None
    initial_assessment: Optional[str] = None
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    api_tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    timestamp: str = None
    # New fields for detailed logging
    full_claims: Optional[List[Dict[str, Any]]] = None  # Full claim data
    search_queries: Optional[List[str]] = None  # Search queries used
    search_results_summary: Optional[List[Dict[str, Any]]] = None  # Search results
    key_evidence: Optional[List[str]] = None  # Key evidence for judgment
    full_reasoning: Optional[str] = None  # Complete reasoning text
    content_summary: Optional[str] = None  # Video content summary
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_log_message(self) -> str:
        msg = f"[Stage {self.stage}] {self.action} | Video: {self.video_id}"
        if self.detected_language:
            msg += f" | Lang: {self.detected_language}"
        if self.verdict:
            msg += f" | Verdict: {self.verdict} (conf: {self.confidence:.2f})"
        if self.claims_extracted is not None:
            msg += f" | Claims: {self.claims_extracted}"
        return msg


@dataclass 
class PipelineLog:
    """Structured log for pipeline flow events."""
    video_id: str
    ground_truth: str
    stage_used: int
    prediction: str
    confidence: float
    is_correct: bool
    stage1_prediction: Optional[str] = None
    stage1_confidence: Optional[float] = None
    needs_web_search: Optional[bool] = None
    processing_time: float = 0.0
    timestamp: str = None
    # Runtime accuracy tracking
    running_correct: Optional[int] = None
    running_total: Optional[int] = None
    running_accuracy: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_log_message(self) -> str:
        status = "✓" if self.is_correct else "✗"
        msg = f"{status} Video: {self.video_id} | Stage: {self.stage_used} | "
        msg += f"Pred: {self.prediction} | Truth: {self.ground_truth} | "
        msg += f"Conf: {self.confidence:.2f} | Time: {self.processing_time:.2f}s"
        if self.running_accuracy is not None:
            msg += f" | Acc: {self.running_accuracy:.1%}"
        return msg


class TrainingLogger:
    """Logger specifically for training progress."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = get_training_logger()
        self.history = []
        
        self.logger.info(f"=" * 60)
        self.logger.info(f"Starting training for model: {model_name}")
        self.logger.info(f"=" * 60)
    
    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_accuracy: float,
        val_accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None
    ):
        """Log a training epoch."""
        log_entry = TrainingLog(
            model_name=self.model_name,
            epoch=epoch,
            total_epochs=total_epochs,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate
        )
        
        self.history.append(asdict(log_entry))
        self.logger.info(log_entry.to_log_message())
    
    def log_final_metrics(self, metrics: Dict[str, float]):
        """Log final training metrics."""
        self.logger.info("-" * 40)
        self.logger.info(f"Training Complete for {self.model_name}")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        self.logger.info("-" * 40)
    
    def save_history(self, filepath: Optional[str] = None):
        """Save training history to JSON."""
        if filepath is None:
            filepath = os.path.join(
                LOGS_DIR, 
                f"training_history_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"Training history saved to: {filepath}")


class LLMDecisionLogger:
    """Logger for LLM decision processes with detailed response logging."""
    
    def __init__(self):
        self.logger = get_llm_logger()
        self.decisions = []
        # Create a detailed log file for full LLM responses
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.detailed_log_path = os.path.join(LOGS_DIR, f"llm_detailed_{timestamp}.json")
        self.detailed_logs = []
    
    def log_claim_extraction(
        self,
        video_id: str,
        detected_language: str,
        claims_count: int,
        red_flags: List[str],
        initial_assessment: str,
        is_debunking: bool,
        processing_time: float,
        # New detailed parameters
        claims_data: Optional[List[Dict[str, Any]]] = None,
        content_summary: Optional[str] = None,
        language_name: Optional[str] = None
    ):
        """Log claim extraction results with full claim details."""
        log_entry = LLMDecisionLog(
            video_id=video_id,
            stage=2,
            action="claim_extraction",
            input_summary=f"Video content analysis",
            output_summary=f"Extracted {claims_count} claims, {len(red_flags)} red flags",
            detected_language=detected_language,
            claims_extracted=claims_count,
            red_flags=red_flags,
            initial_assessment=initial_assessment,
            processing_time=processing_time,
            full_claims=claims_data,
            content_summary=content_summary
        )
        
        self.decisions.append(asdict(log_entry))
        self.logger.info(log_entry.to_log_message())
        
        # Log details at info level for visibility
        self.logger.info(f"  Language: {language_name or detected_language}")
        self.logger.info(f"  Initial Assessment: {initial_assessment}")
        self.logger.info(f"  Is Debunking: {is_debunking}")
        self.logger.info(f"  Content Summary: {content_summary[:150] if content_summary else 'N/A'}...")
        
        # Log full claims
        if claims_data:
            self.logger.info(f"  === EXTRACTED CLAIMS ({len(claims_data)}) ===")
            for i, claim in enumerate(claims_data, 1):
                claim_text = claim.get('claim_text', '')[:200]
                claim_type = claim.get('claim_type', 'unknown')
                verifiable = claim.get('verifiable', False)
                strategy = claim.get('verification_strategy', 'unknown')
                importance = claim.get('importance', 'unknown')
                self.logger.info(f"    Claim {i}: [{claim_type}] {claim_text}...")
                self.logger.info(f"             Verifiable: {verifiable} | Strategy: {strategy} | Importance: {importance}")
        
        if red_flags:
            self.logger.info(f"  === RED FLAGS ({len(red_flags)}) ===")
            for rf in red_flags:
                self.logger.info(f"    - {rf}")
        
        # Save to detailed log
        self._save_detailed_log(log_entry)
    
    def log_search_queries(
        self,
        video_id: str,
        queries: List[str],
        language: str,
        # New detailed parameters
        target_claims: Optional[List[str]] = None,
        expected_sources: Optional[List[str]] = None
    ):
        """Log search query generation with full query details."""
        log_entry = LLMDecisionLog(
            video_id=video_id,
            stage=3,
            action="search_query_generation",
            input_summary=f"Claims for verification",
            output_summary=f"Generated {len(queries)} search queries in {language}",
            detected_language=language,
            search_queries=queries
        )
        
        self.decisions.append(asdict(log_entry))
        self.logger.info(log_entry.to_log_message())
        
        # Log full queries
        self.logger.info(f"  === SEARCH QUERIES ({len(queries)}) ===")
        for i, query in enumerate(queries, 1):
            self.logger.info(f"    Query {i}: {query}")
            if target_claims and i <= len(target_claims):
                self.logger.info(f"             Target Claim: {target_claims[i-1][:100]}...")
            if expected_sources and i <= len(expected_sources):
                self.logger.info(f"             Expected Sources: {expected_sources[i-1]}")
        
        # Save to detailed log
        self._save_detailed_log(log_entry)
    
    def log_verification_result(
        self,
        video_id: str,
        overall_verification: str,
        verification_summary: str,
        # New detailed parameters
        search_results: Optional[List[Dict[str, Any]]] = None,
        processing_time: Optional[float] = None
    ):
        """Log verification results with full search result details."""
        log_entry = LLMDecisionLog(
            video_id=video_id,
            stage=3,
            action="verification",
            input_summary="Search queries executed",
            output_summary=f"Verification: {overall_verification}",
            search_results_summary=search_results,
            processing_time=processing_time
        )
        
        self.decisions.append(asdict(log_entry))
        self.logger.info(log_entry.to_log_message())
        
        # Log full search results
        if search_results:
            self.logger.info(f"  === WEB SEARCH RESULTS ({len(search_results)}) ===")
            for i, result in enumerate(search_results, 1):
                query = result.get('query', 'N/A')
                status = result.get('verification_status', 'unknown')
                summary = result.get('results_summary', 'N/A')
                sources = result.get('sources_found', [])
                facts = result.get('relevant_facts', [])
                
                self.logger.info(f"    Result {i}:")
                self.logger.info(f"      Query: {query[:100]}...")
                self.logger.info(f"      Status: {status}")
                self.logger.info(f"      Summary: {summary[:200]}...")
                if sources:
                    self.logger.info(f"      Sources: {', '.join(sources[:5])}")
                if facts:
                    self.logger.info(f"      Key Facts:")
                    for fact in facts[:3]:
                        self.logger.info(f"        - {fact[:150]}...")
        
        self.logger.info(f"  === OVERALL VERIFICATION: {overall_verification.upper()} ===")
        self.logger.info(f"  Summary: {verification_summary[:300]}...")
        
        # Save to detailed log
        self._save_detailed_log(log_entry)
    
    def log_judgment(
        self,
        video_id: str,
        stage: int,
        verdict: str,
        confidence: float,
        reasoning: str,
        key_evidence: List[str],
        # New detailed parameters
        contradictions_found: Optional[List[str]] = None,
        processing_time: Optional[float] = None
    ):
        """Log final judgment with full reasoning details."""
        log_entry = LLMDecisionLog(
            video_id=video_id,
            stage=stage,
            action="judgment",
            input_summary="All evidence synthesized",
            output_summary=f"Verdict: {verdict}",
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning[:200] if reasoning else None,
            full_reasoning=reasoning,  # Store full reasoning
            key_evidence=key_evidence,
            processing_time=processing_time
        )
        
        self.decisions.append(asdict(log_entry))
        self.logger.info(log_entry.to_log_message())
        
        # Log full judgment details
        self.logger.info(f"  === FINAL JUDGMENT ===")
        self.logger.info(f"  Verdict: {verdict.upper()}")
        self.logger.info(f"  Confidence: {confidence:.2f}")
        self.logger.info(f"  Stage Used: {stage}")
        
        # Log full reasoning
        self.logger.info(f"  === REASONING ===")
        if reasoning:
            # Split reasoning into lines for better readability
            for line in reasoning.split('. '):
                if line.strip():
                    self.logger.info(f"    {line.strip()}.")
        
        # Log key evidence
        if key_evidence:
            self.logger.info(f"  === KEY EVIDENCE ({len(key_evidence)}) ===")
            for i, evidence in enumerate(key_evidence, 1):
                self.logger.info(f"    {i}. {evidence}")
        
        # Log contradictions
        if contradictions_found:
            self.logger.info(f"  === CONTRADICTIONS FOUND ({len(contradictions_found)}) ===")
            for i, contradiction in enumerate(contradictions_found, 1):
                self.logger.info(f"    {i}. {contradiction}")
        
        # Save to detailed log
        self._save_detailed_log(log_entry)
    
    def _save_detailed_log(self, log_entry: LLMDecisionLog):
        """Save a detailed log entry to the detailed log file."""
        self.detailed_logs.append(asdict(log_entry))
        
        # Write incrementally to avoid data loss
        try:
            with open(self.detailed_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.detailed_logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save detailed log: {e}")
    
    def save_decisions(self, filepath: Optional[str] = None):
        """Save all decisions to JSON."""
        if filepath is None:
            filepath = os.path.join(
                LOGS_DIR,
                f"llm_decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.decisions, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"LLM decisions saved to: {filepath}")
        self.logger.info(f"Detailed LLM logs saved to: {self.detailed_log_path}")


class PipelineLogger:
    """Logger for pipeline flow with runtime accuracy tracking."""
    
    def __init__(self):
        self.logger = get_pipeline_logger()
        self.results = []
        self.start_time = None
        # Runtime tracking
        self.running_correct = 0
        self.running_total = 0
    
    def start_batch(self, total_samples: int, dataset_name: str):
        """Log batch processing start."""
        self.start_time = datetime.now()
        # Reset runtime tracking for new batch
        self.running_correct = 0
        self.running_total = 0
        self.results = []
        
        self.logger.info("=" * 60)
        self.logger.info(f"Starting Pipeline Processing")
        self.logger.info(f"  Dataset: {dataset_name}")
        self.logger.info(f"  Total Samples: {total_samples}")
        self.logger.info("=" * 60)
    
    def log_sample(
        self,
        video_id: str,
        ground_truth: str,
        stage_used: int,
        prediction: str,
        confidence: float,
        processing_time: float,
        stage1_prediction: Optional[str] = None,
        stage1_confidence: Optional[float] = None,
        needs_web_search: Optional[bool] = None
    ):
        """Log a single sample result with runtime accuracy."""
        is_correct = prediction == ground_truth
        
        # Update runtime tracking
        self.running_total += 1
        if is_correct:
            self.running_correct += 1
        running_accuracy = self.running_correct / self.running_total
        
        log_entry = PipelineLog(
            video_id=video_id,
            ground_truth=ground_truth,
            stage_used=stage_used,
            prediction=prediction,
            confidence=confidence,
            is_correct=is_correct,
            stage1_prediction=stage1_prediction,
            stage1_confidence=stage1_confidence,
            needs_web_search=needs_web_search,
            processing_time=processing_time,
            running_correct=self.running_correct,
            running_total=self.running_total,
            running_accuracy=running_accuracy
        )
        
        self.results.append(asdict(log_entry))
        self.logger.info(log_entry.to_log_message())
        
        # Log stage transition details at debug level
        if stage1_prediction:
            self.logger.debug(
                f"  Stage 1: {stage1_prediction} (conf: {stage1_confidence:.2f})"
            )
        if needs_web_search is not None:
            self.logger.debug(f"  Web Search Needed: {needs_web_search}")
    
    def end_batch(self):
        """Log batch processing summary."""
        if not self.results:
            return
        
        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])
        accuracy = correct / total
        
        # Stage distribution
        stage_counts = {}
        stage_correct = {}
        for r in self.results:
            s = r['stage_used']
            stage_counts[s] = stage_counts.get(s, 0) + 1
            if r['is_correct']:
                stage_correct[s] = stage_correct.get(s, 0) + 1
        
        total_time = sum(r['processing_time'] for r in self.results)
        
        self.logger.info("=" * 60)
        self.logger.info("Pipeline Processing Complete")
        self.logger.info("-" * 40)
        self.logger.info(f"  Total Samples: {total}")
        self.logger.info(f"  Correct: {correct}")
        self.logger.info(f"  Accuracy: {accuracy:.2%}")
        self.logger.info(f"  Total Time: {total_time:.1f}s")
        self.logger.info("-" * 40)
        self.logger.info("  Stage Distribution:")
        for stage in sorted(stage_counts.keys()):
            count = stage_counts[stage]
            correct_count = stage_correct.get(stage, 0)
            stage_acc = correct_count / count if count > 0 else 0
            self.logger.info(
                f"    Stage {stage}: {count} samples ({count/total:.1%}), "
                f"Accuracy: {stage_acc:.2%}"
            )
        self.logger.info("=" * 60)
    
    def save_results(self, filepath: Optional[str] = None):
        """Save pipeline results to JSON."""
        if filepath is None:
            filepath = os.path.join(
                LOGS_DIR,
                f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Pipeline results saved to: {filepath}")


# Convenience function for quick logging
def log_info(message: str, logger_name: str = "main"):
    """Quick log an info message."""
    if logger_name == "main":
        get_main_logger().info(message)
    elif logger_name == "training":
        get_training_logger().info(message)
    elif logger_name == "llm":
        get_llm_logger().info(message)
    elif logger_name == "pipeline":
        get_pipeline_logger().info(message)


def log_error(message: str, logger_name: str = "main"):
    """Quick log an error message."""
    if logger_name == "main":
        get_main_logger().error(message)
    elif logger_name == "training":
        get_training_logger().error(message)
    elif logger_name == "llm":
        get_llm_logger().error(message)
    elif logger_name == "pipeline":
        get_pipeline_logger().error(message)

