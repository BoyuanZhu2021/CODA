"""Stage 2: Claim Extraction Agent using LLM."""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLM_BACKEND, MODEL_NAME, CLAUDE_MODEL
from utils.llm_client import UnifiedLLMClient
from utils.prompts import (
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_EXTRACTION_USER,
    SIMPLE_CLASSIFICATION_SYSTEM,
    SIMPLE_CLASSIFICATION_USER
)
from utils.text_processing import extract_text_content, detect_content_language
from utils.logger import LLMDecisionLogger, get_llm_logger


@dataclass
class ExtractedClaim:
    """Represents an extracted claim from video content."""
    claim_text: str
    claim_type: str  # scientific, news_event, conspiracy, health, etc.
    verifiable: bool
    verification_strategy: str  # web_search, reasoning, unverifiable
    importance: str  # high, medium, low


@dataclass
class ClaimExtractionResult:
    """Result of claim extraction from a video."""
    video_id: str
    detected_language: str
    language_name: str
    content_summary: str
    claims: List[ExtractedClaim] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    is_debunking_video: bool = False
    initial_assessment: str = "uncertain"  # likely_fake, likely_real, uncertain
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def needs_web_search(self) -> bool:
        """Check if any claims need web search verification."""
        return any(
            c.verification_strategy == "web_search" and c.importance == "high"
            for c in self.claims
        )
    
    @property
    def high_priority_claims(self) -> List[ExtractedClaim]:
        """Get high priority claims that need verification."""
        return [c for c in self.claims if c.importance == "high" and c.verifiable]


class ClaimExtractorAgent:
    """
    Agent for extracting and analyzing claims from video content.
    Uses LLM to identify verifiable claims and determine verification strategy.
    """
    
    def __init__(self, model_name: str = None, backend: str = None, enable_logging: bool = True):
        # Use unified LLM client supporting OpenAI, Anthropic, and SiliconFlow
        self.backend = backend or LLM_BACKEND
        if self.backend == "anthropic":
            self.model_name = model_name or CLAUDE_MODEL
        else:
            self.model_name = model_name or MODEL_NAME
        self.llm_client = UnifiedLLMClient(backend=self.backend, model_name=self.model_name)
        self.enable_logging = enable_logging
        self.decision_logger = LLMDecisionLogger() if enable_logging else None
        self.logger = get_llm_logger()
    
    def extract_claims(
        self, 
        video_data: Dict[str, Any],
        video_id: str = ""
    ) -> ClaimExtractionResult:
        """
        Extract claims from video content.
        
        Args:
            video_data: Raw video data dictionary
            video_id: Video identifier
        
        Returns:
            ClaimExtractionResult with extracted claims and analysis
        """
        start_time = time.time()
        
        # Get combined text content
        if isinstance(video_data, dict):
            video_content = extract_text_content(video_data)
        else:
            video_content = str(video_data)
        
        self.logger.debug(f"[{video_id}] Starting claim extraction...")
        self.logger.debug(f"[{video_id}] Content length: {len(video_content)} chars")
        
        # Call LLM for claim extraction (supports OpenAI and Anthropic)
        try:
            result_json = self.llm_client.chat_json(
                system_prompt=CLAIM_EXTRACTION_SYSTEM,
                user_prompt=CLAIM_EXTRACTION_USER.format(
                    video_content=video_content[:4000]  # Limit context
                )
            )
            
            # Parse claims
            claims = []
            for claim_data in result_json.get("claims", []):
                claims.append(ExtractedClaim(
                    claim_text=claim_data.get("claim_text", ""),
                    claim_type=claim_data.get("claim_type", "unknown"),
                    verifiable=claim_data.get("verifiable", False),
                    verification_strategy=claim_data.get("verification_strategy", "unverifiable"),
                    importance=claim_data.get("importance", "low")
                ))
            
            result = ClaimExtractionResult(
                video_id=video_id,
                detected_language=result_json.get("detected_language", "unknown"),
                language_name=result_json.get("language_name", "Unknown"),
                content_summary=result_json.get("content_summary", ""),
                claims=claims,
                red_flags=result_json.get("red_flags", []),
                is_debunking_video=result_json.get("is_debunking_video", False),
                initial_assessment=result_json.get("initial_assessment", "uncertain"),
                raw_response=result_json
            )
            
            processing_time = time.time() - start_time
            
            # Prepare claims data for detailed logging
            claims_data = [
                {
                    'claim_text': c.claim_text,
                    'claim_type': c.claim_type,
                    'verifiable': c.verifiable,
                    'verification_strategy': c.verification_strategy,
                    'importance': c.importance
                }
                for c in claims
            ]
            
            # Log the extraction result with full details
            if self.decision_logger:
                self.decision_logger.log_claim_extraction(
                    video_id=video_id,
                    detected_language=result.detected_language,
                    claims_count=len(claims),
                    red_flags=result.red_flags,
                    initial_assessment=result.initial_assessment,
                    is_debunking=result.is_debunking_video,
                    processing_time=processing_time,
                    # New detailed parameters
                    claims_data=claims_data,
                    content_summary=result.content_summary,
                    language_name=result.language_name
                )
            
            self.logger.info(f"[{video_id}] Extracted {len(claims)} claims | "
                           f"Lang: {result.language_name} | "
                           f"Assessment: {result.initial_assessment} | "
                           f"Time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{video_id}] Error extracting claims: {e}")
            # Return minimal result on error
            return ClaimExtractionResult(
                video_id=video_id,
                detected_language=detect_content_language(video_data) if isinstance(video_data, dict) else "unknown",
                language_name="Unknown",
                content_summary="Error during extraction",
                initial_assessment="uncertain"
            )
    
    def simple_classify(
        self, 
        video_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simple direct classification without full claim extraction.
        Used as fallback or for quick assessment.
        
        Args:
            video_data: Raw video data dictionary
        
        Returns:
            Dictionary with verdict, confidence, and reasoning
        """
        if isinstance(video_data, dict):
            video_content = extract_text_content(video_data)
        else:
            video_content = str(video_data)
        
        try:
            result = self.llm_client.chat_json(
                system_prompt=SIMPLE_CLASSIFICATION_SYSTEM,
                user_prompt=SIMPLE_CLASSIFICATION_USER.format(
                    video_content=video_content[:4000]
                )
            )
            return {
                'verdict': result.get('verdict', 'uncertain'),
                'confidence': result.get('confidence', 0.5),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            print(f"Error in simple classification: {e}")
            return {
                'verdict': 'uncertain',
                'confidence': 0.5,
                'reasoning': f'Error: {str(e)}'
            }
    
    def batch_extract(
        self, 
        video_samples: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[ClaimExtractionResult]:
        """
        Extract claims from multiple videos.
        
        Args:
            video_samples: List of video data dictionaries
            show_progress: Whether to show progress
        
        Returns:
            List of ClaimExtractionResult objects
        """
        results = []
        
        iterator = video_samples
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(video_samples, desc="Extracting claims")
        
        for sample in iterator:
            video_id = sample.get('video_id', '') if isinstance(sample, dict) else ''
            result = self.extract_claims(sample, video_id)
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test the claim extractor
    from data.loaders import load_dataset
    
    print("Testing Claim Extractor Agent...")
    
    dataset = load_dataset("combined")
    sample = dataset.samples[0]
    
    agent = ClaimExtractorAgent()
    
    print(f"\nProcessing video: {sample.video_id}")
    print(f"Ground truth: {sample.ground_truth}")
    
    result = agent.extract_claims(sample.raw_data, sample.video_id)
    
    print(f"\nDetected language: {result.language_name} ({result.detected_language})")
    print(f"Content summary: {result.content_summary}")
    print(f"Is debunking video: {result.is_debunking_video}")
    print(f"Initial assessment: {result.initial_assessment}")
    print(f"\nRed flags: {result.red_flags}")
    print(f"\nClaims extracted: {len(result.claims)}")
    for i, claim in enumerate(result.claims[:3], 1):
        print(f"  {i}. [{claim.claim_type}] {claim.claim_text[:100]}...")
        print(f"     Verifiable: {claim.verifiable}, Strategy: {claim.verification_strategy}")

