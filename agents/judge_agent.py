"""Final Judge Agent for making authenticity decisions."""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLM_BACKEND, MODEL_NAME, CLAUDE_MODEL
from utils.llm_client import UnifiedLLMClient
from utils.prompts import JUDGE_SYSTEM, JUDGE_USER
from utils.logger import LLMDecisionLogger, get_llm_logger
from agents.claim_extractor import ClaimExtractionResult
from agents.verification_agent import VerificationResult


@dataclass
class JudgmentResult:
    """Final judgment for a video."""
    video_id: str
    verdict: str  # "fake" or "real"
    confidence: float
    reasoning: str
    key_evidence: List[str]
    contradictions_found: List[str]
    prediction_label: int  # 1 = fake, 0 = real
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], video_id: str = "") -> 'JudgmentResult':
        """Create from dictionary."""
        verdict = data.get('verdict', 'uncertain')
        return cls(
            video_id=video_id,
            verdict=verdict,
            confidence=data.get('confidence', 0.5),
            reasoning=data.get('reasoning', ''),
            key_evidence=data.get('key_evidence', []),
            contradictions_found=data.get('contradictions_found', []),
            prediction_label=1 if verdict == 'fake' else 0
        )


class JudgeAgent:
    """
    Final judge agent that synthesizes all evidence to make a verdict.
    Combines Stage 1 classifier confidence, Stage 2 claim analysis, 
    and Stage 3 verification results.
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
    
    def make_judgment(
        self,
        claim_result: ClaimExtractionResult,
        verification_result: Optional[VerificationResult] = None,
        classifier_prediction: Optional[Dict[str, Any]] = None
    ) -> JudgmentResult:
        """
        Make a final judgment based on all available evidence.
        
        Args:
            claim_result: Result from claim extraction
            verification_result: Optional result from web verification
            classifier_prediction: Optional Stage 1 classifier result
        
        Returns:
            JudgmentResult with final verdict
        """
        video_id = claim_result.video_id
        
        # Format claims for prompt
        claims_text = "\n".join([
            f"- [{c.claim_type}] {c.claim_text} (verifiable: {c.verifiable})"
            for c in claim_result.claims
        ]) if claim_result.claims else "No specific claims extracted"
        
        # Format red flags
        red_flags_text = "\n".join([f"- {rf}" for rf in claim_result.red_flags]) \
            if claim_result.red_flags else "None detected"
        
        # Format search results
        if verification_result and verification_result.search_results:
            search_text = f"Overall: {verification_result.overall_verification}\n"
            for sr in verification_result.search_results:
                search_text += f"\nQuery: {sr.query}\n"
                search_text += f"Status: {sr.verification_status}\n"
                search_text += f"Summary: {sr.results_summary}\n"
        else:
            search_text = "No web search performed or no results available"
        
        # Add classifier prediction if available
        if classifier_prediction:
            search_text += f"\n\nStage 1 Classifier: predicted {classifier_prediction.get('prediction', 'N/A')} "
            search_text += f"with confidence {classifier_prediction.get('confidence', 'N/A'):.2f}"
        
        self.logger.info(f"[{video_id}] Making final judgment...")
        start_time = time.time()
        
        try:
            result_json = self.llm_client.chat_json(
                system_prompt=JUDGE_SYSTEM,
                user_prompt=JUDGE_USER.format(
                    summary=claim_result.content_summary,
                    language=f"{claim_result.language_name} ({claim_result.detected_language})",
                    claims=claims_text,
                    red_flags=red_flags_text,
                    search_results=search_text,
                    is_debunking=claim_result.is_debunking_video
                )
            )
            result = JudgmentResult.from_dict(result_json, video_id)
            
            processing_time = time.time() - start_time
            
            # Log the judgment with full details
            stage = 3 if verification_result else 2
            if self.decision_logger:
                self.decision_logger.log_judgment(
                    video_id=video_id,
                    stage=stage,
                    verdict=result.verdict,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    key_evidence=result.key_evidence,
                    contradictions_found=result.contradictions_found,
                    processing_time=processing_time
                )
            
            self.logger.info(f"[{video_id}] Judgment: {result.verdict} (conf: {result.confidence:.2f}) | Time: {processing_time:.2f}s")
            self.logger.debug(f"[{video_id}] Reasoning: {result.reasoning[:200]}...")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{video_id}] Error making judgment: {e}")
            # Fallback to initial assessment
            verdict = "fake" if claim_result.initial_assessment == "likely_fake" else \
                     "real" if claim_result.initial_assessment == "likely_real" else "fake"
            
            return JudgmentResult(
                video_id=video_id,
                verdict=verdict,
                confidence=0.5,
                reasoning=f"Fallback judgment due to error: {str(e)}",
                key_evidence=[],
                contradictions_found=[],
                prediction_label=1 if verdict == "fake" else 0
            )
    
    def quick_judgment_from_claims(
        self, 
        claim_result: ClaimExtractionResult
    ) -> JudgmentResult:
        """
        Make a quick judgment without web search, based on claim analysis alone.
        Used when web search is not needed or for faster processing.
        
        Args:
            claim_result: Result from claim extraction
        
        Returns:
            JudgmentResult based on claim analysis
        """
        video_id = claim_result.video_id
        
        self.logger.info(f"[{video_id}] Making quick judgment (no web search)...")
        
        # Heuristic-based quick judgment
        evidence = []
        
        # Check for debunking video
        if claim_result.is_debunking_video:
            evidence.append("Video is debunking fake content, so topic is misinformation")
        
        # Check red flags
        if claim_result.red_flags:
            evidence.append(f"Red flags detected: {len(claim_result.red_flags)}")
        
        # Check claim types
        conspiracy_claims = [c for c in claim_result.claims if c.claim_type == "conspiracy"]
        if conspiracy_claims:
            evidence.append(f"Contains {len(conspiracy_claims)} conspiracy-related claims")
        
        # Use initial assessment
        if claim_result.initial_assessment == "likely_fake":
            verdict = "fake"
            confidence = 0.75
        elif claim_result.initial_assessment == "likely_real":
            verdict = "real"
            confidence = 0.75
        else:
            # Lean towards fake if there are red flags or conspiracy claims
            if claim_result.red_flags or conspiracy_claims or claim_result.is_debunking_video:
                verdict = "fake"
                confidence = 0.65
            else:
                verdict = "real"
                confidence = 0.55
        
        # Log the quick judgment with full details
        if self.decision_logger:
            self.decision_logger.log_judgment(
                video_id=video_id,
                stage=2,
                verdict=verdict,
                confidence=confidence,
                reasoning=f"Quick judgment based on claim analysis. Initial assessment: {claim_result.initial_assessment}. "
                         f"Red flags: {len(claim_result.red_flags)}. Conspiracy claims: {len(conspiracy_claims)}. "
                         f"Is debunking: {claim_result.is_debunking_video}.",
                key_evidence=evidence,
                contradictions_found=[],
                processing_time=0.0  # Quick judgment is fast
            )
        
        self.logger.info(f"[{video_id}] Quick judgment: {verdict} (conf: {confidence:.2f})")
        self.logger.debug(f"[{video_id}] Evidence: {evidence}")
        
        return JudgmentResult(
            video_id=video_id,
            verdict=verdict,
            confidence=confidence,
            reasoning=f"Quick judgment based on claim analysis. Initial assessment: {claim_result.initial_assessment}",
            key_evidence=evidence,
            contradictions_found=[],
            prediction_label=1 if verdict == "fake" else 0
        )
    
    def batch_judge(
        self,
        claim_results: List[ClaimExtractionResult],
        verification_results: Optional[List[VerificationResult]] = None,
        use_quick_judgment: bool = False,
        show_progress: bool = True
    ) -> List[JudgmentResult]:
        """
        Make judgments for multiple videos.
        
        Args:
            claim_results: List of ClaimExtractionResult objects
            verification_results: Optional list of VerificationResult objects
            use_quick_judgment: If True, skip full LLM judgment for speed
            show_progress: Whether to show progress bar
        
        Returns:
            List of JudgmentResult objects
        """
        results = []
        
        # Match verification results by video_id
        verification_map = {}
        if verification_results:
            verification_map = {v.video_id: v for v in verification_results}
        
        iterator = claim_results
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(claim_results, desc="Making judgments")
        
        for claim_result in iterator:
            if use_quick_judgment:
                result = self.quick_judgment_from_claims(claim_result)
            else:
                verification = verification_map.get(claim_result.video_id)
                result = self.make_judgment(claim_result, verification)
            
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test the judge agent
    from data.loaders import load_dataset
    from agents.claim_extractor import ClaimExtractorAgent
    from agents.verification_agent import VerificationAgent
    
    print("Testing Judge Agent...")
    
    dataset = load_dataset("combined")
    sample = dataset.samples[0]
    
    # Extract claims
    claim_agent = ClaimExtractorAgent()
    claim_result = claim_agent.extract_claims(sample.raw_data, sample.video_id)
    
    print(f"\nVideo: {sample.video_id}")
    print(f"Ground truth: {sample.ground_truth}")
    print(f"Initial assessment: {claim_result.initial_assessment}")
    
    # Make quick judgment
    judge_agent = JudgeAgent()
    quick_result = judge_agent.quick_judgment_from_claims(claim_result)
    
    print(f"\n--- Quick Judgment ---")
    print(f"Verdict: {quick_result.verdict}")
    print(f"Confidence: {quick_result.confidence:.2f}")
    print(f"Correct: {(quick_result.verdict == sample.ground_truth)}")
    
    # Make full judgment (if needed)
    if claim_result.needs_web_search:
        verification_agent = VerificationAgent()
        verification_result = verification_agent.verify_claims(claim_result)
        
        full_result = judge_agent.make_judgment(claim_result, verification_result)
        
        print(f"\n--- Full Judgment with Verification ---")
        print(f"Verdict: {full_result.verdict}")
        print(f"Confidence: {full_result.confidence:.2f}")
        print(f"Reasoning: {full_result.reasoning[:200]}...")
        print(f"Correct: {(full_result.verdict == sample.ground_truth)}")

