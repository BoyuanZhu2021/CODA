"""Stage 3: Language-Aware Web Search Verification Agent."""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, OPENAI_WEB_SEARCH_MODEL, USE_OPENAI_WEB_SEARCH, LLM_BACKEND
from utils.llm_client import UnifiedLLMClient
from utils.prompts import SEARCH_QUERY_SYSTEM, SEARCH_QUERY_USER
from utils.logger import LLMDecisionLogger, get_llm_logger
from agents.claim_extractor import ClaimExtractionResult, ExtractedClaim


@dataclass
class SearchQuery:
    """Represents a search query for verification."""
    query: str
    target_claim: str
    expected_sources: str


@dataclass 
class SearchResult:
    """Result from a web search."""
    query: str
    results_summary: str
    sources_found: List[str] = field(default_factory=list)
    verification_status: str = "unknown"  # verified, contradicted, inconclusive
    relevant_facts: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Complete verification result for a video."""
    video_id: str
    search_queries: List[SearchQuery] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    overall_verification: str = "inconclusive"  # verified, contradicted, mixed, inconclusive
    verification_summary: str = ""
    

class VerificationAgent:
    """
    Agent for verifying claims using language-aware web search.
    Generates search queries in the original language of the content.
    Note: Web search feature requires OpenAI API (not available with Anthropic).
    """
    
    def __init__(self, model_name: str = OPENAI_WEB_SEARCH_MODEL, enable_logging: bool = True):
        # Keep OpenAI client for web search (Anthropic doesn't support web search)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        # Also create unified client for non-web-search LLM calls
        self.llm_client = UnifiedLLMClient(backend=LLM_BACKEND)
        self.use_web_search = USE_OPENAI_WEB_SEARCH
        self.enable_logging = enable_logging
        self.decision_logger = LLMDecisionLogger() if enable_logging else None
        self.logger = get_llm_logger()
    
    def generate_search_queries(
        self, 
        claim_result: ClaimExtractionResult
    ) -> List[SearchQuery]:
        """
        Generate search queries in the original language to verify claims.
        
        Args:
            claim_result: Result from claim extraction
        
        Returns:
            List of SearchQuery objects
        """
        # Get high priority verifiable claims
        claims_to_verify = claim_result.high_priority_claims
        
        if not claims_to_verify:
            # If no high priority claims, get any verifiable claims
            claims_to_verify = [c for c in claim_result.claims if c.verifiable][:3]
        
        if not claims_to_verify:
            return []
        
        # Format claims for prompt
        claims_text = "\n".join([
            f"- [{c.claim_type}] {c.claim_text}" 
            for c in claims_to_verify
        ])
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SEARCH_QUERY_SYSTEM},
                    {"role": "user", "content": SEARCH_QUERY_USER.format(
                        language=claim_result.detected_language,
                        language_name=claim_result.language_name,
                        summary=claim_result.content_summary,
                        claims=claims_text
                    )}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result_json = json.loads(response.choices[0].message.content)
            
            queries = []
            for q in result_json.get("search_queries", []):
                queries.append(SearchQuery(
                    query=q.get("query", ""),
                    target_claim=q.get("target_claim", ""),
                    expected_sources=q.get("expected_sources", "")
                ))
            
            return queries
            
        except Exception as e:
            print(f"Error generating search queries: {e}")
            return []
    
    def execute_web_search(self, query: str, language: str = "en") -> SearchResult:
        """
        Execute a web search and analyze results.
        
        Args:
            query: Search query string (in original language)
            language: Language code for context
        
        Returns:
            SearchResult with findings
        """
        if not self.use_web_search:
            return SearchResult(
                query=query,
                results_summary="Web search disabled",
                verification_status="inconclusive"
            )
        
        try:
            # Use OpenAI with web search capability
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": f"""You are a fact-checker. Search the web to verify information.
The content is in {language}. Analyze search results and determine if the claim can be verified.
Respond in JSON format with:
- results_summary: summary of what you found
- sources_found: list of source names/types
- verification_status: "verified" (claim is true), "contradicted" (claim is false), or "inconclusive"
- relevant_facts: list of relevant facts found"""
                    },
                    {
                        "role": "user",
                        "content": f"Search and verify: {query}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result_json = json.loads(response.choices[0].message.content)
            
            return SearchResult(
                query=query,
                results_summary=result_json.get("results_summary", ""),
                sources_found=result_json.get("sources_found", []),
                verification_status=result_json.get("verification_status", "inconclusive"),
                relevant_facts=result_json.get("relevant_facts", [])
            )
            
        except Exception as e:
            print(f"Error executing web search: {e}")
            return SearchResult(
                query=query,
                results_summary=f"Error: {str(e)}",
                verification_status="inconclusive"
            )
    
    def verify_claims(
        self, 
        claim_result: ClaimExtractionResult
    ) -> VerificationResult:
        """
        Verify claims from a video using web search.
        
        Args:
            claim_result: Result from claim extraction
        
        Returns:
            VerificationResult with all search results
        """
        video_id = claim_result.video_id
        start_time = time.time()
        
        self.logger.info(f"[{video_id}] Starting claim verification...")
        
        # Generate search queries in original language
        search_queries = self.generate_search_queries(claim_result)
        
        if not search_queries:
            self.logger.info(f"[{video_id}] No verifiable claims to search")
            return VerificationResult(
                video_id=video_id,
                overall_verification="inconclusive",
                verification_summary="No verifiable claims found to search"
            )
        
        # Log search queries with full details
        query_texts = [q.query for q in search_queries]
        target_claims = [q.target_claim for q in search_queries]
        expected_sources = [q.expected_sources for q in search_queries]
        
        if self.decision_logger:
            self.decision_logger.log_search_queries(
                video_id=video_id,
                queries=query_texts,
                language=claim_result.language_name,
                target_claims=target_claims,
                expected_sources=expected_sources
            )
        
        self.logger.info(f"[{video_id}] Generated {len(search_queries)} search queries in {claim_result.language_name}")
        for i, q in enumerate(search_queries, 1):
            self.logger.debug(f"[{video_id}] Query {i}: {q.query[:80]}...")
        
        # Execute searches
        search_results = []
        for query in search_queries:
            result = self.execute_web_search(
                query.query, 
                claim_result.detected_language
            )
            search_results.append(result)
            self.logger.debug(f"[{video_id}] Search result: {result.verification_status}")
        
        # Aggregate results
        verification_statuses = [r.verification_status for r in search_results]
        
        if all(s == "verified" for s in verification_statuses):
            overall = "verified"
        elif all(s == "contradicted" for s in verification_statuses):
            overall = "contradicted"
        elif "contradicted" in verification_statuses:
            overall = "mixed"
        else:
            overall = "inconclusive"
        
        # Generate summary
        summaries = [r.results_summary for r in search_results if r.results_summary]
        verification_summary = " ".join(summaries) if summaries else "No conclusive results"
        
        processing_time = time.time() - start_time
        
        # Prepare search results data for detailed logging
        search_results_data = [
            {
                'query': sr.query,
                'verification_status': sr.verification_status,
                'results_summary': sr.results_summary,
                'sources_found': sr.sources_found,
                'relevant_facts': sr.relevant_facts
            }
            for sr in search_results
        ]
        
        # Log verification result with full details
        if self.decision_logger:
            self.decision_logger.log_verification_result(
                video_id=video_id,
                overall_verification=overall,
                verification_summary=verification_summary,
                search_results=search_results_data,
                processing_time=processing_time
            )
        
        self.logger.info(f"[{video_id}] Verification complete: {overall} | Time: {processing_time:.2f}s")
        
        return VerificationResult(
            video_id=video_id,
            search_queries=search_queries,
            search_results=search_results,
            overall_verification=overall,
            verification_summary=verification_summary
        )
    
    def batch_verify(
        self,
        claim_results: List[ClaimExtractionResult],
        show_progress: bool = True
    ) -> List[VerificationResult]:
        """
        Verify claims from multiple videos.
        
        Args:
            claim_results: List of ClaimExtractionResult objects
            show_progress: Whether to show progress
        
        Returns:
            List of VerificationResult objects
        """
        results = []
        
        iterator = claim_results
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(claim_results, desc="Verifying claims")
        
        for claim_result in iterator:
            if claim_result.needs_web_search:
                result = self.verify_claims(claim_result)
            else:
                # Skip web search for claims that don't need it
                result = VerificationResult(
                    video_id=claim_result.video_id,
                    overall_verification="not_needed",
                    verification_summary="No web search needed for these claims"
                )
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test the verification agent
    from data.loaders import load_dataset
    from agents.claim_extractor import ClaimExtractorAgent
    
    print("Testing Verification Agent...")
    
    dataset = load_dataset("combined")
    sample = dataset.samples[5]  # Get a sample that might have verifiable claims
    
    # First extract claims
    claim_agent = ClaimExtractorAgent()
    claim_result = claim_agent.extract_claims(sample.raw_data, sample.video_id)
    
    print(f"\nVideo: {sample.video_id}")
    print(f"Ground truth: {sample.ground_truth}")
    print(f"Claims extracted: {len(claim_result.claims)}")
    print(f"Needs web search: {claim_result.needs_web_search}")
    
    # Verify claims
    verification_agent = VerificationAgent()
    verification_result = verification_agent.verify_claims(claim_result)
    
    print(f"\nSearch queries generated: {len(verification_result.search_queries)}")
    for q in verification_result.search_queries:
        print(f"  - {q.query}")
    
    print(f"\nOverall verification: {verification_result.overall_verification}")
    print(f"Summary: {verification_result.verification_summary[:200]}...")

