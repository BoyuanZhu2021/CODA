"""LLM prompt templates for claim extraction and verification."""

# =============================================================================
# CLAIM EXTRACTION PROMPTS (Stage 2)
# =============================================================================

CLAIM_EXTRACTION_SYSTEM = """You are an expert fact-checker analyzing video content to determine if it contains misinformation.
Your task is to:
1. Identify the primary language of the content
2. Extract specific, verifiable factual claims from the video
3. Classify each claim by type
4. Determine if web search verification would help

Be thorough but focus on the most important claims that would indicate if the video is real or fake."""

CLAIM_EXTRACTION_USER = """Analyze the following video content and extract claims that can be verified.

VIDEO CONTENT:
{video_content}

Please respond in the following JSON format:
{{
    "detected_language": "language code (e.g., zh-cn, en, ro, vi)",
    "language_name": "full language name",
    "content_summary": "brief summary of what the video is about (1-2 sentences)",
    "claims": [
        {{
            "claim_text": "the specific claim in original language",
            "claim_type": "one of: scientific, news_event, conspiracy, health, personal_anecdote, entertainment, historical",
            "verifiable": true/false,
            "verification_strategy": "web_search" or "reasoning" or "unverifiable",
            "importance": "high" or "medium" or "low"
        }}
    ],
    "red_flags": [
        "list any manipulation indicators: sensationalism, emotional appeals, clickbait, conspiracy language, etc."
    ],
    "is_debunking_video": true/false,
    "initial_assessment": "likely_fake" or "likely_real" or "uncertain"
}}

Focus on claims that could distinguish real content from fake/misleading content."""


# =============================================================================
# SEARCH QUERY GENERATION PROMPTS (Stage 3)
# =============================================================================

SEARCH_QUERY_SYSTEM = """You are a fact-checker who needs to generate effective search queries to verify claims.
IMPORTANT: Generate search queries IN THE SAME LANGUAGE as the original content.
Do NOT translate to English if the content is in another language."""

SEARCH_QUERY_USER = """Generate search queries to verify the following claims from a video.

Content Language: {language}
Content Summary: {summary}

Claims to verify:
{claims}

Generate 2-3 search queries IN {language_name} ({language}) to verify these claims.
For Chinese content, search in Chinese. For Romanian content, search in Romanian. Etc.

Respond in JSON format:
{{
    "search_queries": [
        {{
            "query": "search query in original language",
            "target_claim": "which claim this query aims to verify",
            "expected_sources": "type of sources to look for (news, scientific journals, fact-check sites, etc.)"
        }}
    ]
}}"""


# =============================================================================
# VERIFICATION JUDGMENT PROMPTS (Final Stage)
# =============================================================================

JUDGE_SYSTEM = """You are an expert fact-checker making a final determination about whether video content is real or fake/misleading.

Consider:
1. Claim verification results from web search
2. Red flags in the content (sensationalism, conspiracy language, etc.)
3. Source credibility
4. Internal consistency of claims
5. Whether the video is debunking fake content (which means the underlying topic is fake)

Your judgment should be well-reasoned and cite specific evidence."""

JUDGE_USER = """Based on all available evidence, determine if this video content is REAL or FAKE.

CONTENT SUMMARY:
{summary}

LANGUAGE: {language}

EXTRACTED CLAIMS:
{claims}

RED FLAGS DETECTED:
{red_flags}

WEB SEARCH VERIFICATION RESULTS:
{search_results}

IS DEBUNKING VIDEO: {is_debunking}

Please provide your judgment in JSON format:
{{
    "verdict": "fake" or "real",
    "confidence": 0.0 to 1.0,
    "reasoning": "detailed explanation of your judgment",
    "key_evidence": [
        "list of key pieces of evidence supporting your verdict"
    ],
    "contradictions_found": [
        "any contradictions between claims and verified facts"
    ]
}}

Remember: If this is a debunking video that discusses/exposes fake content, the verdict should be "fake" because the underlying topic being discussed is misinformation."""


# =============================================================================
# SIMPLE CLASSIFICATION PROMPT (Fallback/Direct LLM)
# =============================================================================

SIMPLE_CLASSIFICATION_SYSTEM = """You are an expert at identifying misinformation in social media videos.
Analyze the content and determine if it is real or fake/misleading.

Consider these indicators of fake content:
- Conspiracy theories (UFOs, secret government plots, flat earth, etc.)
- Debunking videos (they discuss fake content, so classify as fake)
- Sensationalist or clickbait language
- Unverified or implausible claims
- Emotional manipulation tactics
- Pseudoscience or anti-science claims

Consider these indicators of real content:
- Factual news reporting from credible sources
- Educational content with verifiable information
- Personal experiences without false claims
- Entertainment without misleading claims"""

SIMPLE_CLASSIFICATION_USER = """Classify this video content as REAL or FAKE.

VIDEO CONTENT:
{video_content}

Respond in JSON format:
{{
    "verdict": "fake" or "real",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}"""

