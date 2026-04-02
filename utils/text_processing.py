"""Text processing utilities including language detection and cleaning."""

import re
from typing import Optional, Dict, Any
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for reproducible language detection
DetectorFactory.seed = 42


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common noise patterns from transcription
    noise_patterns = [
        r'No visible text detected\.?',
        r'If you need further assistance.*',
        r'If you need anything else.*',
        r'Thanks for watching\.?',
        r'Please subscribe.*',
        r'Audio transcription failed.*',
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


def detect_language(text: str) -> str:
    """
    Detect the primary language of the text.
    Returns ISO 639-1 language code (e.g., 'zh-cn', 'en', 'ro', 'vi').
    """
    if not text or len(text.strip()) < 10:
        return "unknown"
    
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "unknown"


def get_language_name(lang_code: str) -> str:
    """Convert language code to full name."""
    lang_map = {
        'zh-cn': 'Chinese',
        'zh-tw': 'Chinese',
        'en': 'English',
        'ro': 'Romanian',
        'vi': 'Vietnamese',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'ja': 'Japanese',
        'ko': 'Korean',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'unknown': 'Unknown'
    }
    return lang_map.get(lang_code, lang_code.upper())


def extract_text_content(video_data: Dict[str, Any]) -> str:
    """
    Extract and combine all text content from a video record.
    Combines: image_analysis, audio_transcription, text_extraction, post_description
    """
    parts = []
    
    # Handle image_analysis (can be string or dict)
    image_analysis = video_data.get('image_analysis', '')
    if isinstance(image_analysis, dict):
        image_analysis = image_analysis.get('description', '')
    if image_analysis:
        parts.append(f"[Visual]: {clean_text(image_analysis)}")
    
    # Handle audio_transcription (can be string or dict)
    audio_transcription = video_data.get('audio_transcription', '')
    if isinstance(audio_transcription, dict):
        audio_transcription = audio_transcription.get('transcription', '')
    if audio_transcription:
        parts.append(f"[Audio]: {clean_text(audio_transcription)}")
    
    # Handle text_extraction (usually string)
    text_extraction = video_data.get('text_extraction', '')
    if isinstance(text_extraction, str):
        parts.append(f"[Text]: {clean_text(text_extraction)}")
    
    # Handle post_description
    post_description = video_data.get('post_description', '')
    if post_description:
        parts.append(f"[Description]: {clean_text(post_description)}")
    
    # Handle keywords
    keywords = video_data.get('keywords', '')
    if keywords:
        parts.append(f"[Keywords]: {clean_text(keywords)}")
    
    return ' '.join(parts)


def detect_content_language(video_data: Dict[str, Any]) -> str:
    """
    Detect the primary language of video content.
    Prioritizes audio transcription and text extraction for language detection.
    """
    # Get text content for language detection
    audio = video_data.get('audio_transcription', '')
    if isinstance(audio, dict):
        audio = audio.get('transcription', '')
    
    text = video_data.get('text_extraction', '')
    keywords = video_data.get('keywords', '')
    
    # Combine texts for detection, prioritizing native language content
    detection_text = f"{keywords} {text} {audio}"
    
    return detect_language(detection_text)


def is_debunking_content(video_data: Dict[str, Any]) -> bool:
    """
    Check if video appears to be debunking content.
    Debunking videos discuss fake content, so they should be classified as 'fake'.
    """
    # Check original annotation for debunking indicators
    annotation = video_data.get('original_annotation', '').lower()
    if annotation in ['辟谣', 'debunk', 'debunking']:
        return True
    
    # Check for debunking keywords in content
    content = extract_text_content(video_data).lower()
    debunking_keywords = [
        '辟谣', '假的', '谣言', '不实', '虚假',  # Chinese
        'debunk', 'fake news', 'hoax', 'myth', 'false claim',  # English
        'fals', 'minciuni',  # Romanian
    ]
    
    return any(kw in content for kw in debunking_keywords)


def extract_claims_text(video_data: Dict[str, Any]) -> str:
    """
    Extract text that contains potential claims for verification.
    Focuses on audio transcription and post description.
    """
    parts = []
    
    audio = video_data.get('audio_transcription', '')
    if isinstance(audio, dict):
        audio = audio.get('transcription', '')
    if audio:
        parts.append(clean_text(audio))
    
    text_extraction = video_data.get('text_extraction', '')
    if text_extraction:
        parts.append(clean_text(text_extraction))
    
    post_desc = video_data.get('post_description', '')
    if post_desc:
        parts.append(clean_text(post_desc))
    
    return ' '.join(parts)

