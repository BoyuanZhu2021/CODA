"""
CODA: Cognitive Orchestration of Detection Agents
==================================================
End-to-end video misinformation detection pipeline.

Takes a raw video file as input, extracts multimodal features
(audio transcription, visual analysis, on-screen text), and runs
the CODA cascade pipeline to produce a fake/real verdict.

Usage:
    python main.py --video path/to/video.mp4
    python main.py --video path/to/video.mp4 --description "optional social media caption"
    python main.py --video path/to/video.mp4 --no-stage1   # skip classifier, use LLM only
"""

import os
import sys
import json
import time
import base64
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from openai import OpenAI
from config import (
    OPENAI_API_KEY, CLASSIFIER_PATH, CONFIDENCE_THRESHOLD,
    MODEL_NAME, LLM_BACKEND
)


# ---------------------------------------------------------------------------
# Video processing helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    """Verify ffmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio track from video using ffmpeg.

    Returns the path to the extracted .mp3 file.
    """
    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vn",                      # no video
            "-acodec", "libmp3lame",
            "-ar", "16000",             # 16 kHz for Whisper
            "-ac", "1",                 # mono
            "-y",                       # overwrite
            output_path,
        ],
        capture_output=True, check=True
    )
    return output_path


def transcribe_audio(audio_path: str, client: OpenAI) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    file_size = os.path.getsize(audio_path)
    if file_size < 1000:
        return "(no speech detected)"

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    return response.strip() if response else "(no speech detected)"


def sample_keyframes(video_path: str, max_frames: int = 4) -> List[str]:
    """Sample evenly-spaced keyframes from a video.

    Returns a list of base64-encoded JPEG strings.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

    frames_b64: List[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

    cap.release()
    return frames_b64


def analyze_frames(frames_b64: List[str], client: OpenAI) -> Dict[str, str]:
    """Send keyframes to GPT-4o vision for visual analysis and OCR.

    Returns dict with 'image_analysis' and 'text_extraction'.
    """
    if not frames_b64:
        return {
            "image_analysis": "(no frames available)",
            "text_extraction": "(no frames available)",
        }

    content: list = [
        {
            "type": "text",
            "text": (
                "You are analyzing keyframes from a short video. Provide TWO sections:\n\n"
                "1. **IMAGE_ANALYSIS**: Describe what you see across the frames -- "
                "the setting, objects, people, emotions, and overall narrative.\n\n"
                "2. **TEXT_EXTRACTION**: Extract ALL visible text from the frames "
                "(on-screen captions, overlays, watermarks, subtitles). "
                "If no text is visible, say 'No visible text detected.'\n\n"
                "Format your response exactly as:\n"
                "IMAGE_ANALYSIS: <your description>\n"
                "TEXT_EXTRACTION: <extracted text>"
            ),
        }
    ]

    for b64 in frames_b64[:4]:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low",
            },
        })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        max_tokens=1500,
        temperature=0.2,
    )

    text = response.choices[0].message.content or ""

    image_analysis = ""
    text_extraction = ""

    if "IMAGE_ANALYSIS:" in text and "TEXT_EXTRACTION:" in text:
        parts = text.split("TEXT_EXTRACTION:")
        image_analysis = parts[0].replace("IMAGE_ANALYSIS:", "").strip()
        text_extraction = parts[1].strip()
    else:
        image_analysis = text
        text_extraction = ""

    return {
        "image_analysis": image_analysis,
        "text_extraction": text_extraction,
    }


# ---------------------------------------------------------------------------
# Assemble video_data and run CODA pipeline
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    post_description: str = "",
    use_stage1: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Full end-to-end pipeline: video file -> CODA verdict.

    Returns a dict with verdict, confidence, reasoning, and stage info.
    """
    video_path = str(Path(video_path).resolve())
    video_name = Path(video_path).stem

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not check_ffmpeg():
        raise EnvironmentError(
            "ffmpeg is required but not found. "
            "Install it: https://ffmpeg.org/download.html"
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  CODA - Multimodal Misinformation Detection")
        print(f"{'='*60}")
        print(f"  Video: {Path(video_path).name}")
        print()

    # Step 1: Extract audio
    if verbose:
        print("[1/4] Extracting audio ...")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_path = tmp.name
    try:
        extract_audio(video_path, audio_path)
    except subprocess.CalledProcessError:
        audio_path = None

    # Step 2: Transcribe audio
    if verbose:
        print("[2/4] Transcribing audio (Whisper) ...")
    if audio_path and os.path.getsize(audio_path) > 1000:
        audio_transcription = transcribe_audio(audio_path, client)
    else:
        audio_transcription = "(no audio track)"
    if verbose:
        preview = audio_transcription[:120] + ("..." if len(audio_transcription) > 120 else "")
        print(f"       Audio: {preview}")

    # Step 3: Sample and analyze keyframes
    if verbose:
        print("[3/4] Analyzing video frames (GPT-4o vision) ...")
    try:
        frames = sample_keyframes(video_path, max_frames=4)
        vision_result = analyze_frames(frames, client)
    except Exception as e:
        if verbose:
            print(f"       Warning: frame analysis failed ({e})")
        vision_result = {
            "image_analysis": "(frame analysis failed)",
            "text_extraction": "(frame analysis failed)",
        }
    if verbose:
        preview = vision_result["image_analysis"][:120]
        print(f"       Visual: {preview}...")

    # Cleanup temp audio
    if audio_path and os.path.exists(audio_path):
        os.unlink(audio_path)

    # Assemble video_data dict matching the existing schema
    video_data = {
        "video_id": video_name,
        "filename": Path(video_path).name,
        "ground_truth": "",
        "keywords": "",
        "original_annotation": "",
        "image_analysis": vision_result["image_analysis"],
        "audio_transcription": audio_transcription,
        "text_extraction": vision_result["text_extraction"],
        "post_description": post_description,
    }

    # Step 4: Run CODA pipeline
    if verbose:
        print("[4/4] Running CODA cascade pipeline ...")

    from data.loaders import VideoSample
    from utils.text_processing import extract_text_content, detect_content_language, is_debunking_content
    from pipeline.cascade_pipeline import CascadePipeline

    sample = VideoSample(
        video_id=video_name,
        filename=Path(video_path).name,
        ground_truth="",
        combined_text=extract_text_content(video_data),
        language=detect_content_language(video_data),
        original_annotation="",
        is_debunking=is_debunking_content(video_data),
        raw_data=video_data,
    )

    classifier = None
    if use_stage1 and os.path.exists(CLASSIFIER_PATH):
        from classifiers.models import MLPClassifier
        classifier = MLPClassifier()
        classifier.load(CLASSIFIER_PATH)
        if verbose:
            print(f"       Stage 1 classifier loaded: {CLASSIFIER_PATH}")
    elif use_stage1 and verbose:
        print("       No Stage 1 classifier found -- skipping to LLM analysis")

    pipeline = CascadePipeline(
        classifier=classifier,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        use_web_search=True,
        verbose=False,
        enable_logging=False,
    )

    start = time.time()
    result = pipeline.process_sample(sample)
    elapsed = time.time() - start

    # Format output
    output = {
        "video": Path(video_path).name,
        "verdict": result.prediction,
        "confidence": round(result.confidence, 3),
        "stage_used": result.stage_used,
        "processing_time_seconds": round(elapsed, 2),
        "language": sample.language,
        "details": {},
    }

    if result.claim_result:
        output["details"]["claims_extracted"] = len(result.claim_result.claims)
        output["details"]["red_flags"] = result.claim_result.red_flags
        output["details"]["initial_assessment"] = result.claim_result.initial_assessment

    if result.judgment_result:
        output["details"]["reasoning"] = result.judgment_result.reasoning
        output["details"]["key_evidence"] = result.judgment_result.key_evidence

    if verbose:
        print()
        print(f"{'='*60}")
        print(f"  VERDICT:    {'FAKE' if result.prediction == 'fake' else 'REAL'}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Stage used: {result.stage_used}")
        print(f"  Language:   {sample.language}")
        print(f"  Time:       {elapsed:.1f}s")
        if result.judgment_result and result.judgment_result.reasoning:
            print(f"\n  Reasoning:")
            for line in result.judgment_result.reasoning.split(". "):
                if line.strip():
                    print(f"    - {line.strip()}")
        print(f"{'='*60}")

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CODA: Detect misinformation in short videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --video clip.mp4\n"
            "  python main.py --video clip.mp4 --description '#ufo #alien'\n"
            "  python main.py --video clip.mp4 --no-stage1 --output result.json\n"
        ),
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the video file (.mp4, .webm, .avi, etc.)",
    )
    parser.add_argument(
        "--description", default="",
        help="Social media post caption / description (optional)",
    )
    parser.add_argument(
        "--no-stage1", action="store_true",
        help="Skip Stage 1 classifier and go directly to LLM analysis",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save JSON result to this file path",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    result = process_video(
        video_path=args.video,
        post_description=args.description,
        use_stage1=not args.no_stage1,
        verbose=not args.quiet,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\nResult saved to: {args.output}")

    return result


if __name__ == "__main__":
    main()
