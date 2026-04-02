# Dataset Setup

CODA is evaluated on two third-party short-video misinformation benchmarks. The dataset files are **not included** in this repository due to licensing.

## Datasets

### FakeSV (Chinese)

- **Paper:** Qi et al., "FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms" (AAAI 2023)
- **Repository:** https://github.com/ICTMCG/FakeSV
- **Content:** 5,495 Chinese short videos from Weibo and Douyin

### FakeTT (Multilingual)

- **Paper:** Shang et al., "Multimodal Fake News Detection on TikTok via Multi-level Message Passing" (ACM MM 2021)
- **Content:** 1,992 multilingual TikTok videos (English, Spanish, Portuguese, Vietnamese, Romanian)

## Preparing Transcription Files

After obtaining the raw datasets, you need to generate transcription JSON files that CODA expects. Each JSON file should follow this schema:

```json
{
  "dataset": "FakeTT",
  "total_videos": 1089,
  "videos": [
    {
      "video_id": "7332393503843962144",
      "filename": "7332393503843962144.mp4",
      "ground_truth": "fake",
      "keywords": "",
      "original_annotation": "fake",
      "image_analysis": "Description of what is visible in the video frames...",
      "audio_transcription": "Transcribed speech from the audio track...",
      "text_extraction": "On-screen text, captions, and overlays...",
      "post_description": "Social media caption and hashtags..."
    }
  ]
}
```

### Using CODA's built-in transcription

You can use `main.py` to transcribe individual videos. To batch-process a dataset, adapt the following pattern:

```python
from main import extract_audio, transcribe_audio, sample_keyframes, analyze_frames
from openai import OpenAI

client = OpenAI()

# For each video in your dataset:
#   1. extract_audio(video_path, audio_out_path)
#   2. transcription = transcribe_audio(audio_out_path, client)
#   3. frames = sample_keyframes(video_path)
#   4. analysis = analyze_frames(frames, client)
#   5. Assemble into the JSON schema above
```

## Expected File Placement

Place the generated JSON files here:

```
video_transcriptions/
├── README.md              (this file)
├── fakesv_1082videos.json
└── fakett_1089videos.json
```

Then update the paths in `config.py` if your filenames differ:

```python
FAKESV_PATH = "video_transcriptions/fakesv_1082videos.json"
FAKETT_PATH = "video_transcriptions/fakett_1089videos.json"
```
