import os
import re
import ast
import torch
import string
import pandas as pd
from typing import Dict, Any, List


def _parse_options(raw_options):
    if isinstance(raw_options, list):
        return [str(x).strip() for x in raw_options]

    text = str(raw_options)
    if not text.strip():
        return []

    # Standard Python-list string
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass

    # Numpy-style array string: ['A...' 'B...' ...]
    quoted = re.findall(r"'([^']*)'", text)
    if quoted:
        return [q.strip() for q in quoted]

    # Fallback: newline split
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _load_videomme_from_tsv(tsv_path: str, duration: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(tsv_path, sep='\t')
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        item = row.to_dict()
        if str(item.get('duration', '')).strip() != duration:
            continue

        item['videoID'] = str(item.get('videoID', item.get('video_id', ''))).strip()
        item['question_id'] = str(item.get('question_id', '')).strip()
        item['question'] = str(item.get('question', '')).strip()
        item['domain'] = str(item.get('domain', '')).strip()
        item['sub_category'] = str(item.get('sub_category', '')).strip()
        item['answer'] = str(item.get('answer', '')).strip()
        item['options'] = _parse_options(item.get('options', ''))
        rows.append(item)
    return rows

def load_videomme_dataset(data_dir, duration='short'):
    """
    Load the VideoMME dataset.
    
    Args:
        data_dir: Directory containing VideoMME data
        duration: Video duration type ('short', 'medium', or 'long')
    
    Returns:
        List of data samples
    """
    print(f"Loading VideoMME dataset with duration={duration}")
    
    data_dir_raw = str(data_dir).strip()
    if not data_dir_raw:
        raise ValueError(
            "VideoMME data_dir is empty. Pass --data-dir explicitly "
            "(e.g. /media/chenxi/ISC/VIVID/VideoMME)."
        )

    data_dir = os.path.abspath(data_dir_raw)
    if os.path.basename(data_dir).lower() == 'videos':
        data_dir = os.path.dirname(data_dir)

    if not os.path.isdir(data_dir):
        raise ValueError(
            f"VideoMME data directory does not exist: {data_dir}. "
            "Pass --data-dir explicitly to a valid local dataset directory."
        )

    tsv_candidates = [
        os.path.join(data_dir, "VideoMME.tsv"),
        os.path.join(str(os.environ.get("LMUData", "")).strip(), "VideoMME.tsv"),
        "/media/chenxi/ISC/VIVID/LMUData/VideoMME.tsv",
    ]
    tsv_path = next((p for p in tsv_candidates if p and os.path.exists(p)), "")
    if not tsv_path:
        raise FileNotFoundError(
            f"No local VideoMME.tsv found. Tried: {tsv_candidates}"
        )
    total_data = _load_videomme_from_tsv(tsv_path, duration)
    
    print(f"✓ Loaded {len(total_data)} samples with duration={duration}")
    return total_data

def extract_video_frames_with_timestamps(video_path, fps=2, min_frames=4, max_frames=512):
    """
    Extract frames from video and return their timestamps.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        min_frames: Minimum number of frames
        max_frames: Maximum number of frames
    
    Returns:
        Tuple of (frame_indices, frame_timestamps)
    """
    from decord import VideoReader
    
    video_reader = VideoReader(video_path, num_threads=1)
    video_len = len(video_reader)
    duration = video_len / video_reader.get_avg_fps()
    
    # Calculate number of frames to extract
    nframes = round(duration) * fps
    nframes = min(max(nframes, min_frames), max_frames, video_len // 2 * 2)
    
    # Extract frame indices
    indices = torch.linspace(0, video_len - 1, nframes).round().long().clamp(0, video_len - 1).tolist()
    
    # Get frame timestamps
    frame_timestamps = video_reader.get_frame_timestamp(indices)[:, 0].tolist()
    
    return indices, frame_timestamps

def load_subtitles(subtitle_path, frame_timestamps):
    """
    Load subtitles and match them to video frames.
    
    Args:
        subtitle_path: Path to .srt subtitle file
        frame_timestamps: List of frame timestamps in seconds
    
    Returns:
        String of matched subtitles
    """
    import pysubs2
    
    if not os.path.exists(subtitle_path):
        return ""
    
    subs = pysubs2.load(subtitle_path, encoding='utf-8')
    subtitles = []
    
    for sub in subs:
        for frame_timestamp in frame_timestamps:
            if sub.start / 1000 < frame_timestamp and sub.end / 1000 > frame_timestamp:
                sub_text = sub.text.replace('\\N', ' ')
                if sub_text.strip():
                    subtitles.append(sub_text)
                    break
    
    return ' '.join(subtitles)

def build_videomme_prompt(data, data_dir, use_subtitle=False, fps=2, 
                          min_frames=4, max_frames=512, 
                          min_pixels=128*28*28, max_pixels=512*28*28, 
                          total_pixels=24576*28*28, sys_prompt=None):
    """
    Build VideoMME prompt (consistent with original implementation).
    
    Args:
        data: Single data sample
        data_dir: VideoMME data directory
        use_subtitle: Whether to include subtitles
        fps: Frames per second
        min_frames: Minimum frames
        max_frames: Maximum frames
        min_pixels: Minimum pixels per frame
        max_pixels: Maximum pixels per frame
        total_pixels: Total pixels across all frames
        sys_prompt: Optional system prompt
    
    Returns:
        Tuple of (messages, annotation)
    """
    video_id = data['videoID']
    duration = data['duration']
    domain = data['domain']
    sub_category = data["sub_category"]
    question = data['question']
    choices = data['options']
    answer = data['answer']
    question_id = data['question_id']
    
    video_path = os.path.join(data_dir, 'videos', f'{video_id}.mp4')
    subtitle_path = os.path.join(data_dir, 'subtitle', f'{video_id}.srt')
    
    # Build choices text
    choice_txt = '\n'.join(choices)
    
    # Build prompt
    prompt = ''
    if use_subtitle and os.path.exists(subtitle_path):
        # Extract frame timestamps
        _, frame_timestamps = extract_video_frames_with_timestamps(
            video_path, fps=fps, min_frames=min_frames, max_frames=max_frames
        )
        
        # Load and match subtitles
        subtitles = load_subtitles(subtitle_path, frame_timestamps)
        
        if subtitles:
            prompt = "This video's subtitles are listed below:\n"
            prompt += subtitles + '\n'
    
    prompt += 'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.'
    prompt += f"\nQuestion: {question}\n{choice_txt}\nThe best answer is:"
    
    # Build video content
    video_content = {
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "total_pixels": total_pixels,
        "fps": fps
    }
    
    contents = [
        video_content,
        {
            "text": prompt
        }
    ]
    
    # Build messages
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    
    messages.append({
        "role": "user",
        "content": contents
    })
    
    # Build annotation
    assert answer in ['A', 'B', 'C', 'D', 'E']
    answer_id = ord(answer) - 65
    
    annotation = {
        "question": question,
        "choices": {
            string.ascii_uppercase[i]: choice.split(".", 1)[1].strip() 
            for i, choice in enumerate(choices)
        },
        "answer": answer,
        "answer_id": answer_id,
        "video_path": video_path,
        "domain": domain,
        "sub_category": sub_category,
        "question_id": question_id
    }
    
    return messages, annotation

