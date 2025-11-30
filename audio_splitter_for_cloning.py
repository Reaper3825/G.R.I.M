#!/usr/bin/env python3
"""
Audio Silence Detection and Splitting for Voice Cloning
Detects silence points in a long WAV file, splits it into segments,
and prepares them for Coqui XTTS v2 voice cloning.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import List, Tuple
import argparse


def detect_silence(input_file: str, silence_thresh: str = "-30dB", 
                   silence_duration: float = 0.5) -> List[Tuple[float, float]]:
    """
    Detect silence segments in audio file using ffmpeg.
    
    Args:
        input_file: Path to input WAV file
        silence_thresh: Silence threshold in dB (e.g., "-30dB")
        silence_duration: Minimum silence duration in seconds
    
    Returns:
        List of tuples (start_time, end_time) for silence segments
    """
    print(f"Detecting silence in {input_file}...")
    print(f"  Threshold: {silence_thresh}, Min duration: {silence_duration}s")
    
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-af', f'silencedetect=noise={silence_thresh}:d={silence_duration}',
        '-f', 'null',
        '-'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr
    
    silence_segments = []
    silence_start = None
    
    for line in output.split('\n'):
        if 'silencedetect' in line:
            if 'silence_start' in line:
                # Extract start time
                parts = line.split('silence_start:')
                if len(parts) > 1:
                    silence_start = float(parts[1].strip())
            elif 'silence_end' in line and silence_start is not None:
                # Extract end time
                parts = line.split('silence_end:')
                if len(parts) > 1:
                    end_part = parts[1].split('|')[0].strip()
                    silence_end = float(end_part)
                    silence_segments.append((silence_start, silence_end))
                    silence_start = None
    
    print(f"Found {len(silence_segments)} silence segments")
    return silence_segments


def split_audio_by_silence(input_file: str, silence_segments: List[Tuple[float, float]], 
                           output_dir: str, min_segment_length: float = 1.0,
                           max_segment_length: float = 30.0) -> List[str]:
    """
    Split audio file at silence points into separate files.
    
    Args:
        input_file: Path to input WAV file
        silence_segments: List of (start, end) tuples for silence
        output_dir: Directory to save output segments
        min_segment_length: Minimum segment length in seconds
        max_segment_length: Maximum segment length in seconds (for voice cloning)
    
    Returns:
        List of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total duration
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())
    
    print(f"Total audio duration: {total_duration:.2f}s")
    
    # Calculate split points (middle of silence segments)
    split_points = [0.0]
    for silence_start, silence_end in silence_segments:
        split_point = (silence_start + silence_end) / 2
        split_points.append(split_point)
    split_points.append(total_duration)
    
    # Create segments
    output_files = []
    segment_num = 1
    
    for i in range(len(split_points) - 1):
        start_time = split_points[i]
        end_time = split_points[i + 1]
        duration = end_time - start_time
        
        # Skip segments that are too short
        if duration < min_segment_length:
            print(f"  Skipping segment {segment_num} (too short: {duration:.2f}s)")
            continue
        
        # Split segments that are too long
        if duration > max_segment_length:
            print(f"  Splitting long segment {segment_num} ({duration:.2f}s)")
            num_parts = int(duration / max_segment_length) + 1
            part_duration = duration / num_parts
            
            for part in range(num_parts):
                part_start = start_time + (part * part_duration)
                part_end = start_time + ((part + 1) * part_duration)
                part_duration_actual = part_end - part_start
                
                if part_duration_actual < min_segment_length:
                    continue
                
                output_file = os.path.join(output_dir, f"segment_{segment_num:04d}.wav")
                extract_segment(input_file, output_file, part_start, part_duration_actual)
                output_files.append(output_file)
                segment_num += 1
        else:
            output_file = os.path.join(output_dir, f"segment_{segment_num:04d}.wav")
            extract_segment(input_file, output_file, start_time, duration)
            output_files.append(output_file)
            segment_num += 1
    
    print(f"\nCreated {len(output_files)} audio segments")
    return output_files


def extract_segment(input_file: str, output_file: str, start_time: float, duration: float):
    """Extract a segment from audio file."""
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start_time),
        '-t', str(duration),
        '-acodec', 'pcm_s16le',  # Standard WAV format
        '-ar', '22050',  # Resample to 22050 Hz (good for XTTS)
        '-ac', '1',  # Convert to mono
        '-y',  # Overwrite output file
        output_file
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"  Created: {os.path.basename(output_file)} ({duration:.2f}s)")


def prepare_for_xtts(segments_dir: str) -> dict:
    """
    Prepare metadata for XTTS v2 voice cloning.
    
    Args:
        segments_dir: Directory containing audio segments
    
    Returns:
        Dictionary with segment information
    """
    segments = sorted([f for f in os.listdir(segments_dir) if f.endswith('.wav')])
    
    metadata = {
        'total_segments': len(segments),
        'segments': []
    }
    
    for segment_file in segments:
        segment_path = os.path.join(segments_dir, segment_file)
        
        # Get duration
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            segment_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        
        metadata['segments'].append({
            'filename': segment_file,
            'path': segment_path,
            'duration': duration
        })
    
    # Save metadata
    metadata_file = os.path.join(segments_dir, 'segments_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_file}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Split audio file by silence for voice cloning'
    )
    parser.add_argument('input_file', help='Input WAV file path')
    parser.add_argument('-o', '--output-dir', default='voice_cloning_segments',
                       help='Output directory (default: voice_cloning_segments)')
    parser.add_argument('-t', '--threshold', default='-30dB',
                       help='Silence threshold in dB (default: -30dB)')
    parser.add_argument('-d', '--duration', type=float, default=0.5,
                       help='Minimum silence duration in seconds (default: 0.5)')
    parser.add_argument('--min-length', type=float, default=1.0,
                       help='Minimum segment length in seconds (default: 1.0)')
    parser.add_argument('--max-length', type=float, default=30.0,
                       help='Maximum segment length in seconds (default: 30.0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    print("=" * 60)
    print("Audio Splitter for Voice Cloning")
    print("=" * 60)
    
    # Step 1: Detect silence
    silence_segments = detect_silence(
        args.input_file,
        silence_thresh=args.threshold,
        silence_duration=args.duration
    )
    
    if not silence_segments:
        print("Warning: No silence detected. Creating single segment.")
        # Could add logic to split by fixed duration here
    
    # Step 2: Split audio
    output_files = split_audio_by_silence(
        args.input_file,
        silence_segments,
        args.output_dir,
        min_segment_length=args.min_length,
        max_segment_length=args.max_length
    )
    
    # Step 3: Prepare metadata
    metadata = prepare_for_xtts(args.output_dir)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Total segments: {len(output_files)}")
    print(f"\nTo use with XTTS v2:")
    print(f"  1. Review the segments in: {args.output_dir}")
    print(f"  2. Use these files as speaker references for voice cloning")
    print(f"  3. Select 6-10 high-quality segments for best results")
    
    return 0


if __name__ == '__main__':
    exit(main())
