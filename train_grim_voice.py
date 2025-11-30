#!/usr/bin/env python3
"""
Train GRIM Voice with XTTS v2 using all audio segments
Integrates with existing coqui_bridge.py pipeline
"""

import os
import sys
import json
import glob
from pathlib import Path

def train_grim_voice():
    """
    Configure GRIM voice using all generated audio segments.
    This creates a multi-reference voice configuration that XTTS v2 will use.
    """
    
    # Configuration
    segments_dir = "voice_cloning_segments"
    voice_name = "grim"
    voice_output_dir = "D:/G.R.I.M/resources/voices"
    embeddings_dir = "D:/G.R.I.M/resources/voices/embeddings"
    
    print("=" * 70)
    print("GRIM Voice Training - XTTS v2 Multi-Reference Configuration")
    print("=" * 70)
    
    # Step 1: Verify segments exist
    if not os.path.exists(segments_dir):
        print(f"ERROR: Segments directory not found: {segments_dir}")
        print("Please run audio_splitter_for_cloning.py first!")
        return 1
    
    # Get all segment files
    segment_files = sorted(glob.glob(os.path.join(segments_dir, "segment_*.wav")))
    
    if not segment_files:
        print(f"ERROR: No segment files found in {segments_dir}")
        return 1
    
    print(f"\nFound {len(segment_files)} audio segments")
    
    # Step 2: Load metadata
    metadata_file = os.path.join(segments_dir, "segments_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata['total_segments']} segments")
    else:
        print("Warning: No metadata file found, proceeding without it")
    
    # Step 3: Create voice reference list file
    # XTTS v2 can use multiple reference files for better voice cloning
    reference_list_path = os.path.join(voice_output_dir, f"{voice_name}_references.txt")
    
    os.makedirs(voice_output_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Write absolute paths to all segments
    with open(reference_list_path, 'w') as f:
        for segment_file in segment_files:
            abs_path = os.path.abspath(segment_file)
            f.write(f"{abs_path}\n")
    
    print(f"\n✓ Created reference list: {reference_list_path}")
    
    # Step 4: Create primary reference (concatenate first 10 high-quality segments)
    # XTTS v2 works best with diverse 10-30 second samples
    print("\n" + "=" * 70)
    print("Creating primary voice reference...")
    print("=" * 70)
    
    primary_reference = os.path.join(voice_output_dir, f"{voice_name}_reference.wav")
    
    # Select best segments (6-12 seconds, well-distributed)
    selected_segments = []
    target_duration = 0
    max_duration = 120  # 2 minutes max for primary reference
    
    for segment_file in segment_files:
        # Get duration from filename (we know they're all good quality)
        segment_num = int(os.path.basename(segment_file).replace("segment_", "").replace(".wav", ""))
        
        # Select every 10th segment for diversity (or adjust this logic)
        if segment_num % 10 == 1 or len(selected_segments) < 12:
            selected_segments.append(segment_file)
            target_duration += 6  # Approximate
            
            if target_duration >= max_duration or len(selected_segments) >= 20:
                break
    
    print(f"Selected {len(selected_segments)} segments for primary reference")
    
    # Use ffmpeg to concatenate selected segments
    concat_list = os.path.join(segments_dir, "concat_list.txt")
    with open(concat_list, 'w') as f:
        for seg in selected_segments:
            # FFmpeg concat format
            f.write(f"file '{os.path.abspath(seg)}'\n")
    
    # Concatenate with ffmpeg
    import subprocess
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list,
        '-acodec', 'pcm_s16le',
        '-ar', '22050',
        '-ac', '1',
        '-y',
        primary_reference
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Created primary reference: {primary_reference}")
    else:
        print(f"ERROR: Failed to create primary reference")
        print(result.stderr)
        return 1
    
    # Step 5: Update coqui_bridge.py voice references
    print("\n" + "=" * 70)
    print("Updating voice configuration...")
    print("=" * 70)
    
    # Read current coqui_bridge.py
    coqui_bridge_path = "resources/python/coqui_bridge.py"
    
    with open(coqui_bridge_path, 'r') as f:
        bridge_code = f.read()
    
    # Check if GRIM voice already exists
    if f'"{voice_name}":' in bridge_code:
        print(f"✓ Voice '{voice_name}' already configured in coqui_bridge.py")
        print("  Updating reference path...")
    else:
        print(f"Adding new voice '{voice_name}' to coqui_bridge.py...")
    
    # Find VOICE_REFERENCES section and add/update GRIM voice
    voice_ref_line = f'    "{voice_name}": "{primary_reference.replace(chr(92), "/")}",  # GRIM AI voice (multi-reference)'
    
    if f'"{voice_name}":' in bridge_code:
        # Update existing entry
        import re
        pattern = rf'(\s+)"{voice_name}":\s*"[^"]*"[^,\n]*'
        bridge_code = re.sub(pattern, voice_ref_line, bridge_code)
    else:
        # Add new entry after p226
        bridge_code = bridge_code.replace(
            '"p226": "D:/G.R.I.M/resources/voices/p226_reference.wav",  # ✅ VCTK p226 male voice',
            '"p226": "D:/G.R.I.M/resources/voices/p226_reference.wav",  # ✅ VCTK p226 male voice\n' + voice_ref_line
        )
    
    # Write back
    with open(coqui_bridge_path, 'w') as f:
        f.write(bridge_code)
    
    print(f"✓ Updated coqui_bridge.py")
    
    # Step 6: Generate embedding cache (optional but recommended for speed)
    print("\n" + "=" * 70)
    print("Pre-computing voice embedding (this may take a minute)...")
    print("=" * 70)
    
    try:
        # Import after potentially installing dependencies
        import torch
        import numpy as np
        
        # Try to load TTS model and compute embedding
        try:
            from TTS.api import TTS
            
            print("Loading XTTS v2 model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            
            print(f"Computing speaker embedding from {primary_reference}...")
            
            # Compute embedding (this is what coqui_bridge.py does on first use)
            gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(
                audio_path=[primary_reference],
                gpt_cond_len=30,
                max_ref_length=60
            )
            
            # Save embedding
            embedding_path = os.path.join(embeddings_dir, f"{voice_name}.npz")
            np.savez(
                embedding_path,
                gpt_cond_latent=gpt_cond_latent.cpu().numpy(),
                speaker_embedding=speaker_embedding.cpu().numpy()
            )
            
            print(f"✓ Saved embedding cache: {embedding_path}")
            print("  (This will speed up TTS generation)")
            
        except ImportError as e:
            print(f"Skipping embedding pre-computation: TTS not available ({e})")
            print("Embedding will be computed on first use")
        except Exception as e:
            print(f"Warning: Could not pre-compute embedding: {e}")
            print("Embedding will be computed on first use")
            
    except ImportError:
        print("Skipping embedding pre-computation: torch/numpy not available")
    
    # Step 7: Success summary
    print("\n" + "=" * 70)
    print("✓ GRIM VOICE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nVoice Name: {voice_name}")
    print(f"Reference Audio: {primary_reference}")
    print(f"Total Training Samples: {len(segment_files)}")
    print(f"Primary Reference Samples: {len(selected_segments)}")
    print(f"\nTo use this voice in your C++ code:")
    print(f'  Voice::speak("Hello, I am GRIM", "{voice_name}");')
    print(f'\nOr set as default:')
    print(f'  In main.cpp, change default speaker to "{voice_name}"')
    print("\n" + "=" * 70)
    
    return 0


if __name__ == '__main__':
    try:
        exit(train_grim_voice())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
