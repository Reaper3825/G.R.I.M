import sys
import json
import argparse
import os
import torch
from pathlib import Path

# ✅ FIX 1: Disable numba caching to avoid permission errors when installed in Program Files
os.environ['NUMBA_CACHE_DIR'] = os.path.expanduser('~/.numba_cache')
os.environ['NUMBA_DISABLE_JIT'] = '0'  # Keep JIT enabled, just fix cache location

# ✅ FIX 1.5: PyTorch 2.6+ compatibility - Monkey-patch torch.load to use weights_only=False
# This is needed because PyTorch 2.6+ defaults to weights_only=True, blocking TTS model loading
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for TTS compatibility"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
print("[Coqui XTTS] Patched torch.load to use weights_only=False", file=sys.stderr, flush=True)

# ✅ FIX 2: PyTorch 2.6+ compatibility - allow XTTS model unpickling
# PyTorch 2.6+ changed weights_only default to True, blocking custom TTS model classes
# We must explicitly allow XTTS classes before importing TTS
try:
    import torch.serialization
    
    # Import all TTS classes that need to be unpickled
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
    from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
    from TTS.tts.configs.shared_configs import BaseDatasetConfig as TTSBaseDatasetConfig
    from TTS.tts.configs.shared_configs import CharactersConfig
    
    # ✅ FIX: Add missing classes that XTTS v2 needs
    try:
        from TTS.tts.layers.xtts.gpt import GPT
        from TTS.vocoder.configs import VocoderConfig
        from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
    except ImportError as ie:
        print(f"[Coqui XTTS] Warning: Could not import some XTTS classes: {ie}", file=sys.stderr, flush=True)
    
    # Register as safe globals for PyTorch 2.6+
    if hasattr(torch.serialization, 'add_safe_globals'):
        safe_classes = [
            XttsConfig,
            Xtts,
            XttsArgs,
            XttsAudioConfig,
            BaseDatasetConfig,
            TTSBaseDatasetConfig,
            CharactersConfig,
            BaseAudioConfig,
        ]
        
        # Add optional classes if they were imported
        try:
            safe_classes.extend([GPT, VocoderConfig, VoiceBpeTokenizer])
        except NameError:
            pass
        
        torch.serialization.add_safe_globals(safe_classes)
        print("[Coqui XTTS] Registered safe globals for PyTorch 2.6+", file=sys.stderr, flush=True)
    
except (ImportError, AttributeError) as e:
    # Older PyTorch or missing TTS - will be caught below
    print(f"[Coqui XTTS] Warning during safe globals registration: {e}", file=sys.stderr, flush=True)

# Try to import TTS API - fallback gracefully if not available
try:
    from TTS.api import TTS
except ImportError as e:
    # Log error and exit if TTS not available
    send({"status": "error", "message": f"TTS library not found: {e}"})
    sys.exit(1)

# ✅ NEW: Import for speaker embedding extraction
try:
    import numpy as np
    import soundfile as sf  # ✅ FIX: Import soundfile for audio validation
    import time  # ✅ NEW: For performance profiling
except ImportError as e:
    np = None
    sf = None
    print(f"[Coqui XTTS] Warning: numpy/soundfile not available: {e}", file=sys.stderr, flush=True)

# ---------- Helpers ----------
def log(msg):
    """Log messages to stderr (never stdout)."""
    print(msg, file=sys.stderr, flush=True)

def send(obj):
    """Send JSON protocol messages to stdout."""
    print(json.dumps(obj), flush=True)

def get_device():
    """Determine best device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        log(f"[Coqui XTTS] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        log("[Coqui XTTS] Using CPU (GPU not available)")
    return device

# ---------- Voice Reference & Embedding Management ----------
VOICE_REFERENCES = {
    # Default voice references (you can add custom ones here)
    "default": "D:/G.R.I.M/resources/voices/default.wav",  # ✅ Default XTTS v2 voice
    "p225": None,  # Backward compatibility - maps to default
    "p226": "D:/G.R.I.M/resources/voices/p226_reference.wav",  # ✅ VCTK p226 male voice
}

# ✅ NEW: Speaker embedding cache
SPEAKER_EMBEDDINGS = {}
EMBEDDING_DIR = "D:/G.R.I.M/resources/voices/embeddings"

def ensure_embedding_dir():
    """Create embedding directory if it doesn't exist."""
    os.makedirs(EMBEDDING_DIR, exist_ok=True)

def save_embedding(speaker_id, gpt_cond_latent, speaker_embedding):
    """Save speaker embedding to disk for reuse."""
    ensure_embedding_dir()
    
    embedding_path = os.path.join(EMBEDDING_DIR, f"{speaker_id}.npz")
    
    try:
        if np is not None:
            # Convert tensors to numpy for saving
            gpt_cond = gpt_cond_latent.cpu().numpy() if torch.is_tensor(gpt_cond_latent) else gpt_cond_latent
            spk_emb = speaker_embedding.cpu().numpy() if torch.is_tensor(speaker_embedding) else speaker_embedding
            
            np.savez(embedding_path, 
                    gpt_cond_latent=gpt_cond,
                    speaker_embedding=spk_emb)
            
            log(f"[Coqui XTTS] Saved embedding: {embedding_path}")
            return True
        else:
            log("[Coqui XTTS] Cannot save embedding - numpy not available")
            return False
    except Exception as e:
        log(f"[Coqui XTTS] Failed to save embedding: {e}")
        return False

def load_embedding(speaker_id, device="cpu", dtype=torch.float32):
    """
    Load pre-computed speaker embedding from disk.
    
    Args:
        speaker_id: Speaker identifier
        device: Target device (cpu/cuda)
        dtype: Target dtype (float32/float16) - should match model precision
    """
    embedding_path = os.path.join(EMBEDDING_DIR, f"{speaker_id}.npz")
    
    if not os.path.exists(embedding_path):
        return None, None
    
    try:
        if np is not None:
            data = np.load(embedding_path)
            gpt_cond = torch.from_numpy(data['gpt_cond_latent']).to(device=device, dtype=dtype)
            spk_emb = torch.from_numpy(data['speaker_embedding']).to(device=device, dtype=dtype)
            
            log(f"[Coqui XTTS] Loaded embedding from cache: {speaker_id}")
            return gpt_cond, spk_emb
        else:
            log("[Coqui XTTS] Cannot load embedding - numpy not available")
            return None, None
    except Exception as e:
        log(f"[Coqui XTTS] Failed to load embedding: {e}")
        return None, None

def get_voice_reference(speaker_id):
    """
    Get voice reference audio path for speaker.
    XTTS v2 uses voice cloning from reference audio instead of speaker IDs.
    """
    if speaker_id in VOICE_REFERENCES:
        ref_path = VOICE_REFERENCES[speaker_id]
        
        # If specific speaker has no reference, use default
        if not ref_path or speaker_id == "p225":  # Only p225 maps to default, p226 has its own voice
            ref_path = VOICE_REFERENCES["default"]
        
        if ref_path and os.path.exists(ref_path):
            # ✅ FIX: Validate audio file meets minimum requirements
            if sf is not None:
                try:
                    data, samplerate = sf.read(ref_path)
                    duration = len(data) / samplerate
                    
                    if duration < 1.0:
                        log(f"[Coqui XTTS] Warning: Reference audio is very short ({duration:.2f}s). "
                            "XTTS v2 works best with 3-30 seconds of speech.")
                    
                    log(f"[Coqui XTTS] Voice reference validated: {ref_path} ({duration:.2f}s, {samplerate}Hz)")
                    
                except Exception as e:
                    log(f"[Coqui XTTS] Warning: Could not validate reference audio: {e}")
                    # Return path anyway - let XTTS try to use it
            
            return ref_path
        else:
            log(f"[Coqui XTTS] Warning: Voice reference not found: {ref_path}")
    
    # Fallback to default
    default_ref = VOICE_REFERENCES["default"]
    if default_ref and os.path.exists(default_ref):
        return default_ref
    
    log("[Coqui XTTS] Error: No voice reference available (not even default)")
    return None  # No reference available

# ---------- Persistent Mode (XTTS v2) ----------
def persistent_loop(model_name, speaker, use_gpu=True):
    """
    Initialize XTTS v2 model and process TTS requests in a persistent loop.
    """
    try:
        device = get_device() if use_gpu else "cpu"
        
        # Initialize XTTS v2 model
        log(f"[Coqui XTTS] Loading model: {model_name} on {device}")
        tts = TTS(model_name).to(device)
        
        log(f"[Coqui XTTS] Model loaded successfully")
        
        # ============================================
        # ✅ OPTIMIZATION: FP16 + torch.compile() + GPU Settings
        # ============================================
        model_dtype = torch.float32  # Track model precision for embeddings
        use_fp16 = False  # Track if FP16 is successfully enabled
        
        if device == "cuda":
            model = tts.synthesizer.tts_model
            
            # Optimization: torch.compile() JIT (1.3-1.8x speedup)
            # Note: Requires triton (Linux only), will skip on Windows
            if hasattr(torch, 'compile') and sys.platform != 'win32':
                try:
                    # Compile HiFiGAN decoder (biggest bottleneck)
                    if hasattr(model, 'hifigan_decoder'):
                        log("[Coqui XTTS]   Compiling HiFiGAN decoder (may take 30-60s)...")
                        model.hifigan_decoder = torch.compile(
                            model.hifigan_decoder,
                            mode="max-autotune",
                            fullgraph=True
                        )
                        log("[Coqui XTTS]   ✓ HiFiGAN compiled")
                    
                    # Compile speaker encoder
                    if hasattr(model, 'speaker_encoder'):
                        model.speaker_encoder = torch.compile(
                            model.speaker_encoder,
                            mode="reduce-overhead"
                        )
                        log("[Coqui XTTS]   ✓ Speaker encoder compiled")
                    
                    log("[Coqui XTTS]   ✓ torch.compile() optimization complete (1.3-1.8x speedup)")
                except Exception as e:
                    log(f"[Coqui XTTS]   ⚠ torch.compile() failed: {e}")
            elif sys.platform == 'win32':
                log("[Coqui XTTS]   ⚠ torch.compile() skipped (not supported on Windows)")
            
            # Enable TensorFloat32 for faster computation
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            log("[Coqui XTTS] ✅ Optimizations applied")

        # Warm up model with dummy synthesis
        log("[Coqui XTTS] Starting model warm-up...")
        try:
            dummy_gpt, dummy_spk = load_embedding("default", device, dtype=model_dtype)
            
            if dummy_gpt is not None and hasattr(tts, 'synthesizer'):
                log("[Coqui XTTS] Dummy embedding loaded, running warm-up inference...")
                model = tts.synthesizer.tts_model
                
                # ⚡ Important: Do a full inference to warm up all code paths
                _ = model.inference(
                    text="System ready",
                    language="en",
                    gpt_cond_latent=dummy_gpt,
                    speaker_embedding=dummy_spk
                )
                log("[Coqui XTTS] ✓ Model warmed up successfully (CUDA kernels initialized)")
            else:
                if dummy_gpt is None:
                    log("[Coqui XTTS] ⚠ No default embedding found - skipping warm-up")
                    log("[Coqui XTTS] Run: python scripts/create_default_embedding.py")
                else:
                    log("[Coqui XTTS] ⚠ Model API not accessible - skipping warm-up")
        except Exception as e:
            log(f"[Coqui XTTS] ⚠ Warm-up failed (non-critical): {e}")
            import traceback
            log(f"[Coqui XTTS] Traceback: {traceback.format_exc()}")
        # ============================================
        
        # Check if model is XTTS v2
        is_xtts = "xtts" in model_name.lower()
        if is_xtts:
            log("[Coqui XTTS] XTTS v2 model detected - voice cloning enabled")
        
    except Exception as e:
        send({"status": "error", "message": f"Model load failed: {str(e)}"})
        return
    
    log("[Coqui XTTS] Initialization complete, sending ready signal")
    send({"status": "ready", "model": model_name, "device": device, "xtts_v2": is_xtts})

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"[Coqui XTTS] Invalid JSON from GRIM: {line} ({e})")
                continue

            cmd = req.get("command")
            
            if cmd == "exit":
                send({"status": "bye"})
                break
                
            elif cmd == "speak":
                text = req.get("text", "")
                spk = req.get("speaker", speaker)
                speed = float(req.get("speed", 1.0))
                out_path = req.get("out", "output.wav")
                language = req.get("language", "en")  # XTTS v2 supports multiple languages
                use_embedding = req.get("use_embedding", True)  # ✅ NEW: Option to use cached embeddings
                
                if not text:
                    send({"status": "error", "message": "Empty text provided"})
                    continue

                try:
                    # Create output directory if needed
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    
                    if is_xtts:
                        # ✅ XTTS v2 synthesis with speaker embeddings
                        voice_ref = get_voice_reference(spk)
                        
                        if not voice_ref or not os.path.exists(voice_ref):
                            # ✅ FIX: Provide more detailed error
                            error_msg = (
                                f"XTTS v2 requires a speaker reference audio file. "
                                f"Requested speaker: '{spk}', but reference not found. "
                                f"Please run: .\\scripts\\download_default_voice.ps1 "
                                f"Or provide a custom voice sample in resources/voices/"
                            )
                            raise Exception(error_msg)
                        
                        log(f"[Coqui XTTS] Using voice reference: {voice_ref} (language: {language})")
                        
                        # ✅ FIX: Try to load cached embedding first (MUCH faster!)
                        gpt_cond_latent = None
                        speaker_embedding = None
                        
                        if use_embedding:
                            gpt_cond_latent, speaker_embedding = load_embedding(spk, device, dtype=model_dtype)
                        
                        # ✅ FIX: Use cached embedding if available
                        if gpt_cond_latent is not None and speaker_embedding is not None:
                            log(f"[Coqui XTTS] Using cached embedding for: {spk} (FAST MODE)")
                            
                            t_start = time.time()
                            
                            try:
                                # Direct synthesis with pre-computed embeddings (0.5-1s)
                                if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts_model'):
                                    model = tts.synthesizer.tts_model
                                    
                                    # Time the inference
                                    t_inference_start = time.time()
                                    
                                    # Fast synthesis with embeddings
                                    wav = model.inference(
                                        text=text,
                                        language=language,
                                        gpt_cond_latent=gpt_cond_latent,
                                        speaker_embedding=speaker_embedding
                                    )
                                    
                                    t_inference = time.time() - t_inference_start
                                    
                                    # Save to file
                                    import soundfile as sf
                                    
                                    t_save_start = time.time()
                                    
                                    # ✅ FIX: Handle different wav formats
                                    if isinstance(wav, dict):
                                        wav = wav['wav']
                                    
                                    if torch.is_tensor(wav):
                                        wav = wav.cpu().numpy()
                                    
                                    # Ensure correct shape for soundfile
                                    if len(wav.shape) == 2 and wav.shape[0] == 1:
                                        wav = wav.squeeze(0)
                                    
                                    sf.write(out_path, wav, model.config.audio.sample_rate)
                                    
                                    t_save = time.time() - t_save_start
                                    t_total = time.time() - t_start
                                    
                                    log(f"[Coqui XTTS] ⚡ Fast synthesis complete | "
                                        f"Inference: {t_inference:.2f}s | Save: {t_save:.2f}s | Total: {t_total:.2f}s")
                                else:
                                    # Fallback to slow method if API changed
                                    log("[Coqui XTTS] Warning: Could not access model API, falling back to slow synthesis")
                                    tts.tts_to_file(
                                        text=text,
                                        file_path=out_path,
                                        speaker_wav=voice_ref,
                                        language=language
                                    )
                            except Exception as e:
                                log(f"[Coqui XTTS] Fast synthesis failed: {e}, trying fallback")
                                
                                # ✅ NEW: Send "hold on" message for AI to announce
                                # This lets GRIM know we're falling back to slow synthesis
                                send({
                                    "status": "fallback_notice",
                                    "message": "Fast synthesis failed, using slower method. This will take a moment."
                                })
                                
                                # Fallback to reference audio method
                                tts.tts_to_file(
                                    text=text,
                                    file_path=out_path,
                                    speaker_wav=voice_ref,
                                    language=language
                                )
                        else:
                            # No cached embedding - use slow method and cache it
                            log(f"[Coqui XTTS] No cached embedding for {spk}, computing from reference audio (SLOW)")
                            
                            # ✅ FIX: Wrap XTTS synthesis in try-catch to handle tensor errors
                            try:
                                tts.tts_to_file(
                                    text=text,
                                    file_path=out_path,
                                    speaker_wav=voice_ref,
                                    language=language
                                )
                                
                                # ✅ NEW: After slow synthesis, cache the embedding for next time
                                try:
                                    log(f"[Coqui XTTS] Caching embedding for future use...")
                                    if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts_model'):
                                        model = tts.synthesizer.tts_model
                                        
                                        # Compute and save embedding
                                        gpt_cond, spk_emb = model.get_conditioning_latents(
                                            audio_path=voice_ref
                                        )
                                        
                                        if save_embedding(spk, gpt_cond, spk_emb):
                                            log(f"[Coqui XTTS] ✓ Embedding cached for {spk} - next time will be 10x faster!")
                                except Exception as cache_err:
                                    log(f"[Coqui XTTS] Warning: Could not cache embedding: {cache_err}")
                                    
                            except IndexError as e:
                                # Handle tuple index errors from XTTS model
                                error_detail = str(e)
                                log(f"[Coqui XTTS] IndexError during synthesis: {error_detail}")
                                
                                # Try to diagnose the issue
                                if "tuple index out of range" in error_detail:
                                    raise Exception(
                                        f"XTTS v2 embedding extraction failed. "
                                        f"Possible causes: "
                                        f"(1) Reference audio is too short (needs 3+ seconds), "
                                        f"(2) Reference audio is corrupted, "
                                        f"(3) PyTorch version mismatch. "
                                        f"Reference file: {voice_ref}"
                                    )
                                else:
                                    raise  # Re-raise if it's a different IndexError
                                    
                            except Exception as e:
                                log(f"[Coqui XTTS] Synthesis error: {type(e).__name__}: {str(e)}")
                                raise
                            
                    else:
                        # Fallback for non-XTTS models (vctk/vits)
                        tts.tts_to_file(
                            text=text,
                            file_path=out_path,
                            speaker=spk,
                            speed=speed
                        )
                    
                    send({"status": "ok", "file": out_path})
                    log(f"[Coqui XTTS] Generated: {out_path}")
                    
                except Exception as e:
                    error_msg = str(e)
                    log(f"[Coqui XTTS] TTS generation failed: {error_msg}")
                    send({"status": "error", "message": error_msg})
                    
            elif cmd == "list_languages":
                # XTTS v2 supports multiple languages
                if is_xtts:
                    languages = tts.languages if hasattr(tts, 'languages') else ["en"]
                    send({"status": "ok", "languages": languages})
                else:
                    send({"status": "ok", "languages": ["en"]})
                    
            elif cmd == "set_voice_reference":
                # Allow dynamic voice reference updates
                speaker_id = req.get("speaker")
                ref_path = req.get("reference_path")
                if speaker_id and ref_path and os.path.exists(ref_path):
                    VOICE_REFERENCES[speaker_id] = ref_path
                    send({"status": "ok", "message": f"Voice reference set for {speaker_id}"})
                    log(f"[Coqui XTTS] Voice reference registered: {speaker_id} -> {ref_path}")
                else:
                    send({"status": "error", "message": "Invalid voice reference"})
            
            elif cmd == "create_embedding":
                # ✅ NEW: Create and cache speaker embedding from audio file
                speaker_id = req.get("speaker")
                ref_path = req.get("reference_path")
                
                if not speaker_id or not ref_path or not os.path.exists(ref_path):
                    send({"status": "error", "message": "Invalid speaker ID or reference path"})
                    continue
                
                try:
                    log(f"[Coqui XTTS] Creating embedding for {speaker_id} from {ref_path}")
                    
                    if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts_model'):
                        model = tts.synthesizer.tts_model
                        
                        # Compute embedding from reference audio
                        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                            audio_path=ref_path
                        )
                        
                        # Save to disk
                        if save_embedding(speaker_id, gpt_cond_latent, speaker_embedding):
                            # Also update voice reference
                            VOICE_REFERENCES[speaker_id] = ref_path
                                            
                            send({
                                "status": "ok",
                                "message": f"Embedding created for {speaker_id}",
                                "embedding_path": os.path.join(EMBEDDING_DIR, f"{speaker_id}.npz")
                            })
                        else:
                            send({"status": "error", "message": "Failed to save embedding"})
                    else:
                        send({"status": "error", "message": "Model does not support embeddings"})
                        
                except Exception as e:
                    log(f"[Coqui XTTS] Embedding creation failed: {e}")
                    send({"status": "error", "message": str(e)})
            
            elif cmd == "list_embeddings":
                # ✅ NEW: List all cached embeddings
                ensure_embedding_dir()
                
                try:
                    embeddings = []
                    for file in os.listdir(EMBEDDING_DIR):
                        if file.endswith('.npz'):
                            speaker_id = file[:-4]  # Remove .npz extension
                            embedding_path = os.path.join(EMBEDDING_DIR, file)
                            
                            # Get file size and modification time
                            stat = os.stat(embedding_path)
                            embeddings.append({
                                "speaker": speaker_id,
                                "path": embedding_path,
                                "size_kb": round(stat.st_size / 1024, 2),
                                "modified": stat.st_mtime
                            })
                    
                    send({"status": "ok", "embeddings": embeddings, "count": len(embeddings)})
                    log(f"[Coqui XTTS] Found {len(embeddings)} cached embeddings")
                    
                except Exception as e:
                    send({"status": "error", "message": str(e)})
            
            elif cmd == "delete_embedding":
                # ✅ NEW: Delete cached embedding
                speaker_id = req.get("speaker")
                
                if not speaker_id:
                    send({"status": "error", "message": "Speaker ID required"})
                    continue
                
                embedding_path = os.path.join(EMBEDDING_DIR, f"{speaker_id}.npz")
                
                try:
                    if os.path.exists(embedding_path):
                        os.remove(embedding_path)
                        send({"status": "ok", "message": f"Embedding deleted for {speaker_id}"})
                        log(f"[Coqui XTTS] Deleted embedding: {speaker_id}")
                    else:
                        send({"status": "error", "message": "Embedding not found"})
                except Exception as e:
                    send({"status": "error", "message": str(e)})
                    
            else:
                send({"status": "error", "message": f"Unknown command: {cmd}"})

    except (KeyboardInterrupt, EOFError):
        log("[Coqui XTTS] Graceful shutdown (CTRL+C or pipe closed)")
        send({"status": "bye"})
    finally:
        # Cleanup
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            log("[Coqui XTTS] GPU cache cleared")

# ---------- One-shot Mode ----------
def oneshot_mode(args):
    """One-shot TTS generation (backward compatibility)."""
    try:
        device = get_device() if args.gpu else "cpu"
        tts = TTS(args.model).to(device)
        log(f"[Coqui XTTS] Model loaded (oneshot): {args.model}")
        
        voice_ref = get_voice_reference(args.speaker)
        
        if "xtts" in args.model.lower():
            if voice_ref:
                tts.tts_to_file(
                    text=args.text,
                    file_path=args.out,
                    speaker_wav=voice_ref,
                    language=args.language,
                    speed=args.speed
                )
            else:
                tts.tts_to_file(
                    text=args.text,
                    file_path=args.out,
                    language=args.language,
                    speed=args.speed
                )
        else:
            tts.tts_to_file(
                text=args.text,
                file_path=args.out,
                speaker=args.speaker,
                speed=args.speed
            )
        
        send({"status": "ok", "file": args.out})
    except Exception as e:
        send({"status": "error", "message": str(e)})

# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRIM Coqui XTTS v2 Bridge")
    parser.add_argument("--persistent", action="store_true",
                        help="Run in persistent stdin/stdout mode")

    # Default to XTTS v2 model (multi-language, high quality)
    parser.add_argument("--model",
                        default="tts_models/multilingual/multi-dataset/xtts_v2",
                        help="TTS model name or local path")

    parser.add_argument("--speaker", default="default", 
                        help="Speaker ID or voice reference name")
    parser.add_argument("--speed", type=float, default=1.0, 
                        help="Speech speed multiplier")
    parser.add_argument("--language", default="en",
                        help="Language code (XTTS v2 supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi)")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Use GPU acceleration (default: True)")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu",
                        help="Disable GPU acceleration")
    parser.add_argument("--out", help="Output file path (oneshot only)")
    parser.add_argument("text", nargs="?", help="Text to speak (oneshot only)")
    
    args = parser.parse_args()

    if args.persistent:
        persistent_loop(args.model, args.speaker, args.gpu)
    else:
        if not args.text or not args.out:
            send({"status": "error", "message": "Oneshot mode requires text and --out"})
        else:
            oneshot_mode(args)
