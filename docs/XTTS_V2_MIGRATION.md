# XTTS v2 Migration Checklist

## Pre-Migration

- [ ] Backup current `ai_config.json`
- [ ] Note current speaker IDs in use
- [ ] Check available disk space (~2GB for model)
- [ ] Verify Python 3.9+ installed
- [ ] Check GPU availability (optional but recommended)

## Migration Steps

### 1. Install XTTS v2

```powershell
cd D:\G.R.I.M\scripts
.\setup_coqui_xtts_v2.ps1
```

**Expected output:**
```
? Found: Python 3.x.x
? NVIDIA GPU detected
? PyTorch installed
? Coqui TTS installed
? XTTS v2 model ready
```

### 2. Update Configuration

Edit `D:\G.R.I.M\resources\ai_config.json`:

**Before:**
```json
{
  "voice": {
    "engine": "coqui",
    "speaker": "p225",
    "speed": 1.0
  }
}
```

**After:**
```json
{
  "voice": {
    "engine": "coqui",
    "speaker": "default",
    "speed": 1.0,
    "language": "en"
  }
}
```

### 3. Rebuild Project

```powershell
cd D:\G.R.I.M\out\build
cmake --build . --config Release
```

### 4. Test XTTS v2

Launch G.R.I.M and run:
```
test_tts Hello, this is XTTS version two!
```

**Expected behavior:**
- First run: 30-60s model load time
- Subsequent runs: 0.5-2s per phrase (GPU) or 5-10s (CPU)
- High quality, natural voice

### 5. Verify GPU Usage (if applicable)

In separate terminal:
```powershell
nvidia-smi
```

Look for `python.exe` process using GPU memory (~1.5GB).

### 6. Clean Old Data

```powershell
cd D:\G.R.I.M\scripts
.\cleanup_old_coqui.ps1
```

This frees disk space from old models.

## Post-Migration Testing

### Test Commands

1. **Basic TTS:**
   ```
   test_tts This is a test
   ```

2. **List Configuration:**
   ```
   list_voices
   ```
   Should show "XTTS v2" in output.

3. **Test Cache:**
   ```
   test_tts Welcome back, Austin.
   ```
   Run twice - second should be instant (cached).

4. **Test Speed:**
   ```json
   // In ai_config.json
   "speed": 1.2  // Faster
   "speed": 0.8  // Slower, clearer
   ```

### Verification Checklist

- [ ] XTTS v2 shown in `list_voices` output
- [ ] GPU usage visible in `nvidia-smi` (if applicable)
- [ ] Speech quality improved vs old Coqui
- [ ] Cache working (repeat phrases are instant)
- [ ] No errors in logs

## Troubleshooting

### Issue: Model Download Fails

**Symptom:** Timeout during setup or first run

**Fix:**
```powershell
# Manual download
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

### Issue: GPU Not Used

**Symptom:** Slow synthesis, no GPU usage in nvidia-smi

**Fix:**
```powershell
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify:**
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Bridge Timeout

**Symptom:** "No handshake received from Coqui XTTS v2 after 120s"

**Fix:**
1. Check Python bridge manually:
   ```powershell
   cd D:\G.R.I.M\resources\python
   python coqui_bridge.py --persistent
   ```
2. Check for errors in output
3. Increase timeout in `voice_speak.cpp` if needed

### Issue: Poor Audio Quality

**Possible causes:**
- Wrong language setting
- Speed too high/low
- CPU vs GPU (GPU generally better)

**Fix:**
```json
{
  "voice": {
    "language": "en",  // Must match text language
    "speed": 1.0,      // Try 0.9-1.1 range
    "speaker": "default"
  }
}
```

### Issue: Out of Memory (GPU)

**Symptom:** CUDA out of memory errors

**Fix:**
```powershell
# Use CPU mode temporarily
python coqui_bridge.py --persistent --no-gpu
```

Or reduce other GPU usage (close games, browsers).

## Rollback Plan

If XTTS v2 has issues, rollback to old Coqui:

### 1. Edit `voice_speak.cpp`:

Change line in `initTTS()`:
```cpp
// From:
std::string cmd = "python ... --model tts_models/multilingual/multi-dataset/xtts_v2 --gpu";

// To:
std::string cmd = "python ... --model tts_models/en/vctk/vits";
```

### 2. Rebuild:
```powershell
cmake --build . --config Release
```

### 3. Update config:
```json
{
  "voice": {
    "speaker": "p225"  // Old speaker ID
  }
}
```

## Success Criteria

Migration is successful when:

? G.R.I.M launches without errors  
? `test_tts` produces high-quality speech  
? GPU is utilized (if available)  
? Speech synthesis < 2s per phrase (GPU) or < 10s (CPU)  
? Cache working (instant playback on repeat)  
? No Python errors in logs  

## Next Steps

After successful migration:

1. **Explore Languages:**
   ```cpp
   Voice::setLanguage("es");
   Voice::speak("Hola mundo", "system");
   ```

2. **Add Custom Voices:**
   - Record voice samples
   - Update `VOICE_REFERENCES` in `coqui_bridge.py`

3. **Optimize Performance:**
   - Adjust cache settings
   - Tune speed parameter
   - Monitor GPU usage

4. **Monitor Logs:**
   - Check `D:\G.R.I.M\logs\` for issues
   - Watch for cache hit rate

## Support

If issues persist:
1. Check full documentation: `docs/COQUI_XTTS_V2.md`
2. Review logs in `D:\G.R.I.M\logs\`
3. Test Python bridge independently
4. Verify CUDA installation (if using GPU)
