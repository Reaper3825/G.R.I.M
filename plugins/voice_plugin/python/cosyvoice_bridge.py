import argparse
import json
import sys
import traceback
from pathlib import Path


def send(payload):
    print(json.dumps(payload), flush=True)


def log(message):
    print(message, file=sys.stderr, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="GRIM local Fun-CosyVoice 3 bridge")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-normalization", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def configure_local_text_normalization(asset_path):
    asset_directory = Path(asset_path).resolve()
    required_files = (
        "en/tn/tagger.fst",
        "en/tn/verbalizer.fst",
        "zh/tn/tagger.fst",
        "zh/tn/verbalizer.fst",
    )
    for relative_path in required_files:
        path = asset_directory / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Text normalization asset not found: {path}")

    import wetext

    upstream_normalizer = wetext.Normalizer

    class LocalNormalizer:
        def __init__(
            self,
            lang="auto",
            operator="tn",
            remove_erhua=False,
            enable_0_to_9=False,
            **_kwargs,
        ):
            if operator != "tn":
                raise ValueError("GRIM's local CosyVoice normalizer supports TN only")
            self.lang = lang
            verbalizer_name = (
                "verbalizer_remove_erhua.fst" if remove_erhua else "verbalizer.fst"
            )
            self.normalizers = {}
            for language in ("en", "zh"):
                if lang not in ("auto", language):
                    continue
                tagger_name = "tagger_enable_0_to_9.fst" if (
                    language == "zh" and enable_0_to_9
                ) else "tagger.fst"
                language_directory = asset_directory / language / "tn"
                selected_verbalizer = language_directory / verbalizer_name
                if not selected_verbalizer.is_file():
                    selected_verbalizer = language_directory / "verbalizer.fst"
                self.normalizers[language] = upstream_normalizer(
                    tagger_path=str(language_directory / tagger_name),
                    verbalizer_path=str(selected_verbalizer),
                    lang=language,
                    operator="tn",
                )

        def normalize(self, text):
            if self.lang == "auto":
                language = "zh" if upstream_normalizer.contains_chinese(text) else "en"
            else:
                language = self.lang
            return self.normalizers[language].normalize(text)

    wetext.Normalizer = LocalNormalizer
    log(f"Using local text normalization assets from {asset_directory}")


def prepare_prompt_text(reference_text):
    prompt_text = reference_text.strip()
    if "<|endofprompt|>" not in prompt_text:
        prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text
    return prompt_text


def register_zero_shot_speaker(model, reference_audio, reference_text, speaker_id):
    reference_path = Path(reference_audio).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference_path}")
    if not speaker_id.strip():
        raise ValueError("CosyVoice speaker id is empty")

    prompt_text = prepare_prompt_text(reference_text)
    normalized_prompt = model.frontend.text_normalize(
        prompt_text,
        split=False,
        text_frontend=True,
    )
    if not model.add_zero_shot_spk(
        normalized_prompt,
        str(reference_path),
        speaker_id,
    ):
        raise RuntimeError(f"Unable to register zero-shot speaker: {speaker_id}")
    log(f"Registered in-memory zero-shot speaker: {speaker_id}")


def load_runtime(
    repository,
    model_path,
    text_normalization_path,
    reference_audio,
    reference_text,
    speaker_id,
    fp16,
):
    repository_path = Path(repository).resolve()
    model_directory = Path(model_path).resolve()
    matcha_path = repository_path / "third_party" / "Matcha-TTS"

    if not repository_path.is_dir():
        raise FileNotFoundError(f"CosyVoice repository not found: {repository_path}")
    if not model_directory.is_dir():
        raise FileNotFoundError(f"CosyVoice model not found: {model_directory}")

    sys.path.insert(0, str(repository_path))
    if matcha_path.is_dir():
        sys.path.insert(0, str(matcha_path))

    import torch
    import torchaudio
    configure_local_text_normalization(text_normalization_path)
    from cosyvoice.cli.cosyvoice import AutoModel

    log(f"Loading Fun-CosyVoice 3 model from {model_directory}")
    model = AutoModel(model_dir=str(model_directory), fp16=fp16)
    register_zero_shot_speaker(
        model,
        reference_audio,
        reference_text,
        speaker_id,
    )
    return model, torch, torchaudio


def synthesize(model, torch, torchaudio, request, registered_speaker_id):
    text = str(request.get("text", "")).strip()
    raw_output_path = str(request.get("out", "")).strip()
    speaker = str(request.get("speaker", "")).strip()
    reference_audio = str(request.get("reference_audio", "")).strip()
    reference_text = str(request.get("reference_text", "")).strip()
    speed = max(0.5, min(2.0, float(request.get("speed", 1.0))))

    if not text:
        raise ValueError("TTS request text is empty")
    if not raw_output_path:
        raise ValueError("TTS output path is empty")
    output_path = Path(raw_output_path).resolve()

    if speaker and speaker == registered_speaker_id:
        generated = model.inference_zero_shot(
            text,
            "",
            "",
            zero_shot_spk_id=registered_speaker_id,
            stream=False,
            speed=speed,
        )
        speaker_mode = "registered"
    elif reference_audio:
        reference_path = Path(reference_audio).resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {reference_path}")
        if reference_text:
            prompt_text = prepare_prompt_text(reference_text)
            generated = model.inference_zero_shot(
                text,
                prompt_text,
                str(reference_path),
                stream=False,
                speed=speed,
            )
            speaker_mode = "request_reference"
        else:
            generated = model.inference_cross_lingual(
                text,
                str(reference_path),
                stream=False,
                speed=speed,
            )
            speaker_mode = "cross_lingual"
    elif speaker and speaker != "default":
        generated = model.inference_zero_shot(
            text,
            "",
            "",
            zero_shot_spk_id=speaker,
            stream=False,
            speed=speed,
        )
        speaker_mode = "registered_request"
    else:
        raise ValueError(
            "CosyVoice requires reference_audio or a registered zero-shot speaker id"
        )

    chunks = []
    for result in generated:
        speech = result.get("tts_speech")
        if speech is not None:
            chunks.append(speech.cpu())
    if not chunks:
        raise RuntimeError("CosyVoice returned no audio chunks")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = torch.cat(chunks, dim=1)
    torchaudio.save(
        str(output_path),
        waveform,
        model.sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )
    return output_path, speaker_mode


def persistent_loop(model, torch, torchaudio, registered_speaker_id):
    send(
        {
            "status": "ready",
            "provider": "fun-cosyvoice3",
            "sample_rate": model.sample_rate,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "precision": "fp16" if getattr(model.model, "fp16", False) else "fp32",
            "speaker": registered_speaker_id,
        }
    )

    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            command = request.get("command")
            if command == "exit":
                send({"status": "bye"})
                return
            if command == "ping":
                send({"status": "ok"})
                continue
            if command != "speak":
                send(
                    {
                        "status": "error",
                        "error_code": "ERR_TTS_COMMAND_UNKNOWN",
                        "message": f"Unknown bridge command: {command}",
                    }
                )
                continue

            output_path, speaker_mode = synthesize(
                model,
                torch,
                torchaudio,
                request,
                registered_speaker_id,
            )
            send(
                {
                    "status": "ok",
                    "file": str(output_path),
                    "speaker_mode": speaker_mode,
                }
            )
        except Exception as error:
            log(traceback.format_exc())
            send(
                {
                    "status": "error",
                    "error_code": "ERR_TTS_SYNTHESIS_FAILED",
                    "message": str(error),
                }
            )


def main():
    args = parse_args()
    try:
        model, torch, torchaudio = load_runtime(
            args.repo,
            args.model,
            args.text_normalization,
            args.reference_audio,
            args.reference_text,
            args.speaker_id,
            args.fp16,
        )
        persistent_loop(model, torch, torchaudio, args.speaker_id)
    except Exception as error:
        log(traceback.format_exc())
        send(
            {
                "status": "error",
                "error_code": "ERR_TTS_INITIALIZATION_FAILED",
                "message": str(error),
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
