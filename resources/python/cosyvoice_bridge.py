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
    return parser.parse_args()


def load_runtime(repository, model_path):
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
    from cosyvoice.cli.auto_model import AutoModel

    log(f"Loading Fun-CosyVoice 3 model from {model_directory}")
    model = AutoModel(model_dir=str(model_directory))
    return model, torch, torchaudio


def synthesize(model, torch, torchaudio, request):
    text = str(request.get("text", "")).strip()
    raw_output_path = str(request.get("out", "")).strip()
    speaker = str(request.get("speaker", "")).strip()
    reference_audio = str(request.get("reference_audio", "")).strip()
    reference_text = str(request.get("reference_text", "")).strip()

    if not text:
        raise ValueError("TTS request text is empty")
    if not raw_output_path:
        raise ValueError("TTS output path is empty")
    output_path = Path(raw_output_path).resolve()

    if reference_audio and reference_text:
        reference_path = Path(reference_audio).resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {reference_path}")
        prompt_text = reference_text
        if "<|endofprompt|>" not in prompt_text:
            prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text
        generated = model.inference_zero_shot(
            text,
            prompt_text,
            str(reference_path),
            stream=False,
        )
    elif speaker and speaker != "default":
        generated = model.inference_zero_shot(
            text,
            "",
            "",
            zero_shot_spk_id=speaker,
            stream=False,
        )
    else:
        raise ValueError(
            "CosyVoice requires reference_audio plus reference_text, "
            "or a registered zero-shot speaker id"
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
    torchaudio.save(str(output_path), waveform, model.sample_rate)
    return output_path


def persistent_loop(model, torch, torchaudio):
    send(
        {
            "status": "ready",
            "provider": "fun-cosyvoice3",
            "sample_rate": model.sample_rate,
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

            output_path = synthesize(model, torch, torchaudio, request)
            send({"status": "ok", "file": str(output_path)})
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
        model, torch, torchaudio = load_runtime(args.repo, args.model)
        persistent_loop(model, torch, torchaudio)
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
