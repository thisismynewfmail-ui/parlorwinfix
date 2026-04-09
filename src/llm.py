"""Platform-aware LLM backend.

- Linux / macOS: litert-lm with Gemma 4 E2B (GPU, multimodal, native tool calling).
- Windows:      HuggingFace transformers for text + vision, faster-whisper for audio.

Both backends expose the same `Backend.create_conversation()` / `Conversation.send()`
interface, so server.py doesn't need to know which one is running.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import wave
from typing import Protocol

import numpy as np


SEMANTIC_SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. "
    "Keep replies to 1-4 short, natural sentences."
)


class Conversation(Protocol):
    def send(self, content: list[dict]) -> dict:
        """Send a user turn with multimodal content.

        ``content`` is a list of items shaped like:
            {"type": "audio", "blob": <base64 WAV>}
            {"type": "image", "blob": <base64 JPEG>}
            {"type": "text",  "text": <str>}

        Returns:
            {"transcription": str | None, "response": str}
        """
        ...

    def close(self) -> None: ...


class Backend(Protocol):
    def create_conversation(self) -> Conversation: ...
    def close(self) -> None: ...


# ─────────────────────────── LiteRT-LM backend ───────────────────────────


HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"


def _resolve_litert_model() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


_LITERT_SYSTEM_PROMPT = SEMANTIC_SYSTEM_PROMPT + (
    " You MUST always use the respond_to_user tool to reply. "
    "First transcribe exactly what the user said, then write your response."
)


class LiteRTConversation:
    def __init__(self, inner, tool_result: dict):
        self._inner = inner
        self._tool_result = tool_result
        self._inner.__enter__()

    def send(self, content: list[dict]) -> dict:
        self._tool_result.clear()
        response = self._inner.send_message({"role": "user", "content": content})

        if self._tool_result:
            def strip(s: str) -> str:
                return s.replace('<|"|>', "").strip()

            transcription = strip(self._tool_result.get("transcription", "")) or None
            return {
                "transcription": transcription,
                "response": strip(self._tool_result.get("response", "")),
            }
        return {
            "transcription": None,
            "response": response["content"][0]["text"],
        }

    def close(self) -> None:
        self._inner.__exit__(None, None, None)


class LiteRTBackend:
    def __init__(self) -> None:
        import litert_lm

        model_path = _resolve_litert_model()
        print(f"Loading Gemma 4 E2B from {model_path}...")
        self._engine = litert_lm.Engine(
            model_path,
            backend=litert_lm.Backend.GPU,
            vision_backend=litert_lm.Backend.GPU,
            audio_backend=litert_lm.Backend.CPU,
        )
        self._engine.__enter__()
        print("LLM: litert-lm (Gemma 4 E2B, GPU)")

    def create_conversation(self) -> LiteRTConversation:
        tool_result: dict = {}

        def respond_to_user(transcription: str, response: str) -> str:
            """Respond to the user's voice message.

            Args:
                transcription: Exact transcription of what the user said in the audio.
                response: Your conversational response to the user. Keep it to 1-4 short sentences.
            """
            tool_result["transcription"] = transcription
            tool_result["response"] = response
            return "OK"

        inner = self._engine.create_conversation(
            messages=[{"role": "system", "content": _LITERT_SYSTEM_PROMPT}],
            tools=[respond_to_user],
        )
        return LiteRTConversation(inner, tool_result)

    def close(self) -> None:
        self._engine.__exit__(None, None, None)


# ─────────────────────────── Transformers backend (Windows) ───────────────────────────


def _decode_wav_to_mono16k(b64_wav: str) -> np.ndarray:
    """base64 WAV → float32 mono PCM @ 16 kHz (what Whisper expects)."""
    wav_bytes = base64.b64decode(b64_wav)
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        frames = w.readframes(w.getnframes())

    if sampwidth == 2:
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        pcm = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)

    if sr != 16000 and len(pcm) > 0:
        # Linear-interpolation resample. Whisper is forgiving.
        new_len = int(round(len(pcm) * 16000 / sr))
        xs = np.linspace(0, len(pcm) - 1, num=new_len, dtype=np.float32)
        pcm = np.interp(xs, np.arange(len(pcm), dtype=np.float32), pcm).astype(np.float32)

    return pcm


class TransformersConversation:
    def __init__(self, backend: "TransformersBackend") -> None:
        self._backend = backend
        self._messages: list[dict] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SEMANTIC_SYSTEM_PROMPT}],
            }
        ]

    def send(self, content: list[dict]) -> dict:
        from PIL import Image

        transcription: str | None = None
        user_parts: list[dict] = []
        stored_parts: list[dict] = []  # text-only copy kept in history

        for item in content:
            kind = item.get("type")
            if kind == "audio" and item.get("blob"):
                pcm = _decode_wav_to_mono16k(item["blob"])
                segments, _info = self._backend.whisper.transcribe(pcm, language="en")
                transcription = " ".join(s.text.strip() for s in segments).strip()
                if transcription:
                    user_parts.append({"type": "text", "text": f'User said: "{transcription}"'})
                    stored_parts.append({"type": "text", "text": f'User said: "{transcription}"'})
            elif kind == "image" and item.get("blob"):
                img_bytes = base64.b64decode(item["blob"])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                user_parts.append({"type": "image", "image": img})
                stored_parts.append({"type": "text", "text": "[image]"})
            elif kind == "text" and item.get("text"):
                user_parts.append({"type": "text", "text": item["text"]})
                stored_parts.append({"type": "text", "text": item["text"]})

        if not user_parts:
            return {"transcription": transcription, "response": ""}

        # Run generation with the full turn (may include image)
        turn_messages = self._messages + [{"role": "user", "content": user_parts}]
        reply = self._backend.generate(turn_messages)

        # Store a lighter, text-only copy to keep KV/memory under control
        self._messages.append({"role": "user", "content": stored_parts})
        self._messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": reply}]}
        )

        return {"transcription": transcription, "response": reply}

    def close(self) -> None:
        self._messages = []


class TransformersBackend:
    def __init__(self) -> None:
        import torch
        from faster_whisper import WhisperModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = os.environ.get("HF_MODEL_ID", "google/gemma-3-4b-it")
        whisper_model = os.environ.get("WHISPER_MODEL", "base.en")

        self._torch = torch

        print(f"Loading {model_id} via transformers...")
        self.processor = AutoProcessor.from_pretrained(model_id)

        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = "auto" if use_cuda else None

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if device_map is None:
            self.model = self.model.to("cpu")
        self.model.eval()
        self._device = next(self.model.parameters()).device

        print(f"Loading faster-whisper ({whisper_model})...")
        whisper_device = "cuda" if use_cuda else "cpu"
        whisper_compute = "float16" if use_cuda else "int8"
        self.whisper = WhisperModel(
            whisper_model, device=whisper_device, compute_type=whisper_compute
        )

        print(
            f"LLM: transformers ({model_id} on {self._device}) "
            f"+ faster-whisper ({whisper_model} on {whisper_device})"
        )

    def generate(self, messages: list[dict]) -> str:
        torch = self._torch
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        return self.processor.decode(
            output[0][input_len:], skip_special_tokens=True
        ).strip()

    def create_conversation(self) -> TransformersConversation:
        return TransformersConversation(self)

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.whisper = None


# ─────────────────────────── Factory ───────────────────────────


def load() -> Backend:
    """Load the best available LLM backend for this platform."""
    if sys.platform == "win32":
        return TransformersBackend()
    return LiteRTBackend()
