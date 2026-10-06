import os
import numpy as np
from typing import Optional, Set
from huggingface_hub import snapshot_download
from ovos_plugin_manager.templates.stt import STT
from ovos_utils.log import LOG

try:
    import needle
except ImportError:
    needle = None


class NeedleSTTPlugin(STT):
    """OVOS STT Plugin powered by the Needle / Whistle model."""

    SUPPORTED_LANGUAGES = {"en", "de", "fr", "es", "it", "nl", "pl"}
    TARGET_SAMPLE_RATE = 16000

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.can_stream = False

        if needle is None:
            raise ImportError(
                "The 'needle' package is not installed. "
                "Please install needle via pip or from https://github.com/cactus-compute/needle"
            )

        self.model_name = self.config.get("model", "cactus-compute/whistle")

        model_file_path = self._resolve_model_path(self.model_name)

        LOG.info(f"[NeedleSTT] Loading Whistle model with weights from: '{model_file_path}'")
        try:
            self.model = needle.Whistle(weights=model_file_path)
        except Exception as e:
            LOG.exception(f"[NeedleSTT] Failed to load Whistle model from '{model_file_path}': {e}")
            raise

    @property
    def available_languages(self) -> Set[str]:
        """Returns supported ISO 639-1 language codes for dynamic OVOS discovery."""
        return self.SUPPORTED_LANGUAGES

    def _resolve_model_path(self, model_name_or_path: str) -> str:
        """Downloads the model from Hugging Face Hub (if needed) and returns the full path to the weight file."""
        target_path = model_name_or_path

        # Download directly via HF snapshot if it is not an existing local file or directory
        if not os.path.exists(model_name_or_path):
            LOG.info(f"[NeedleSTT] Checking/downloading model '{model_name_or_path}' from Hugging Face...")
            try:
                target_path = snapshot_download(repo_id=model_name_or_path)
                LOG.info(f"[NeedleSTT] Model ready at: '{target_path}'")
            except Exception as e:
                LOG.warning(f"[NeedleSTT] Auto-download failed: {e}. Falling back to raw path.")
                return model_name_or_path

        # Return directly if the path is a file
        if os.path.isfile(target_path):
            return target_path

        # If the path is a directory, find the specific model weights file
        if os.path.isdir(target_path):
            files = os.listdir(target_path)
            weight_extensions = (".bin", ".safetensors", ".pt", ".onnx", ".gguf", ".tflite")

            # 1. Search for known weight file extensions
            for filename in files:
                if filename.endswith(weight_extensions):
                    resolved = os.path.join(target_path, filename)
                    LOG.info(f"[NeedleSTT] Detected weight file: '{resolved}'")
                    return resolved

            # 2. Fallback: select the largest file in the directory
            non_hidden = [
                os.path.join(target_path, f)
                for f in files
                if not f.startswith(".") and os.path.isfile(os.path.join(target_path, f))
            ]
            if non_hidden:
                resolved = max(non_hidden, key=os.path.getsize)
                LOG.info(f"[NeedleSTT] Selected largest file as weights: '{resolved}'")
                return resolved

        return target_path

    def execute(self, audio, language: Optional[str] = None) -> str:
        """Transcribes OVOS AudioData to text."""
        raw_lang = language or self.config.get("lang") or self.lang or "en"
        lang = raw_lang.replace("_", "-").split("-")[0].lower()

        if lang not in self.SUPPORTED_LANGUAGES:
            LOG.warning(
                f"[NeedleSTT] Language '{lang}' is not in Whistle's supported set "
                f"({', '.join(sorted(self.SUPPORTED_LANGUAGES))}). Falling back to 'en'."
            )
            lang = "en"

        try:
            audio_array = self._audio_data_to_float32(audio)

            LOG.debug(f"[NeedleSTT] Transcribing {len(audio_array)} samples in language '{lang}'...")
            result = self.model.transcribe(audio_array, language=lang)

            if isinstance(result, dict):
                transcript = result.get("text", "")
            else:
                transcript = str(result)

            transcript = transcript.strip()
            LOG.info(f"[NeedleSTT] Result ({lang}): '{transcript}'")
            return transcript

        except Exception as e:
            LOG.exception(f"[NeedleSTT] Error during transcription execution: {e}")
            return ""

    def _audio_data_to_float32(self, audio) -> np.ndarray:
        """Extracts raw PCM bytes from OVOS AudioData and normalizes to float32 (-1.0 to 1.0)."""
        raw_bytes = audio.get_raw_data()
        sample_width = getattr(audio, "sample_width", 2)
        sample_rate = getattr(audio, "sample_rate", self.TARGET_SAMPLE_RATE)

        if sample_width == 2:
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio_np = np.frombuffer(raw_bytes, dtype=np.float32)
        elif sample_width == 1:
            audio_np = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes")

        if sample_rate != self.TARGET_SAMPLE_RATE and len(audio_np) > 0:
            LOG.debug(f"[NeedleSTT] Resampling audio from {sample_rate}Hz to {self.TARGET_SAMPLE_RATE}Hz")
            duration = len(audio_np) / sample_rate
            orig_indices = np.linspace(0, duration, len(audio_np), endpoint=False)
            target_indices = np.linspace(0, duration, int(duration * self.TARGET_SAMPLE_RATE), endpoint=False)
            audio_np = np.interp(target_indices, orig_indices, audio_np).astype(np.float32)

        return audio_np