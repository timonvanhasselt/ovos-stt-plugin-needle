import os
import tempfile
import wave
from typing import Optional, Set
from ovos_plugin_manager.templates.stt import STT
from ovos_utils.log import LOG

try:
    import needle
except ImportError:
    needle = None


class NeedleSTTPlugin(STT):
    """OVOS STT Plugin powered by the Needle / Whistle model."""

    SUPPORTED_LANGUAGES = {"en", "de", "fr", "es", "it", "nl", "pl"}

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.can_stream = False

        if needle is None:
            raise ImportError(
                "The 'cactus-needle' package is not installed. "
                "Please install it via 'pip install cactus-needle'."
            )

        # Retrieve specified model or weights path from config
        weights_path = self.config.get("model") or self.config.get("weights")

        # Only use weights_path if it refers to an existing local file.
        # Fall back to None if omitted or set to a non-existent path/HF repo ID ("cactus-compute/whistle").
        if weights_path and not os.path.exists(weights_path):
            LOG.warning(
                f"[NeedleSTT] Weights path '{weights_path}' was not found as a local file. "
                "Falling back to default Needle model."
            )
            weights_path = None

        LOG.info(f"[NeedleSTT] Loading Whistle model (weights: {weights_path or 'default'})...")
        try:
            # If weights_path is None, Whistle initializes its built-in default model
            if weights_path:
                self.model = needle.Whistle(weights=weights_path)
            else:
                self.model = needle.Whistle()

            LOG.info("[NeedleSTT] Whistle model loaded successfully!")
        except Exception as e:
            LOG.exception(f"[NeedleSTT] Error while loading Whistle model: {e}")
            raise

    @property
    def available_languages(self) -> Set[str]:
        return self.SUPPORTED_LANGUAGES

    def execute(self, audio, language: Optional[str] = None) -> str:
        """Converts OVOS AudioData to a temporary WAV file and transcribes via Whistle."""
        raw_lang = language or self.config.get("lang") or self.lang or "en"
        lang = raw_lang.replace("_", "-").split("-")[0].lower()

        if lang not in self.SUPPORTED_LANGUAGES:
            LOG.warning(f"[NeedleSTT] Language '{lang}' is not supported. Falling back to 'en'.")
            lang = "en"

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                with wave.open(temp_wav.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(getattr(audio, "sample_width", 2))
                    wf.setframerate(getattr(audio, "sample_rate", 16000))
                    wf.writeframes(audio.get_raw_data())

                result = self.model.transcribe(temp_wav.name, language=lang)

            if isinstance(result, dict):
                transcript = result.get("text", "")
            else:
                transcript = str(result)

            transcript = transcript.strip()
            LOG.info(f"[NeedleSTT] Result ({lang}): '{transcript}'")
            return transcript

        except Exception as e:
            LOG.exception(f"[NeedleSTT] Error during transcription: {e}")
            return ""
