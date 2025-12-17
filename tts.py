"""
Text-to-Speech service using Mimic3.
Replacement for ChatterBox TTS to work with Python 3.12.

Lucy: reemplazamos chatterbox-tts por Mimic3TTS para mantener todo local y
compatible con Python 3.12 y alineado con Proyecto-VSCode.
"""
import io
import subprocess
from typing import Tuple, Optional

import nltk
import numpy as np
import soundfile as sf


class TextToSpeechService:
    """
    Implementación simple de TTS que llama al binario `mimic3`
    y devuelve (sample_rate, numpy_array) como espera app.py.
    """

    def __init__(
        self,
        voice: str = "es_ES/m-ailabs_low#karen_savage",
        sample_rate: int = 16000,
        device: Optional[str] = None,
    ):
        """
        Initializes the TextToSpeechService class with Mimic3 TTS.

        Args:
            voice (str): The Mimic3 voice to use. Defaults to Spanish.
            device (str, optional): Ignored for compatibility with original interface.
        """
        self.voice = voice
        self.sample_rate = sample_rate
        self._mimic3_sample_rate = 22050  # default mimic3 output
        print(f"Using Mimic3 voice: {self.voice} @ {self.sample_rate}Hz")

    def synthesize(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> Tuple[int, np.ndarray]:
        """
        Synthesizes audio from the given text using Mimic3 TTS.

        Args:
            text (str): The input text to be synthesized.
            audio_prompt_path (str, optional): Ignored (kept for compatibility).
            exaggeration (float, optional): Ignored (kept for compatibility).
            cfg_weight (float, optional): Ignored (kept for compatibility).

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        # Call mimic3 by CLI and capture the WAV output
        cmd = ["mimic3", "--voice", self.voice, "--stdout"]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                input=text.encode("utf-8"),
                check=True,
            )
            wav_bytes = proc.stdout

            # Si mimic3 escribe algo a stderr aun con rc=0, no lo traguemos:
            stderr_txt = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()
            if stderr_txt:
                lines = [ln.strip() for ln in stderr_txt.splitlines() if ln.strip()]
                non_info = [ln for ln in lines if not ln.startswith("INFO:")]
                if non_info:
                    print("[Mimic3] stderr:", " | ".join(non_info[:5]), flush=True)
        except subprocess.CalledProcessError as e:
            print(f"Error calling mimic3: {e} | stderr={e.stderr.decode(errors='ignore')}")
            # Return silence on error
            return self.sample_rate, np.zeros(self.sample_rate)
        except FileNotFoundError:
            print("Error: mimic3 command not found. Please install Mimic3.")
            return self.sample_rate, np.zeros(self.sample_rate)

        # Convert WAV bytes -> (float32 array, sample_rate)
        try:
            audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        except Exception as e:
            print(f"Error reading audio from mimic3: {e}")
            return self.sample_rate, np.zeros(self.sample_rate)

        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample to target sample_rate if needed so downstream playback is consistent.
        if sample_rate != self.sample_rate:
            duration = len(audio) / sample_rate
            target_len = int(duration * self.sample_rate)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, num=target_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
            sample_rate = self.sample_rate

        return sample_rate, audio

    def long_form_synthesize(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> Tuple[int, np.ndarray]:
        """
        Synthesizes audio from the given long-form text using Mimic3 TTS.

        Args:
            text (str): The input text to be synthesized.
            audio_prompt_path (str, optional): Ignored (kept for compatibility).
            exaggeration (float, optional): Ignored (kept for compatibility).
            cfg_weight (float, optional): Ignored (kept for compatibility).

        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(
                sent,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            pieces += [audio_array, silence.copy()]

        return self.sample_rate, np.concatenate(pieces)

    def save_voice_sample(
        self, text: str, output_path: str, audio_prompt_path: str | None = None
    ):
        """
        Saves a voice sample to file for later use as voice prompt.

        Args:
            text (str): The text to synthesize.
            output_path (str): Path where to save the audio file.
            audio_prompt_path (str, optional): Ignored (kept for compatibility).
        """
        sample_rate, audio = self.synthesize(text)
        sf.write(output_path, audio, sample_rate)
