import argparse
import json
import os
import re
import sys
import threading
from queue import Queue

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import sounddevice as sd
import webrtcvad
import whisper
import yaml
from langchain_core.chat_history import InMemoryChatMessageHistory
# Updated imports for modern LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM
from ollama._types import ResponseError
from rich.console import Console

from tts import TextToSpeechService
from lucy_agents.desktop_bridge import run_desktop_command
from lucy_agents.voice_actions import maybe_handle_desktop_intent

console = Console()
SAMPLE_RATE = 16000  # Hz, used for both VAD and Whisper


def _log(msg: str) -> None:
    print(msg, flush=True)

DEFAULT_CONFIG = {
    "tts": {
        "engine": "mimic3",
        "voice": "es_ES/m-ailabs_low",
        "sample_rate": 16000,
    },
    "stt": {
        "model": "small",
        "language": "es",
        "task": "transcribe",
    },
    "llm": {
        "model": "gpt-oss:20b",
        "base_url": "http://localhost:11434",
    },
}


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        return DEFAULT_CONFIG

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return DEFAULT_CONFIG

    # merge defaults
    cfg = DEFAULT_CONFIG.copy()
    for section, values in data.items():
        if isinstance(values, dict):
            cfg.setdefault(section, {}).update(values)
    return cfg


config = load_config()

# Parse command line arguments using config defaults
parser = argparse.ArgumentParser(description="Local Voice Assistant with Mimic3 TTS (Lucy-style)")
parser.add_argument("--voice", type=str, default=config["tts"]["voice"], help="Mimic3 voice id (e.g., es_ES/m-ailabs_low)")
parser.add_argument("--sample-rate", type=int, default=config["tts"]["sample_rate"], help="Output sample rate for TTS/audio playback")
parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (0.0-1.0)")
parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight for pacing (0.0-1.0)")
parser.add_argument("--model", type=str, default=config["llm"]["model"], help="Ollama model to use")
parser.add_argument("--save-voice", action="store_true", help="Save generated voice samples")
parser.add_argument("--stt-model", type=str, default=config["stt"]["model"], help="Whisper model name (Spanish-capable)")
parser.add_argument("--stt-language", type=str, default=config["stt"]["language"], help="Target language for Whisper")
parser.add_argument("--stt-task", type=str, default=config["stt"]["task"], help="Whisper task: transcribe/translate")
args = parser.parse_args()

# Initialize TTS with Mimic3 (default Spanish voice)
tts = TextToSpeechService(voice=args.voice, sample_rate=args.sample_rate)

# Spanish-first model; language enforced during transcribe to keep STT stable.
stt = whisper.load_model(args.stt_model)

# Modern prompt template using ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "Sos Lucy, asistente de voz local en español rioplatense. "
        "Usás la herramienta 'desktop_agent' del Desktop Agent para abrir aplicaciones y URLs. "
        "Ante pedidos de abrir o buscar en Google o YouTube ('abrí Google y buscá X', 'buscá X en YouTube'), "
        "preferí los intents preprocesados de lucy_agents.voice_actions o un tool call 'desktop_agent'; nunca respondas que no podés si Desktop Agent puede abrirlo. "
        "Si la acción exige clics/scroll dentro de la página, igual abrí la búsqueda o URL con Desktop Agent y avisá que aún no podés interactuar adentro. "
        "Si una orden de escritorio no se entiende, pedí que la repitan o aclaren. "
        "Respondé siempre en respuestas breves (<=20 palabras), tono directo y cordial."
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize LLM
llm = OllamaLLM(model=args.model, base_url=config["llm"].get("base_url", "http://localhost:11434"))

# Create the chain with modern LCEL syntax
chain = prompt_template | llm

# LLM warm-up flag
_LLM_WARMED_UP = False

# Chat history storage
chat_sessions = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]


def _warmup_llm(chain_runnable) -> None:
    """
    Hace una llamada dummy al LLM para forzar la carga del modelo.
    No debe tumbar el nodo; los errores se loguean y se continúan.
    """
    global _LLM_WARMED_UP
    if _LLM_WARMED_UP:
        return
    try:
        print("[LucyVoice] Warming up LLM (gpt-oss:20b)...")
        _ = chain_runnable.invoke({"input": "Decí solo OK."}, config={"session_id": "warmup_session"})
        print("[LucyVoice] LLM warm-up done.")
    except Exception as e:  # noqa: BLE001
        print(f"[LucyVoice] LLM warm-up FAILED: {e!r}")
    finally:
        _LLM_WARMED_UP = True

# Create the runnable with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def record_audio(vad: webrtcvad.Vad, max_silence_ms: int = 1000, frame_duration_ms: int = 30) -> np.ndarray:
    """
    Records a single utterance using VAD to auto-start and auto-stop on silence.
    Returns float32 PCM normalized to [-1, 1].
    """
    frame_samples = int(SAMPLE_RATE * frame_duration_ms / 1000)
    max_silence_frames = max(1, max_silence_ms // frame_duration_ms)
    data_queue = Queue()  # type: ignore[var-annotated]
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))
        if stop_event.is_set():
            raise sd.CallbackStop()

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        dtype="int16",
        channels=1,
        blocksize=frame_samples,
        callback=callback,
    ):
        frames_bytes = []
        triggered = False
        silence_frames = 0

        while True:
            try:
                frame = data_queue.get(timeout=1)
            except Exception:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            if is_speech:
                silence_frames = 0
                triggered = True
                frames_bytes.append(frame)
            elif triggered:
                silence_frames += 1
                frames_bytes.append(frame)
                if silence_frames >= max_silence_frames:
                    break

        stop_event.set()

    audio_data = b"".join(frames_bytes)
    if not audio_data:
        return np.array([], dtype=np.float32)

    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np


def record_until_silence(
    sample_rate: int = SAMPLE_RATE,
    frame_duration_ms: int = 30,
    max_silence_ms: int = 1000,
    max_utterance_ms: int = 15000,
    device: int | None = None,
):
    """
    Graba desde el micrófono hasta detectar silencio final usando VAD.

    - sample_rate: frecuencia de muestreo (debe ser 16000 Hz para webrtcvad).
    - frame_duration_ms: tamaño de cada frame de análisis en milisegundos (10, 20 o 30 ms).
    - max_silence_ms: cuánto silencio seguido se tolera antes de cortar (ej: 1000 ms).
    - max_utterance_ms: duración máxima dura de un turno, por seguridad (ej: 15000 ms).
    - device: índice de dispositivo de entrada (None = por defecto).

    Devuelve:
      - audio_np: np.ndarray float32 1D en rango [-1.0, 1.0], o None si no hubo voz.
      - sample_rate: la frecuencia de muestreo usada.
    """
    vad = webrtcvad.Vad()
    # Modo de VAD (0 a 3): 0 = muy permisivo, 3 = muy estricto.
    # Usamos 1 como término medio para no cortar demasiado agresivo.
    vad.set_mode(1)

    frame_size = int(sample_rate * frame_duration_ms / 1000)  # muestras por frame
    bytes_per_sample = 2  # int16
    frame_bytes = frame_size * bytes_per_sample

    max_frames = max_utterance_ms // frame_duration_ms
    max_silence_frames = max_silence_ms // frame_duration_ms

    voiced_frames: list[bytes] = []
    triggered = False
    num_silence_frames = 0

    # RawInputStream nos da bytes int16, que es justo lo que webrtcvad espera.
    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=frame_size,
        channels=1,
        dtype="int16",
        device=device,
    ) as stream:
        while True:
            data, overflowed = stream.read(frame_size)
            if overflowed:
                continue

            if len(data) < frame_bytes:
                continue

            is_speech = vad.is_speech(data, sample_rate)

            if not triggered:
                if is_speech:
                    triggered = True
                    voiced_frames.append(data)
                    num_silence_frames = 0
                else:
                    continue
            else:
                voiced_frames.append(data)

                if is_speech:
                    num_silence_frames = 0
                else:
                    num_silence_frames += 1

                if num_silence_frames >= max_silence_frames:
                    break

                if len(voiced_frames) >= max_frames:
                    break

    if not voiced_frames:
        return None, sample_rate

    pcm_bytes = b"".join(voiced_frames)
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    return audio_np, sample_rate


def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(
        audio_np,
        fp16=False,  # Set fp16=True if using a GPU
        language=args.stt_language,
        task=args.stt_task,
    )
    text = result["text"].strip()
    return text


JSON_CODE_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


def _extract_desktop_tool_json(text: str) -> str | None:
    """
    Si el texto del LLM contiene un JSON para la tool 'desktop_agent',
    devuelve el bloque JSON como string. Si no, devuelve None.
    """
    if not text:
        return None

    t = text.strip()

    if t.startswith("{") and t.endswith("}"):
        return t

    return None


def _extract_json_code_blocks(text: str) -> list[str]:
    """
    Devuelve todos los bloques JSON dentro de ```json ... ``` en orden.
    """
    if not text:
        return []

    return [m.strip() for m in JSON_CODE_BLOCK_RE.findall(text)]


def _strip_json_code_blocks(text: str) -> str:
    """Elimina cualquier bloque ```json ... ``` del texto."""
    if not text:
        return ""
    return JSON_CODE_BLOCK_RE.sub("", text).strip()


def _handle_desktop_tool_json(json_str: str) -> tuple[bool, str]:
    """
    Intenta interpretar un JSON de tool 'desktop_agent' y ejecutar la acción
    vía Lucy Desktop Agent. Devuelve (handled, texto_para_el_usuario).
    """
    try:
        data = json.loads(json_str)
    except Exception as e:  # noqa: BLE001
        _log(f"[LucyVoice] No pude parsear JSON de tool: {e}: {json_str!r}")
        return False, ""

    if not isinstance(data, dict):
        return False, ""

    if data.get("name") != "desktop_agent":
        return False, ""

    args = data.get("arguments") or {}
    command = args.get("command")
    url = args.get("url")

    # Si falta "command" pero hay URL, asumimos open_url (el modelo a veces omite el campo).
    if isinstance(url, str) and not command:
        command = "open_url"

    if command == "open_url" and isinstance(url, str):
        if not (url.startswith("http://") or url.startswith("https://")):
            _log(f"[LucyVoice] URL de desktop_agent no segura: {url!r}")
            return False, ""

        desktop_cmd = f"xdg-open {url}"
        _log(f"[LucyVoice] Executing desktop_agent.open_url -> {desktop_cmd!r}")
        exit_code = run_desktop_command(desktop_cmd)
        _log(f"[LucyVoice] desktop_agent.open_url exit_code: {exit_code}")

        if exit_code == 0:
            return True, "Listo, ya lo abrí en tu escritorio."
        else:
            return True, (
                "Intenté abrirlo pero algo falló en el escritorio. "
                "Revisá si el navegador está instalado y funcionando."
            )

    _log(f"[LucyVoice] Comando desktop_agent no soportado: {command!r}")
    return False, ""


def get_llm_response(text: str) -> str:
    """
    Devuelve el texto final que Lucy le va a decir al usuario.

    Orden de trabajo:
      1) Intentar resolver con voice_actions (reglas + Desktop Agent).
      2) Si no, llamar al LLM (chain_with_history).
      3) Si el LLM devuelve un JSON de tool 'desktop_agent', ejecutarlo
         vía Lucy Desktop Agent y devolver un resumen amigable.
      4) Si el LLM falla o devuelve vacío, usar un fallback seguro.
    """
    _log(f"[LucyVoice] get_llm_response() input: {text!r}")

    # 1) Agencia rule-based (sin invocar el LLM).
    handled = False
    handled_text = "Listo, ya lo abrí en tu escritorio."
    try:
        agency_result = maybe_handle_desktop_intent(text)
        if isinstance(agency_result, tuple) and len(agency_result) == 2:
            handled, handled_text = agency_result
        elif isinstance(agency_result, bool):
            handled = agency_result
    except Exception as exc:  # noqa: BLE001
        _log(f"[LucyVoice] Error en voice_actions: {exc!r}")
        handled = False

    if handled:
        _log("[LucyVoice] Resuelto por voice_actions (desktop agent).")
        _log(f"[LucyVoice] get_llm_response() final spoken: {handled_text!r}")
        return handled_text

    # 2) Llamar al LLM con manejo de errores.
    raw = ""
    try:
        session_id = "voice_assistant_session"
        llm_out = chain_with_history.invoke({"input": text}, config={"session_id": session_id})
        if isinstance(llm_out, str):
            raw = llm_out
        elif hasattr(llm_out, "content"):
            raw = str(llm_out.content)
        else:
            raw = str(llm_out)
    except ResponseError as e:
        _log(f"[LucyVoice] ResponseError desde Ollama: {e}")
        raw = ""
    except Exception as e:  # noqa: BLE001
        _log(f"[LucyVoice] Error llamando al LLM: {e}")
        raw = ""

    raw = (raw or "").strip()
    _log(f"[LucyVoice] get_llm_response() raw output: {raw!r}")

    # 3) Si parece un JSON de desktop_agent, intentamos ejecutarlo.
    tool_json = _extract_desktop_tool_json(raw)
    if tool_json is not None:
        handled_json, spoken = _handle_desktop_tool_json(tool_json)
        if handled_json and spoken:
            _log("[LucyVoice] desktop_agent ejecutado a partir de JSON del LLM.")
            _log(f"[LucyVoice] get_llm_response() final spoken: {spoken!r}")
            return spoken

    # 3b) Si el JSON viene dentro de bloques ```json ... ```, ejecutamos todos en orden.
    try:
        block_jsons = _extract_json_code_blocks(raw)
    except Exception as e:  # noqa: BLE001
        _log(f"[LucyVoice] Error extrayendo bloques JSON: {e}")
        block_jsons = []

    if block_jsons:
        actions_executed = 0
        for block_json in block_jsons:
            try:
                handled_json, spoken = _handle_desktop_tool_json(block_json)
            except Exception as e:  # noqa: BLE001
                _log(f"[LucyVoice] Error manejando bloque JSON: {e}")
                continue

            if handled_json and spoken:
                actions_executed += 1

        cleaned_raw = _strip_json_code_blocks(raw).strip()
        raw = cleaned_raw

        _log(f"[LucyVoice] Executed {actions_executed} desktop_agent action(s) from JSON code blocks.")

        if actions_executed > 0:
            output_text = cleaned_raw or "Listo, ya abrí las búsquedas que me pediste en tu escritorio."
            _log(f"[LucyVoice] get_llm_response() output (after JSON blocks): {output_text!r}")
            _log(f"[LucyVoice] get_llm_response() final spoken: {output_text!r}")
            return output_text

        if cleaned_raw:
            _log(f"[LucyVoice] get_llm_response() output (after JSON blocks): {cleaned_raw!r}")
            _log(f"[LucyVoice] get_llm_response() final spoken: {cleaned_raw!r}")
            return cleaned_raw

    # 4) Fallback si no hay texto útil.
    if not raw:
        fallback = (
            "No entendí bien qué querías que haga. "
            "Probá pedírmelo de nuevo, más despacio o en pasos separados."
        )
        _log("[LucyVoice] Respuesta vacía del LLM; usando fallback seguro.")
        _log(f"[LucyVoice] get_llm_response() final spoken: {fallback!r}")
        return fallback

    # 5) Respuesta normal del modelo (texto plano).
    _log(f"[LucyVoice] get_llm_response() final spoken: {raw!r}")
    return raw


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


def analyze_emotion(text: str) -> float:
    """
    Simple emotion analysis to dynamically adjust exaggeration.
    Returns a value between 0.3 and 0.9 based on text content.
    """
    # Keywords that suggest more emotion
    emotional_keywords = ['amazing', 'terrible', 'love', 'hate', 'excited', 'sad', 'happy', 'angry', 'wonderful', 'awful', '!', '?!', '...']

    emotion_score = 0.5  # Default neutral

    text_lower = text.lower()
    for keyword in emotional_keywords:
        if keyword in text_lower:
            emotion_score += 0.1

    # Cap between 0.3 and 0.9
    return min(0.9, max(0.3, emotion_score))


def is_sleep_command(text: str) -> bool:
    """
    Devuelve True si el usuario pidió que Lucy 'duerma' (e.g., 'lucy dormi' o 'lucy dormí').
    """
    if not text:
        return False

    normalized = text.strip().lower()
    for ch in ["¡", "!", "¿", "?", ".", ","]:
        normalized = normalized.replace(ch, " ")

    normalized = " ".join(normalized.split())
    return "lucy dormi" in normalized or "lucy dormí" in normalized


if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant with Mimic3 TTS")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if args.voice:
        console.print(f"[green]Using Mimic3 voice: {args.voice}")

    console.print(f"[blue]Emotion exaggeration: {args.exaggeration}")
    console.print(f"[blue]CFG weight: {args.cfg_weight}")
    console.print(f"[blue]LLM model: {args.model}")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Press Ctrl+C to exit.\n")

    # Create voices directory if saving voices
    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    response_count = 0

    # Warm-up LLM to reduce first-call latency
    _warmup_llm(chain_with_history)

    console.print("🎤 Presioná Enter una vez para empezar a hablar (Ctrl+C para salir).")
    console.input("")  # Single prompt to kick off the session

    try:
        while True:
            console.print("[cyan]Escuchando...[/cyan]")

            audio_np, sr = record_until_silence(sample_rate=SAMPLE_RATE)

            if audio_np is None:
                console.print("[red]Solo silencio detectado. Escuchando nuevamente...[/red]")
                continue

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="dots"):
                    text = transcribe(audio_np)

                if not text:
                    _log("[LucyVoice] STT text vacío.")
                    console.print("[red]Silencio detectado. Escuchando nuevamente...[/red]")
                    continue

                _log(f"[LucyVoice] STT text: {text!r}")
                console.print(f"You: {text}")

                if is_sleep_command(text):
                    console.print("[cyan][Lucy] Recibí la orden 'lucy dormi'. Me voy a dormir y cierro la sesión.[/cyan]")
                    break

                with console.status("Generating response...", spinner="dots"):
                    _log(f"[LucyVoice] STT text: {text!r}")
                    response = get_llm_response(text)

                    # Analyze emotion and adjust exaggeration dynamically
                    reply_text = response or ""
                    if not reply_text.strip():
                        print("[LucyVoice] Respuesta vacía del LLM / agencia, usando fallback de aclaración.")
                        reply_text = "No entendí bien lo que querías que haga, ¿podés repetirlo?"

                    dynamic_exaggeration = analyze_emotion(reply_text)

                    # Use lower cfg_weight for more expressive responses
                    dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight

                    sample_rate, audio_array = tts.long_form_synthesize(
                        reply_text,
                        audio_prompt_path=args.voice,
                        exaggeration=dynamic_exaggeration,
                        cfg_weight=dynamic_cfg
                    )

                _log(f"[LucyVoice] Assistant spoken: {reply_text!r}")
                console.print(f"Assistant: {reply_text}")
                console.print(f"[dim](Emotion: {dynamic_exaggeration:.2f}, CFG: {dynamic_cfg:.2f})[/dim]")

                # Save voice sample if requested
                if args.save_voice:
                    response_count += 1
                    filename = f"voices/response_{response_count:03d}.wav"
                    tts.save_voice_sample(reply_text, filename, args.voice)
                    console.print(f"[dim]Voice saved to: {filename}[/dim]")

                play_audio(sample_rate, audio_array)
            else:
                console.print("[red]No se detectó voz. Escuchando nuevamente...[/red]")

    except KeyboardInterrupt:
        console.print("\n[red]Saliendo del asistente de voz...")

    console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")
