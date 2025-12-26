import argparse
import json
import os
import re
import sys
import threading
from queue import Queue
from typing import Any, Optional

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
from lucy_web_agent import find_youtube_video_url

console = Console()
SAMPLE_RATE = 16000  # Hz, used for both VAD and Whisper


def _log(msg: str) -> None:
    print(msg, flush=True)

DEFAULT_CONFIG = {
    "tts": {
        "engine": "mimic3",
        "voice": "es_ES/m-ailabs_low#karen_savage",
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
parser.add_argument(
    "--voice",
    type=str,
    default=config["tts"]["voice"],
    help="Mimic3 voice id (e.g., es_ES/m-ailabs_low#karen_savage)",
)
parser.add_argument("--sample-rate", type=int, default=config["tts"]["sample_rate"], help="Output sample rate for TTS/audio playback")
parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration (0.0-1.0)")
parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight for pacing (0.0-1.0)")
parser.add_argument("--model", type=str, default=config["llm"]["model"], help="Ollama model to use")
parser.add_argument("--save-voice", action="store_true", help="Save generated voice samples")
parser.add_argument("--stt-model", type=str, default=config["stt"]["model"], help="Whisper model name (Spanish-capable)")
parser.add_argument("--stt-language", type=str, default=config["stt"]["language"], help="Target language for Whisper")
parser.add_argument("--stt-task", type=str, default=config["stt"]["task"], help="Whisper task: transcribe/translate")
parser.add_argument("--text", action="store_true", help="Modo texto: lee líneas por stdin (sin mic/tts)")
args = parser.parse_args()

# Initialize TTS with Mimic3 (default Spanish voice)
tts = TextToSpeechService(voice=args.voice, sample_rate=args.sample_rate)

# Spanish-first model; language enforced during transcribe to keep STT stable.
stt = whisper.load_model(args.stt_model)

LUCY_SYSTEM_PROMPT = """Sos Lucy, asistente de voz local en castellano rioplatense.
Corrés en la PC de Diego, offline, con acceso limitado a herramientas locales.
No controlás directo el navegador ni el escritorio: siempre usás las capas existentes.

Antes de llamarte, se evalúa lucy_agents.voice_actions para pedidos simples de escritorio
(por ejemplo, abrir Google o YouTube con una búsqueda). Si eso alcanza, ya se ejecuta
sin que vos intervengas.

Tu foco cuando intervenís:
- Responder de forma clara, cercana y respetuosa.
- Si necesitás usar herramientas, devolvé un JSON con las claves "name" y "arguments".
  * "name" puede ser "desktop_agent" o "web_agent".
  * Para desktop_agent: usá "arguments" con "command" (por ejemplo xdg-open + URL) o con "action":"close_window" y "window_title" con el título a cerrar.
  * Para web_agent: "arguments" incluye "kind":"youtube_latest", "query" breve y opcional "channel_hint".
- Herramientas disponibles (tools):
  1) desktop_agent:
     - Uso: acciones directas en el escritorio.
     - Ejemplos:
       - Abrir una URL en el navegador: name=desktop_agent, arguments con action=open_url y url con la dirección completa.
       - Cerrar una ventana: name=desktop_agent, arguments con action=close_window y window_title con el título de la ventana (ej. "YouTube").
  2) web_agent:
     - Uso: encontrar un video en YouTube a partir de una descripción.
     - Soporta: kind="youtube_latest", query="<búsqueda breve>", channel_hint="<canal/persona opcional>".
     - No habla con el usuario: devuelve una URL para que desktop_agent la abra.
     - Ejemplo de pedido y tool-call esperado:
       Usuario: "Buscá una entrevista en YouTube de Alejandro Dolina con Luis Navarro y reproducila."
       Tool-call: name=web_agent, arguments con kind=youtube_latest, query="Alejandro Dolina Luis Navarro entrevista", channel_hint="Alejandro Dolina".
- Para web_agent con kind="youtube_latest":
  * Resumí la frase del usuario a un query corto (3 a 8 palabras) solo con nombres propios y palabras clave como entrevista, programa, charla.
  * Ignorá frases de relleno y verbos tipo "quiero que", "buscá", "en YouTube", "dale play".
  * No copies toda la oración: ejemplo correcto → query: "Alejandro Dolina Luis Novaresio entrevista".
  * Si mencionan canal o programa, ponelo en channel_hint en forma corta (solo el nombre); no lo concatentes dentro del query.
  * Para pedidos de YouTube devolvé tool-call con name="web_agent", kind="youtube_latest", query breve y limpio, y channel_hint opcional solo con el nombre del canal.
- Si la acción requiere clicks/scroll dentro de una página, abrí igual la URL con desktop_agent
  y explicá que no podés interactuar adentro.
- No prometas acciones que no tengas herramienta para hacer.
- Si una herramienta falla, contalo en lenguaje natural.

Respondé breve (<=20 palabras) y cordial."""

# Modern prompt template using ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", LUCY_SYSTEM_PROMPT),
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


def _extract_tool_json(text: str) -> str | None:
    """
    Si el texto del LLM contiene únicamente un JSON (tool-call),
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


def _parse_tool_json(json_str: str) -> tuple[Optional[str], dict[str, Any]]:
    """Devuelve (name, args) o (None, {}) si falla el parseo."""
    try:
        data = json.loads(json_str)
    except Exception as exc:  # noqa: BLE001
        _log(f"[LucyVoice] Error parseando JSON de tool-call: {exc!r} - raw={json_str!r}")
        return None, {}

    if not isinstance(data, dict):
        _log(f"[LucyVoice] Tool-call JSON no es dict: {data!r}")
        return None, {}

    name = data.get("name")
    args = data.get("arguments") if isinstance(data.get("arguments"), dict) else data

    if not isinstance(args, dict):
        _log(f"[LucyVoice] Argumentos de tool-call no son dict: {args!r}")
        return name, {}

    if not name:
        if any(k in args for k in ("command", "url", "action")):
            name = "desktop_agent"
        elif "kind" in args:
            name = "web_agent"

    return name, args


def _handle_desktop_tool(args: dict[str, Any]) -> tuple[bool, str | None]:
    command = args.get("command")

    url = args.get("url")
    if not command and isinstance(url, str) and url.startswith(("http://", "https://")):
        command = f"xdg-open {url}"
        _log(f"[LucyVoice] desktop_agent URL detectada -> comando: {command!r}")
    elif url and not command:
        _log(f"[LucyVoice] URL inválida en desktop_agent: {url!r}")
        return True, "No pude abrir esa URL; verificá el formato."

    action = args.get("action")
    if not command and action == "close_window":
        window_title = args.get("window_title")
        if isinstance(window_title, str):
            safe_title = window_title.strip().replace('"', "")
            if safe_title:
                command = f'wmctrl -c "{safe_title[:120]}"'
                _log(f"[LucyVoice] desktop_agent close_window → {command}")
            else:
                _log("[LucyVoice] close_window sin window_title válido.")
                return True, "Decime qué ventana querés cerrar."
        else:
            _log("[LucyVoice] close_window sin window_title string.")
            return True, "Decime qué ventana querés cerrar."

    if not isinstance(command, str) or not command.strip():
        _log(f"[LucyVoice] Comando desktop_agent no soportado: {args!r}")
        return False, None

    _log(f"[LucyVoice] desktop_agent command: {command!r}")
    exit_code = run_desktop_command(command)
    _log(f"[LucyVoice] desktop_agent exit code: {exit_code}")

    if action == "close_window":
        spoken = "Cerré la ventana que pediste." if exit_code == 0 else "No pude cerrar la ventana; probá con el título exacto."
        return True, spoken

    spoken = "Listo, ya lo abrí en tu escritorio." if exit_code == 0 else "Intenté abrirlo, pero falló el comando en el escritorio."
    return True, spoken


def _handle_web_agent_tool(args: dict[str, Any]) -> tuple[bool, str | None]:
    kind = args.get("kind")
    if kind == "youtube_latest":
        query = args.get("query")
        channel_hint = args.get("channel_hint") if isinstance(args.get("channel_hint"), str) else None
        strategy = args.get("strategy") if isinstance(args.get("strategy"), str) else "latest"

        if not isinstance(query, str) or not query.strip():
            _log("[LucyVoice] web_agent youtube_latest sin query válida.")
            return True, "Necesito el nombre o búsqueda exacta para YouTube."

        _log(f"[LucyVoice] web_agent youtube_latest query={query!r} channel_hint={channel_hint!r} strategy={strategy!r}")
        url = find_youtube_video_url(query.strip(), channel_hint=channel_hint, strategy=strategy)
        if not url:
            _log("[LucyWebAgent] No se encontró URL de video.")
            return True, "No pude encontrar el video, probá con el nombre completo o revisá la conexión."

        _log(f"[LucyWebAgent] URL encontrada: {url}")
        exit_code = run_desktop_command(f"xdg-open {url}")
        _log(f"[LucyVoice] desktop_agent (via web_agent) exit code: {exit_code}")

        is_search = isinstance(url, str) and url.startswith("https://www.youtube.com/results?search_query=")
        if exit_code == 0:
            spoken = (
                "Abrí el video en tu navegador."
                if not is_search
                else "No encontré una entrevista exacta, pero te abrí la búsqueda en YouTube para que elijas."
            )
            return True, spoken
        return True, "Encontré el video pero no pude abrirlo; probá de nuevo."

    _log(f"[LucyVoice] web_agent kind no soportado: {kind!r}")
    return False, None



def _wants_playback_text(text: str) -> bool:
    low = (text or "").lower()
    markers = (
        "reproduc", "play",
        "poné", "pone", "ponlo", "ponelo", "poneme",
        "quiero verlo", "quiero ver", "quiero escucharlo", "quiero escuchar",
        "que se reproduzca", "dale play",
    )
    return any(m in low for m in markers)


def _is_youtube_search_request(text: str) -> bool:
    low = (text or "").lower()
    if ("youtube" not in low) and ("you tube" not in low):
        return False
    if not re.search(r"\bbusc", low):
        return False
    # si pide playback, no aplicamos guardrail
    if _wants_playback_text(low):
        return False
    return True


def _extract_youtube_search_query_from_text(text: str) -> str:
    """
    Extracción simple/determinista: sacamos wakeword + verbos + plataforma + coletillas,
    dejamos el objeto de búsqueda.
    """
    low = (text or "").lower()

    # wakeword
    low = re.sub(r"\blucy\b[:,]?\s*", " ", low)

    # verbos comunes
    low = re.sub(r"\b(abr[ií]r|abr[ií]|abre|busc[aá]r|busc[aá]|busca|busqu(?:e|es|en)|buscame|bokk?a)\b", " ", low)

    # plataforma
    low = re.sub(r"\b(en|por)?\s*(youtube|you\s*tube)\b", " ", low)
    low = re.sub(r"\b(youtube|you\s*tube)\b", " ", low)

    # coletillas típicas
    low = re.sub(r"\b(y\s+)?contame\b.*$", " ", low)
    low = re.sub(r"\b(y\s+)?breve(?:mente)?\b.*$", " ", low)
    low = re.sub(r"\b(y\s+)?resum\w*\b.*$", " ", low)

    # limpieza
    low = re.sub(r"[\"'`]", " ", low)
    low = re.sub(r"[^0-9a-záéíóúñü\s]+", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    return low


def _youtube_results_url(query: str) -> str:
    from urllib.parse import quote_plus
    q = (query or "").strip()
    return "https://www.youtube.com/results?search_query=" + quote_plus(q)


def _handle_tool_json(json_str: str, source: str = "inline") -> tuple[bool, str | None]:
    _log(f"[LucyVoice] Detecté tool-call JSON ({source}).")
    name, args = _parse_tool_json(json_str)
    if not name or not args:
        _log(f"[LucyVoice] Tool-call sin nombre/args utilizables: name={name!r} args={args!r}")
        return False, None

    if name == "desktop_agent":
        return _handle_desktop_tool(args)
    if name == "web_agent":
        return _handle_web_agent_tool(args)

    _log(f"[LucyVoice] Tool no soportada: {name!r}")
    return False, None


def get_llm_response(text: str) -> str:
    """
    Devuelve el texto final que Lucy le va a decir al usuario.

    Orden de trabajo:
      1) Intentar resolver con voice_actions (reglas + Desktop Agent).
      2) Si no, llamar al LLM (chain_with_history).
      3) Si el LLM devuelve un JSON de tool-call (desktop_agent / web_agent), procesarlo
         y devolver un resumen amigable.
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

    # 3) Si parece un JSON de tool-call, intentamos ejecutarlo.
    tool_json = _extract_tool_json(raw)
    if tool_json is not None:
        # Guardrail: si el LLM devuelve youtube_latest pero el usuario pidió BÚSQUEDA (no playback),
        # abrimos results?search_query=... y evitamos elegir un video incorrecto.
        try:
            _name, _args = _parse_tool_json(tool_json)
        except Exception:
            _name, _args = None, None

        if _name == "web_agent" and isinstance(_args, dict) and _args.get("kind") == "youtube_latest":
            if _is_youtube_search_request(text):
                q = _extract_youtube_search_query_from_text(text) or str(_args.get("query") or "")
                q = (q or "").strip()
                if q:
                    url = _youtube_results_url(q)
                    _log(f"[LucyVoice] Guardrail: youtube_latest -> open search url={url}")
                    rc = run_desktop_command(f"xdg-open {url}")
                    _log(f"[LucyVoice] desktop_agent (guardrail yt search) exit code: {rc}")
                    return "Te abrí la búsqueda en YouTube." if rc == 0 else "Intenté abrir la búsqueda en YouTube, pero falló el comando."
        handled, spoken = _handle_tool_json(tool_json, source="top-level")
        if handled:
            output_text = spoken or "Listo, ya lo abrí en tu escritorio."
            _log("[LucyVoice] Tool-call ejecutado a partir de JSON del LLM.")
            _log(f"[LucyVoice] get_llm_response() final spoken: {output_text!r}")
            return output_text
        fallback = "No pude ejecutar esa acción; probá pedírmelo de otra forma."
        _log("[LucyVoice] Tool-call detectado pero no ejecutado; respondo en texto.")
        _log(f"[LucyVoice] get_llm_response() final spoken: {fallback!r}")
        return fallback

    # 3b) Si el JSON viene dentro de bloques ```json ... ```, ejecutamos todos en orden.
    try:
        block_jsons = _extract_json_code_blocks(raw)
    except Exception as e:  # noqa: BLE001
        _log(f"[LucyVoice] Error extrayendo bloques JSON: {e}")
        block_jsons = []

    if block_jsons:
        _log(f"[LucyVoice] Detecté {len(block_jsons)} bloque(s) JSON para tool-calls.")
        actions_handled = 0
        last_spoken: str | None = None
        for block_json in block_jsons:
            try:
                _name, _args = _parse_tool_json(block_json)
            except Exception:
                _name, _args = None, None
            if _name == "web_agent" and isinstance(_args, dict) and _args.get("kind") == "youtube_latest":
                if _is_youtube_search_request(text):
                    q = _extract_youtube_search_query_from_text(text) or str(_args.get("query") or "")
                    q = (q or "").strip()
                    if q:
                        url = _youtube_results_url(q)
                        _log(f"[LucyVoice] Guardrail: youtube_latest -> open search url={url}")
                        rc = run_desktop_command(f"xdg-open {url}")
                        _log(f"[LucyVoice] desktop_agent (guardrail yt search) exit code: {rc}")
                        actions_handled += 1
                        last_spoken = "Te abrí la búsqueda en YouTube." if rc == 0 else "Intenté abrir la búsqueda en YouTube, pero falló el comando."
                        continue
            try:
                handled, spoken = _handle_tool_json(block_json, source="json_block")
                if handled:
                    actions_handled += 1
                    if spoken:
                        last_spoken = spoken
            except Exception as e:  # noqa: BLE001
                _log(f"[LucyVoice] Error manejando bloque JSON: {e}")
                continue

        cleaned_raw = _strip_json_code_blocks(raw).strip()
        raw = cleaned_raw

        _log(f"[LucyVoice] Handled {actions_handled} tool-call action(s) from JSON code blocks.")

        if actions_handled > 0:
            output_text = last_spoken or cleaned_raw or "Listo, ya abrí las búsquedas que me pediste en tu escritorio."
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
    _log("[LucyVoice] Sin tool-calls; respondo en texto plano.")
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
    if not args.text:
        _warmup_llm(chain_with_history)

    if args.text:
        console.print("[cyan]📝 Modo texto activo (stdin). Escribí una línea y Enter. Ctrl+D para salir.[/cyan]")
        try:
            while True:
                try:
                    text = console.input("> ")
                except EOFError:
                    break
                text = (text or "").strip()
                if not text:
                    continue
                _log(f"[LucyVoice] TEXT input: {text!r}")
                console.print(f"You: {text}")
                if is_sleep_command(text):
                    console.print("[cyan][Lucy] Recibí la orden 'lucy dormi'. Me voy a dormir y cierro la sesión.[/cyan]")
                    break
                response = get_llm_response(text)
                reply_text = response or ""
                if not reply_text.strip():
                    reply_text = "No entendí bien lo que querías que haga, ¿podés repetirlo?"
                _log(f"[LucyVoice] Assistant (text): {reply_text!r}")
                console.print(f"Assistant: {reply_text}")
        except KeyboardInterrupt:
            console.print("\n[red]Saliendo del asistente (modo texto)...")
        console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")
        raise SystemExit(0)

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
