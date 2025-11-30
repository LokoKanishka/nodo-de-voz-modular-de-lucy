import argparse
import os
import threading
import time
from queue import Queue

import numpy as np
import sounddevice as sd
import whisper
import yaml
from langchain_core.chat_history import InMemoryChatMessageHistory
# Updated imports for modern LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM
from rich.console import Console

from tts import TextToSpeechService

console = Console()

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
    ("system", "You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize LLM
llm = OllamaLLM(model=args.model, base_url=config["llm"].get("base_url", "http://localhost:11434"))

# Create the chain with modern LCEL syntax
chain = prompt_template | llm

# Chat history storage
chat_sessions = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]

# Create the runnable with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


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


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    # Use a default session ID for this simple voice assistant
    session_id = "voice_assistant_session"

    # Invoke the chain with history
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )

    # The response is now a string from the LLM, no need to remove "Assistant:" prefix
    # since we're using a proper chat model setup
    return response.strip()


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
    record_duration = 8  # seconds per capture to keep the flow hands-free

    console.print("🎤 Presioná Enter una vez para empezar a hablar (Ctrl+C para salir).")
    console.input("")  # Single prompt to kick off the session

    try:
        while True:
            console.print("[cyan]Escuchando...[/cyan]")

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            time.sleep(record_duration)
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="dots"):
                    text = transcribe(audio_np)

                if not text:
                    console.print("[red]Silencio detectado. Escuchando nuevamente...[/red]")
                    continue

                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="dots"):
                    response = get_llm_response(text)

                    # Analyze emotion and adjust exaggeration dynamically
                    dynamic_exaggeration = analyze_emotion(response)

                    # Use lower cfg_weight for more expressive responses
                    dynamic_cfg = args.cfg_weight * 0.8 if dynamic_exaggeration > 0.6 else args.cfg_weight

                    sample_rate, audio_array = tts.long_form_synthesize(
                        response,
                        audio_prompt_path=args.voice,
                        exaggeration=dynamic_exaggeration,
                        cfg_weight=dynamic_cfg
                    )

                console.print(f"[cyan]Assistant: {response}")
                console.print(f"[dim](Emotion: {dynamic_exaggeration:.2f}, CFG: {dynamic_cfg:.2f})[/dim]")

                # Save voice sample if requested
                if args.save_voice:
                    response_count += 1
                    filename = f"voices/response_{response_count:03d}.wav"
                    tts.save_voice_sample(response, filename, args.voice)
                    console.print(f"[dim]Voice saved to: {filename}[/dim]")

                play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Saliendo del asistente de voz...")

    console.print("[blue]Session ended. Thank you for using ChatterBox Voice Assistant!")
