# Lucy Notes – Mimic3 integration

## Resumen de cambios
- chatterbox-tts reemplazado por `Mimic3TTS` en `tts.py`, con voz por defecto `es_ES/m-ailabs_low` y re-muestreo a `sample_rate` configurado.
- Configuración por defecto en castellano vía `config.yaml` (`stt.language=es`, `stt.task=transcribe`, `tts.sample_rate=16000`).
- `app.py` ahora carga config, fuerza Whisper a español, usa Ollama local (`base_url` configurable) y reproduce audio con Mimic3.
- Dependencias ajustadas: se elimina chatterbox-tts y se declara mimic3-tts en `requirements.txt` y `pyproject.toml`.

## Integración futura con Lucy
- Este módulo puede actuar como **frontend de voz**: micrófono → Whisper → Ollama → Mimic3. El texto reconocido (`transcribe()`) o la respuesta LLM (`get_llm_response()`) pueden enrutar-se al pipeline de herramientas de Lucy.
- Compartir voz TTS con Lucy: usar el mismo `voice` que en `Proyecto-VSCode` (ej. `es_ES/m-ailabs_low`) para consistencia de timbre.
- Se puede sustituir la reproducción directa (`play_audio`) por un envío de frames hacia Pipecat o por un callback que entregue texto/PCM a Lucy Voice.
- Mantener el lock/serialización de turnos si se conecta en paralelo a otros frontends para evitar respuestas superpuestas.
