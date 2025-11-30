# Estado del nodo de voz modular de Lucy – 2025-11-29

## Entorno

- Repo remoto: https://github.com/LokoKanishka/nodo-de-voz-modular-de-lucy
- Ruta local: `~/Lucy_Workspace/nodo-de-voz-modular-de-lucy`
- Entorno virtual usado (compartido con Proyecto-VSCode):  
  `~/Lucy_Workspace/Proyecto-VSCode/.venv-lucy-voz`
- Python: 3.12.3

## Dependencias adicionales instaladas hoy (en la venv, no versionadas)

Instaladas dentro de `.venv-lucy-voz`:

- `openai-whisper-20250625`
- `torch-2.9.1` + stack CUDA 12 (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.)
- `langchain-core-1.1.0`
- `langchain-ollama-1.0.0`
- `ollama-0.6.1` (cliente Python)
- `rich-14.2.0`
- `nltk-3.9.2` (ya estaba, pero hoy se completaron los datos)

Estas dependencias NO están listadas todavía en un `requirements.txt` propio del repo, pero sí están documentadas acá.

## Datos NLTK descargados

Usando:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
