# Interaction Scripts

## Input.py

This script sends textual data to the LLM handler, this script can interrupt the inference and tts generation

```
python3 input.py
```

## Inference Server

This script interacts with the ollama api as well as the piper inference server we run inside of the jetson container, this script can have current inference interrupted upon new input, which kills the current thread and starts a new one to provide ultra low latency responses

```
python3 model_to_tts.py --history chat.json --stream
```
