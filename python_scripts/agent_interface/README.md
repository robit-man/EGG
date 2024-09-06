# TTS, ASR, and an LLM in between

1. Install Riva and nano_llm in the intro read-me.
2. Download and incorporate the pretrained TTS Voice Model from [DavesArmory Huggingface Repo](https://huggingface.co/DavesArmoury/GLaDOS_TTS)
3. Run all 3 python scripts Via the following bash scripts, for the TTS, ASR, and LLM components

## Run TTS Service
add the following to startup scripts, or convert to a bash script
![image](https://github.com/user-attachments/assets/426b239f-581a-4376-949c-4d57597abcfa)
```gnome-terminal -- bash -c "cd container_shared && python3 tts_chunk.py"```

## Run ASR Service
```gnome-terminal -- bash -c "cd container_shared && python3 asr_echo_check.py"```
