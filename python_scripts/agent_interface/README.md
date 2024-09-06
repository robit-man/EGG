# TTS, ASR, and an LLM in between

1. Install [Riva](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) and [nano_llm](https://dusty-nv.github.io/NanoLLM/install.html) in the intro [readme](https://github.com/robit-man/EGG/blob/main/README.md).
2. Download and incorporate the [pretrained TTS Voice Model](https://github.com/davesarmoury/GLaDOS?tab=readme-ov-file) from [DavesArmory Huggingface Repo](https://huggingface.co/DavesArmoury/GLaDOS_TTS)
3. Run all 3 python scripts Via the following bash scripts, for the TTS, ASR, and LLM components

## Run TTS Service
add the following to startup scripts, or convert to a bash script

![image](https://github.com/user-attachments/assets/426b239f-581a-4376-949c-4d57597abcfa)

```gnome-terminal -- bash -c "cd container_shared && python3 tts_chunk.py"```

## Run ASR Service

![image](https://github.com/user-attachments/assets/8f3b9209-89bf-425d-bf9a-be60ddd238a8)

```gnome-terminal -- bash -c "cd container_shared && python3 asr_echo_check.py"```

## Run LLM Service

![image](https://github.com/user-attachments/assets/bc45a3c5-671a-4fa1-b9f2-73e66bc7ae9c)

```gnome-terminal -- bash -c "/home/roko/run_default.sh; exec bash"```
