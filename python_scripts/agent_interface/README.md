# TTS, ASR, and an LLM in between

1. Install [Riva](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) and [nano_llm](https://dusty-nv.github.io/NanoLLM/install.html) in the intro [readme](https://github.com/robit-man/EGG/blob/main/README.md).
2. Download and incorporate the [pretrained TTS Voice Model](https://github.com/davesarmoury/GLaDOS?tab=readme-ov-file) from [DavesArmory Huggingface Repo](https://huggingface.co/DavesArmoury/GLaDOS_TTS)
3. Run all 3 python scripts Via the following bash scripts, for the TTS, ASR, and LLM components

## Get system info

Check available audio devices
```aplay -l```

Update audio device [default sink and source](https://web.archive.org/web/20240906045018/https://askubuntu.com/questions/1038490/how-do-you-set-a-default-audio-output-device-in-ubuntu#1038492) to be persistent across reboots

Check system settings for correct default audio devices

## Run TTS Service
add the following to [startup applications](https://help.ubuntu.com/stable/ubuntu-help/startup-applications.html.en), or convert to a bash script and run it at startup with [this guide](https://web.archive.org/web/20240906044738/https://medium.com/@girishbhasinofficial/configuring-a-script-to-run-at-startup-on-ubuntu-22-04-ffe1f3e649d1)

![image](https://github.com/user-attachments/assets/426b239f-581a-4376-949c-4d57597abcfa)

```gnome-terminal -- bash -c "cd container_shared && python3 tts_chunk.py"```
Runs [tts_chunk.py](https://github.com/robit-man/EGG/blob/main/python_scripts/agent_interface/tts_chunk.py) 

## Run ASR Service

![image](https://github.com/user-attachments/assets/8f3b9209-89bf-425d-bf9a-be60ddd238a8)

```gnome-terminal -- bash -c "cd container_shared && python3 asr_echo_check.py"```
Runs [asr_echo_check.py](https://github.com/robit-man/EGG/blob/main/python_scripts/agent_interface/asr_echo_check.py)

## Run LLM Service

![image](https://github.com/user-attachments/assets/bc45a3c5-671a-4fa1-b9f2-73e66bc7ae9c)

```gnome-terminal -- bash -c "/home/roko/run_default.sh; exec bash"```
Runs a bash script called [run_default.sh](https://github.com/robit-man/EGG/blob/main/bash_scripts/run_default.sh) which handles [spinning up the LLM](https://github.com/robit-man/EGG/blob/main/python_scripts/agent_interface/llm_settings_demo.py) with permissions by way of the password cache from [dummy_script.sh](https://github.com/robit-man/EGG/blob/main/bash_scripts/dummy_script.sh)
