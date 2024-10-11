# 🥚 EGG
![egg_transparent](https://github.com/user-attachments/assets/58ca5637-7819-4e6d-8d7b-121a936afb14)

A multi-modal data acquisition tool based on an [AGX ORIN](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) with peripherals!

```
Scripts tested on the following system:
Jetson AGX ORIN 32GB DEVKIT
L4T 35.2.1
JETPACK 5.1.2
CUDA 11.4.315
cuDNN 8.6.*
PYTHON 3.8.10
```
Currently Testing and Refactoring on JETPACK 6, [Read instructions here!](https://github.com/robit-man/EGG/blob/main/instructions_jp6_WIP.md)
```
Scripts tested on the following system:
Jetson AGX ORIN 32GB DEVKIT
L4T 36.3.0
JETPACK 6.0 (rev 2)
CUDA 12.2
cuDNN 8.9.4.25
PYTHON 3.10.12
```
## Acquire Parts and Construct
1. Order all parts in the [bill of materials](https://github.com/robit-man/EGG/blob/main/hardware/README.md)
2. Print all parts present in the [egg-parts.stp file](https://github.com/robit-man/EGG/blob/main/hardware/egg-parts.stp)
3. Construct According to the Assembly Guide (WIP)

## Setup and Remote Access
1. Determine the appropriate version of Jetpack based on hardware ([e-con systems for example requires 5.1~](https://www.e-consystems.com/nvidia-cameras/jetson-agx-orin-cameras/ar0821-4k-hdr-gmsl2-camera.asp)) and relative Host machine from [this Compatability matrix](https://docs.nvidia.com/sdk-manager/system-requirements/index.html).
2. [Install Ubuntu](https://ubuntu.com/download/desktop/thank-you?version=24.04.1&architecture=amd64&lts=true) on a host machine of the particular version which is compatable to then host Jetson SDK Manager which will be used to [install the desired Jetpack version](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html) on an AGX ORIN
3. Connect to a network and install [NOMACHINE](https://downloads.nomachine.com/download/?id=114&distro=ARM) for arm64 and use machine credentials to access on LAN or port forward!

## Install Dependencies
You must install RIVA for ASR / TTS Services or incorporate whisper for some scripts to function.

1. Install [RIVA docker service](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64), automate via bash script on startup
2. Install Custom Voice Models like [GLaDOS TTS](https://huggingface.co/DavesArmoury/GLaDOS_TTS)
3. Install [jetson-containers](https://github.com/dusty-nv/jetson-containers) and [NANOLLM](https://dusty-nv.github.io/NanoLLM/install.html) by [dusty-nv](https://github.com/dusty-nv)
4. Install any missing dependencies and [provide console output in issues](https://github.com/robit-man/EGG/issues) for improving this readme. you can also find helpful resources on [jetsonhacks](https://jetsonhacks.com/2023/09/04/use-these-jetson-docker-containers-tutorial/)!

## Experiment with the [Orchestator / Peripherals](https://github.com/robit-man/EGG/tree/main/Orchestrator) System

## Experiment with TTS / ASR LLM Services
Refer to the [agent_interface folder](https://github.com/robit-man/EGG/tree/main/python_scripts/agent_interface) to see how we automate the python scripts there.

# This Project is a Hardware / Software Collaboration with Fractional Robots, [Hyperspawn](https://github.com/Hyperspawn), and [RokoNetwork](https://x.com/RokoNetwork)[![roko-amll](https://github.com/user-attachments/assets/c0e19c4f-6c3b-461c-9866-937424b12c3e)](https://roko.network/)
[![Group 2](https://github.com/robit-man/dropbear-neck-assembly/assets/36677806/bd13c6f5-7a3f-4262-9891-4259f17abbe0)](https://t.me/fractionalrobots)

