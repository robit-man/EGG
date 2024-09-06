# 🥚 EGG
![egg_transparent](https://github.com/user-attachments/assets/58ca5637-7819-4e6d-8d7b-121a936afb14)

A multi-modal data acquisition tool based on an [AGX ORIN](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) with peripherals!

```
Scripts tested on the following system:
Jetson AGX ORIN 32BIT DEVKIT
L4T 35.2.1
JETPACK 5.1.2
CUDA 11.4.315
cuDNN 8.6.
PYTHON 3.8.10
```

## Setup and Remote Access
1. Determine the appropriate version of Jetpack based on hardware ([e-con systems for example requires 5.1~](https://www.e-consystems.com/nvidia-cameras/jetson-agx-orin-cameras/ar0821-4k-hdr-gmsl2-camera.asp)) and relative Host machine from [this Compatability matrix](https://docs.nvidia.com/sdk-manager/system-requirements/index.html).
2. [Install Ubuntu](https://ubuntu.com/download/desktop/thank-you?version=24.04.1&architecture=amd64&lts=true) on a host machine of the particular version which is compatable to then host Jetson SDK Manager which will be used to [install the desired Jetpack version](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html) on an AGX ORIN
3. Connect to a network and install [NOMACHINE](https://downloads.nomachine.com/download/?id=114&distro=ARM) for arm64 and use machine credentials to access on LAN or port forward!

## Install Dependencies
You must install RIVA for ASR / TTS Services or incorporate whisper for some scripts to function.

1. Install [RIVA docker service](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64), automate via bash script on startup
2. Install [jetson-containers](https://github.com/dusty-nv/jetson-containers) and [NANOLLM](https://dusty-nv.github.io/NanoLLM/install.html) by [dusty-nv](https://github.com/dusty-nv)
3. Install any missing dependencies and [provide console output in issues](https://github.com/robit-man/EGG/issues) for improving this readme!

## Start TTS / ASR LLM Services
Refer to the [agent_interface folder](https://github.com/robit-man/EGG/tree/main/python_scripts/agent_interface) to see how we automate the python scripts there.
