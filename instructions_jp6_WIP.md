# NVIDIA Jetson AGX Orin Setup Guide

This comprehensive guide provides step-by-step instructions for setting up the NVIDIA Jetson AGX Orin with essential software components, including JetPack 6, NoMachine, Jetson Containers, Nilecam81 drivers, NVIDIA Riva, and the EGG Orchestrator system. By following these instructions, you will enable advanced AI functionalities, remote access, and robust audio capabilities on your Jetson platform.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install JetPack 6](#2-install-jetpack-6)
3. [Install NoMachine](#3-install-nomachine)
4. [Install Jetson Containers](#4-install-jetson-containers)
5. [Install Nilecam81 Drivers](#5-install-nilecam81-drivers)
6. [Install NVIDIA Riva 2.16.0](#6-install-nvidia-riva-2160)
7. [Configure Default Audio Devices](#7-configure-default-audio-devices)
8. [Download and Install GLaDOS TTS](#8-download-and-install-glados-tts)
9. [Run Ollama Jetson Container](#9-run-ollama-jetson-container)
10. [Set Up EGG Orchestrator System](#10-set-up-egg-orchestrator-system)
11. [Conclusion](#11-conclusion)

---

## 1. Prerequisites

Before beginning the installation process, ensure you have the following:

- **Host Linux Machine**: A Linux machine with more than 100 GB of available storage.
- **Jetson SDK Manager**: Downloaded from [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack).
- **USBA-USBC Cable**: Provided with the Jetson AGX Orin for firmware connection.
- **Internet Connection**: Required for downloading necessary packages and dependencies.
- **Sudo Privileges**: Ensure you have administrative rights on the host machine.

---

## 2. Install JetPack 6

JetPack 6 provides the necessary SDK for the Jetson AGX Orin. Follow these steps to install it:

### 2.1. Download JetPack 6

1. Visit the [JetPack 6 Download Page](https://developer.nvidia.com/embedded/jetpack) and download the SDK Manager suitable for your host Linux machine.

### 2.2. Install Jetson SDK

1. Launch the SDK Manager:

   ```bash
   sudo ./sdkmanager
   ```

2. Follow the on-screen instructions to install JetPack 6.
   
   **Important**: Ensure that you **save and build all firmware on the same drive** to avoid installation issues.

### 2.3. Connect Jetson AGX Orin

1. **Connect the Cable**:
   - Use the provided USBA-USBC cable to connect the Jetson AGX Orin to the host machine for firmware updates.

2. **Enter DFU Mode**:
   - Press and hold the two rightmost buttons on the Jetson AGX Orin.
   - While holding these buttons, apply power to the device.
   - Release the rightmost button but continue holding the center button until DFU mode is activated.

---

## 3. Install NoMachine

NoMachine allows remote access to your Jetson AGX Orin, enabling a headless setup.

### 3.1. Download NoMachine

- Download the ARM64 version of NoMachine from the [official website](https://downloads.nomachine.com/download/?id=115&distro=ARM).

### 3.2. Install NoMachine

```bash
sudo dpkg -i nomachine_8.14.2_1_arm64.deb
```

### 3.3. Configure NoMachine for Audio

1. **Disable Host NX Audio Option**:
   - Navigate to **Server Settings** > **Devices** in the NoMachine interface.
   - Disable the NX audio option to enable audio streaming and microphone forwarding.

---

## 4. Install Jetson Containers

Jetson Containers facilitate the use of Docker containers optimized for NVIDIA Jetson devices.

### 4.1. Clone the Jetson Containers Repository

```bash
git clone https://github.com/dusty-nv/jetson-containers
```

### 4.2. Install Container Tools

```bash
bash jetson-containers/install.sh
```

### 4.3. Pull and Run a Container

Automatically pull and run a container (e.g., PyTorch):

```bash
jetson-containers run $(autotag l4t-pytorch)
```

---

## 5. Install Nilecam81 Drivers (JP6)

The Nilecam81 drivers are essential for camera functionality.

### 5.1. Access Developer Portal

- **E-Con Systems Developer Portal**: Access your account to download the necessary drivers from [Nilecam81 Drivers](https://github.com/robit-man/EGG/tree/main/drivers/JP6FWNC81).

### 5.2. Extract and Install Firmware

1. **Extract Firmware**:
   - Extract the downloaded firmware to a designated folder on your system.

2. **Install Binaries**:

   ```bash
   sudo bash install_binaries.sh 81
   ```

### 5.3. Expose Camera Stream

Choose one of the following `gst-launch-1.0` commands based on your camera orientation:

- **Clockwise Rotated**:

  ```bash
  gst-launch-1.0 v4l2src device=/dev/video0 ! \
      videoconvert ! \
      videoflip method=clockwise ! \
      jpegenc ! \
      multifilesink location=/tmp/latest_frame_0.jpg
  ```

- **Not Rotated**:

  ```bash
  gst-launch-1.0 v4l2src device=/dev/video0 ! \
      videoconvert ! \
      jpegenc ! \
      multifilesink location=/tmp/latest_frame_0.jpg
  ```

- **Flipped (180 Degrees)**:

  ```bash
  gst-launch-1.0 v4l2src device=/dev/video0 ! \
      videoconvert ! \
      videoflip method=rotate-180 ! \
      jpegenc ! \
      multifilesink location=/tmp/latest_frame_0.jpg
  ```

### 5.4. Route Frames to Localhost

Use the provided script to route frames:

```bash
python3 http-gw.py
```

> **Note**: The script can be found [here](https://github.com/robit-man/EGG/blob/main/Shared/http-gw.py).

### 5.5. Run Frame Inference Test

Execute the Python script within a Jetson Container:

```bash
jetson-containers run -v /home/$(whoami)/Shared:/Shared  \
    -e HUGGINGFACE_TOKEN=<YOUR-HF-TOKEN> \
    $(autotag llama-vision) \
      python3 /Shared/frame-inference-test.py
```

> **Note**: Replace `<YOUR-HF-TOKEN>` with your actual Hugging Face token. The script is available [here](https://github.com/robit-man/EGG/blob/main/Shared/frame-inference-test.py).

---

## 6. Install NVIDIA Riva 2.16.0

NVIDIA Riva provides AI-powered conversational interfaces.

### 6.1. Download Riva Quickstart

- **Download via NGC Catalog**: Access the [Riva Quickstart ARM64](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) and download version 2.16.0.

### 6.2. Alternatively, Download via CLI

If you have the NGC CLI installed, execute:

```bash
ngc registry resource download-version nvidia/riva/riva_quickstart:2.16.0
```

### 6.3. Organize Riva Files

1. **Create RIVA Directory**:

   ```bash
   mkdir -p /home/$(whoami)/RIVA
   ```

2. **Place Downloaded Content**:
   - Extract and place all Riva files parallel to `riva_start.sh` in the `RIVA` folder, resulting in paths like `/home/$(whoami)/RIVA/riva_start.sh`.

### 6.4. Initialize Riva

```bash
sudo bash /home/$(whoami)/RIVA/riva_init.sh
```

### 6.5. Configure Docker Daemon

Sometimes `riva_start.sh` fails due to Docker daemon issues. Configure Docker to recognize NVIDIA as the default runtime.

1. **Edit Docker Daemon Configuration**:

   ```bash
   sudo gedit /etc/docker/daemon.json
   ```

2. **Add the Following Configuration**:

   ```json
   {
       "default-runtime": "nvidia",
       "runtimes": {
           "nvidia": {
               "path": "nvidia-container-runtime",
               "args": []
           }
       }
   }
   ```

3. **Restart Docker**:

   ```bash
   sudo systemctl restart docker
   ```

### 6.6. Start Riva

```bash
sudo bash /home/$(whoami)/RIVA/riva_start.sh
```

> **Note**: After starting Riva, proceed to the GLaDOS TTS Model installation in step 8. Ensure audio devices are correctly configured to prevent issues with NoMachine and virtual audio routing.

---

## 7. Configure Default Audio Devices

Proper audio device configuration is crucial for audio streaming and microphone functionalities.

### 7.1. Disable NoMachine Audio

1. **Edit NoMachine Configuration**:

   ```bash
   sudo gedit /usr/NX/etc/node.cfg
   ```

2. **Disable Audio**:
   - Locate the line `EnableAudio` and set it to `0`:

     ```ini
     EnableAudio=0
     ```

3. **Restart NoMachine Server**:

   ```bash
   sudo /etc/NX/nxserver --restart
   ```

### 7.2. Set Default Sink and Source

1. **List Current Audio Sinks and Sources**:

   - **List Sinks**:

     ```bash
     pactl get-default-sink
     ```

   - **List Sources**:

     ```bash
     pactl get-default-source
     ```

2. **Configure PulseAudio Defaults**:

   ```bash
   sudo gedit /etc/pulse/default.pa
   ```

   - Add the following lines to prevent automatic switching and set default devices (adjust device names as necessary):

     ```ini
     ### Disable module-switch-on-connect
     unload-module module-switch-on-connect

     # Set default sink and source
     set-default-sink alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo
     set-default-source alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input
     ```

3. **Restart PulseAudio**:

   ```bash
   pulseaudio -k
   pulseaudio --start
   ```

### 7.3. Handle NoMachine Audio Persistence

If NoMachine interferes with audio device settings, create a script to enforce defaults.

1. **Create Reset Script**:

   ```bash
   gedit /home/$(whoami)/reset_pulseaudio.sh
   ```

   - Add the following content:

     ```bash
     #!/bin/bash
     while true; do
       pactl set-default-source alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input
       sleep 5
     done
     ```

2. **Make the Script Executable**:

   ```bash
   chmod +x /home/$(whoami)/reset_pulseaudio.sh
   ```

3. **Add to Startup Applications**:
   - Open **Startup Applications** GUI.
   - Add the following line to execute the script on startup:

     ```bash
     /home/$(whoami)/reset_pulseaudio.sh
     ```

---

## 8. Download and Install GLaDOS TTS

GLaDOS TTS provides a text-to-speech model for conversational AI applications. Place the .riva files in the RIVA folder (parallel to riva_init.sh).

### 8.1. Download GLaDOS TTS Files

- Access and download the necessary files from the [GLaDOS_TTS Repository](https://huggingface.co/DavesArmoury/GLaDOS_TTS/tree/main).

### 8.2. Initialize Riva for GLaDOS TTS

1. **Copy TTS Models to Riva Artifacts**:

   ```bash
   sudo cp glados_hifigan.riva glados_fastpitch.riva /home/$(whoami)/RIVA/artifacts/
   ```

2. **Run Riva Speech Service Container**:

   ```bash
   sudo docker run --runtime=nvidia -it --rm \
       -v /home/$(whoami)/RIVA/artifacts:/servicemaker-dev \
       -v /home/$(whoami)/RIVA/riva_repo:/data \
       --entrypoint="/bin/bash" \
       nvcr.io/nvidia/riva/riva-speech:2.13.1-servicemaker-l4t-aarch64
   ```

3. **Build Speech Synthesis Models**:
   
   Inside the Docker container, execute:

   ```bash
   riva-build speech_synthesis \
       /servicemaker-dev/glados.rmir:tlt_encode \
       /servicemaker-dev/glados_fastpitch.riva:tlt_encode \
       /servicemaker-dev/glados_hifigan.riva:tlt_encode \
       --voice_name=GLaDOS \
       --sample_rate=22050
   ```

4. **Deploy Riva Models**:

   ```bash
   riva-deploy /servicemaker-dev/glados.rmir:tlt_encode /data/models
   ```

5. **Copy Models to Repository**:

   ```bash
   sudo cp -r /home/$(whoami)/RIVA/riva_repo/models/* /home/$(whoami)/RIVA/model_repository/models/
   ```

---

## 9. Run Ollama Jetson Container

Ollama provides advanced AI models that can be utilized within Jetson Containers.

### 9.1. Add Ollama to Startup Programs

1. **Open Startup Applications** GUI.

2. **Add a New Startup Program** with the following command:

   ```bash
#!/bin/bash

# Sudo Privaledge Caching
/home/$(whoami)/Startup/dummy_script.sh

# Ollama Jetson Container
jetson-containers run --name ollama $(autotag ollama)

   ```

   - This command launches the Ollama container, making it accessible over port `11434` during runtime.

---

## 10. Set Up EGG Orchestrator System

The **EGG Orchestrator** system is a comprehensive framework designed to facilitate seamless communication and data processing between various peripherals within a networked environment. It leverages modular components to handle specific tasks such as speech recognition, text-to-speech synthesis, and data routing. The core components of the EGG system include the **Orchestrator**, **ASR Engine**, **TTS Engine**, and **SLM Engine**. Each component operates independently while interacting cohesively to provide a robust and scalable solution for real-time data processing and peripheral management.

### 10.1. Overview of Components

- **Orchestrator**: Central hub managing interactions between peripherals.
- **ASR Engine**: Automatic Speech Recognition for converting spoken language into text.
- **TTS Engine**: Text-to-Speech synthesis for generating spoken language from text.
- **SLM Engine**: Speech Language Model for processing and routing data between ASR and TTS Engines.

### 10.2. Setting Up the Orchestrator

Follow these steps to set up the EGG Orchestrator system:

1. **Navigate to Orchestrator Directory**:

   ```bash
   cd /path/to/orchestrator
   ```

   > **Note**: Replace `/path/to/orchestrator` with the actual path where the Orchestrator folder is located.

2. **Set Up Each Component**:

   For each subfolder (`orch`, `slm`, `asr`, `tts`), execute the corresponding Python script.

   - **Orchestrator**:

     ```bash
     cd orch
     python3 orch.py
     ```

   - **SLM Engine**:

     ```bash
     cd ../slm
     python3 slm.py
     ```

   - **ASR Engine**:

     ```bash
     cd ../asr
     python3 asr.py
     ```

   - **TTS Engine**:

     ```bash
     cd ../tts
     python3 tts.py
     ```

3. **Automate Startup (Optional)**

   To ensure that the orchestrator and its components start automatically on system boot, you can create a systemd service or add scripts to the startup applications.

   **Example using systemd**:

   - **Create Service File**:

     ```bash
     sudo gedit /etc/systemd/system/egg-orchestrator.service
     ```

     - **Add the Following Content**:

       ```ini
       [Unit]
       Description=EGG Orchestrator Service
       After=network.target

       [Service]
       Type=simple
       ExecStart=/usr/bin/python3 /path/to/orchestrator/orch/orch.py
       ExecStartPost=/usr/bin/python3 /path/to/orchestrator/slm/slm.py
       ExecStartPost=/usr/bin/python3 /path/to/orchestrator/asr/asr.py
       ExecStartPost=/usr/bin/python3 /path/to/orchestrator/tts/tts.py
       Restart=on-failure
       User=your_username

       [Install]
       WantedBy=multi-user.target
       ```

     > **Note**: Replace `/path/to/orchestrator` with the actual path and `your_username` with your actual username.

   - **Enable and Start the Service**:

     ```bash
     sudo systemctl enable egg-orchestrator.service
     sudo systemctl start egg-orchestrator.service
     ```

4. **Verify Operation**

   Ensure that all components are running correctly by checking their logs or status.

   ```bash
   sudo systemctl status egg-orchestrator.service
   ```

   Additionally, monitor the terminal outputs of each script to verify successful initialization and communication between components.

### 10.3. System Overview

Below is a high-level diagram illustrating the interaction between the EGG system's components:

```plaintext
+----------------+        +-----------------+        +----------------+        +----------------+
|                |        |                 |        |                |        |                |
|  ASR Engine    +------->+  Orchestrator   +------->+  SLM Engine    +------->+  TTS Engine    |
|                |        |                 |        |                |        |                |
+----------------+        +-----------------+        +----------------+        +----------------+
        ^                         ^                          ^                         ^
        |                         |                          |                         |
        +-------------------------+--------------------------+-------------------------+
```

- **ASR Engine** captures audio, transcribes it to text, and sends it to the **Orchestrator**.
- **Orchestrator** routes the transcribed text to the **SLM Engine** for processing.
- **SLM Engine** generates responses based on language models and sends them back to the **Orchestrator**.
- **Orchestrator** routes the response to the **TTS Engine** for speech synthesis and output.

### 10.4. Additional Configuration

For detailed configuration and operational instructions, refer to the respective README files for each component:

- [Orchestrator README](https://github.com/robit-man/EGG/blob/main/Orchestrator/orch/README.md)
- [ASR Engine README](https://github.com/robit-man/EGG/blob/main/Orchestrator/asr/README.md)
- [TTS Engine README](https://github.com/robit-man/EGG/blob/main/Orchestrator/tts/README.md)
- [SLM Engine README](https://github.com/robit-man/EGG/blob/main/Orchestrator/slm/README.md)

---

## 11. Conclusion

By following this guide, you have successfully set up your NVIDIA Jetson AGX Orin with essential software components, drivers, and configurations. This setup enables advanced AI functionalities, remote access, robust audio capabilities, and a modular orchestrator system, paving the way for developing sophisticated applications on the Jetson platform.

For further assistance or troubleshooting, refer to the official [NVIDIA Developer Documentation](https://developer.nvidia.com/embedded/jetson-documentation) or consult the respective tool's support resources.

---

## Additional Resources

- **EGG Orchestrator System Overview**: For a comprehensive understanding of the EGG Orchestrator system, including detailed descriptions of each component and their interactions, refer to the [EGG Orchestrator Documentation](https://github.com/robit-man/EGG/blob/main/Orchestrator/README.md).

- **NVIDIA Developer Forums**: Engage with the community and seek help at the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/).

- **GitHub Repositories**:
  - [Jetson Containers](https://github.com/dusty-nv/jetson-containers)
  - [EGG Project](https://github.com/robit-man/EGG)
