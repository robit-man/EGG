# 🥚 EGG Project

![Egg Transparent](https://github.com/user-attachments/assets/58ca5637-7819-4e6d-8d7b-121a936afb14)

## Overview

**EGG** is a multi-modal data acquisition, inference, storage, processing, and training/fine-tuning system deployed at the edge. Built on the [NVIDIA Jetson AGX Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) platform, EGG integrates various peripherals to deliver comprehensive AI capabilities.

Currently undergoing testing and refactoring on **JetPack 6**, with a focus on the [Orchestrator](https://github.com/robit-man/EGG/tree/main/Orchestrator) system and the integration of additional peripherals. For detailed instructions, refer to the [Installation Guide](https://github.com/robit-man/EGG/blob/main/instructions_jp6_WIP.md).


[![Download and Run Orchestrator](https://img.shields.io/badge/Download%20and%20Run%20Orchestrator-brightgreen?style=for-the-badge)](https://raw.githubusercontent.com/robit-man/EGG/main/install_and_run_orchestrator.sh)

### Tested Systems

#### [Orchestrator Scripts](https://github.com/robit-man/EGG/tree/main/Orchestrator)

- **Hardware**: Jetson AGX ORIN 32GB DEVKIT
- **Software Versions**:
  - L4T: 36.3.0
  - JetPack: 6.0 (rev 2)
  - CUDA: 12.2
  - cuDNN: 8.9.4.25
  - Python: 3.10.12

![Orchestrator Diagram](https://github.com/user-attachments/assets/e980b2e7-8d4b-4240-9e68-3d923a72f259)

#### [Agent Interface](https://github.com/robit-man/EGG/tree/main/python_scripts/agent_interface)

- **Version**: 5.1.2
- **Hardware**: Jetson AGX ORIN 32GB DEVKIT
- **Software Versions**:
  - L4T: 35.2.1
  - JetPack: 5.1.2
  - CUDA: 11.4.315
  - cuDNN: 8.6.*
  - Python: 3.8.10

---

## Table of Contents

1. [Acquire Parts and Construct](#acquire-parts-and-construct)
2. [Setup and Remote Access](#setup-and-remote-access)
3. [Install Dependencies](#install-dependencies)
4. [Experiment with the Orchestrator and Peripherals](#experiment-with-the-orchestrator-and-peripherals)
5. [Experiment with TTS / ASR LLM Services](#experiment-with-tts--asr-llm-services)
6. [Collaborations](#collaborations)
7. [Additional Resources](#additional-resources)
8. [Conclusion](#conclusion)

---

## 1. Acquire Parts and Construct

To build the EGG system, follow these steps:

1. **Order Components**: Refer to the [Bill of Materials](https://github.com/robit-man/EGG/blob/main/hardware/README.md) to order all necessary parts.
2. **3D Print Parts**: Print all components available in the [egg-parts.stp file](https://github.com/robit-man/EGG/blob/main/hardware/egg-parts.stp).
3. **Assembly**: Construct the system following the [Assembly Guide](https://github.com/robit-man/EGG/blob/main/hardware/Assembly_Guide.md) *(Work in Progress)*.

---

## 2. Setup and Remote Access

### 2.1. Determine JetPack Compatibility

- Consult the [Compatibility Matrix](https://docs.nvidia.com/sdk-manager/system-requirements/index.html) to select the appropriate JetPack version based on your hardware and host machine.

### 2.2. Install Ubuntu

1. **Download Ubuntu**:
   - Obtain [Ubuntu 24.04 LTS](https://ubuntu.com/download/desktop/thank-you?version=24.04.1&architecture=amd64&lts=true).

2. **Install on Host Machine**:
   - Install Ubuntu on a compatible host machine that will run the Jetson SDK Manager.

### 2.3. Install JetPack

1. **Download JetPack**:
   - Access the [JetPack 6 Download Page](https://developer.nvidia.com/embedded/jetpack) and download the SDK Manager suitable for your host Linux machine.

2. **Install JetPack**:
   - Launch the SDK Manager:
     ```bash
     sudo ./sdkmanager
     ```
   - Follow the on-screen instructions to install JetPack 6.
   - **Important**: Ensure that you **save and build all firmware on the same drive** to avoid installation issues.

### 2.4. Set Up Remote Access with NoMachine

1. **Download NoMachine**:
   - Obtain the ARM64 version of NoMachine from the [official website](https://downloads.nomachine.com/download/?id=114&distro=ARM).

2. **Install NoMachine**:
   ```bash
   sudo dpkg -i nomachine_8.14.2_1_arm64.deb
   ```

3. **Configure NoMachine for Audio**:
   - Open NoMachine settings and navigate to **Server Settings** > **Devices**.
   - Disable the NX audio option to enable audio streaming and microphone forwarding.

4. **Access the Device Remotely**:
   - Use machine credentials to access the Jetson AGX Orin on the LAN or via port forwarding.

---

## 3. Install Dependencies

EGG relies on various software dependencies to function correctly. Follow these steps to install them:

### 3.1. Install NVIDIA RIVA

1. **Download RIVA Quickstart**:
   - Access the [Riva Quickstart ARM64](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart_arm64) and download version 2.16.0.

2. **Alternatively, Download via CLI**:
   ```bash
   ngc registry resource download-version nvidia/riva/riva_quickstart:2.16.0
   ```

3. **Organize Riva Files**:
   ```bash
   mkdir -p /home/$(whoami)/RIVA
   ```
   - Extract and place all Riva files in the `RIVA` folder, resulting in paths like `/home/$(whoami)/RIVA/riva_start.sh`.

4. **Initialize Riva**:
   ```bash
   sudo bash /home/$(whoami)/RIVA/riva_init.sh
   ```

5. **Configure Docker Daemon**:
   - Edit the Docker daemon configuration to recognize NVIDIA as the default runtime:
     ```bash
     sudo gedit /etc/docker/daemon.json
     ```
   - Add the following content:
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
   - Restart Docker:
     ```bash
     sudo systemctl restart docker
     ```

6. **Start Riva**:
   ```bash
   sudo bash /home/$(whoami)/RIVA/riva_start.sh
   ```

### 3.2. Install Custom Voice Models

- For example, install [GLaDOS TTS](https://huggingface.co/DavesArmoury/GLaDOS_TTS).

### 3.3. Install Jetson Containers and NANOLLM

1. **Install Jetson Containers**:
   ```bash
   git clone https://github.com/dusty-nv/jetson-containers
   bash jetson-containers/install.sh
   ```

2. **Install NANOLLM**:
   - Follow the installation instructions provided by [dusty-nv](https://github.com/dusty-nv).

### 3.4. Install Additional Dependencies

- Install any missing dependencies as needed.
- If issues arise, [provide console output](https://github.com/robit-man/EGG/issues) to improve this documentation.
- Refer to [JetsonHacks](https://jetsonhacks.com/2023/09/04/use-these-jetson-docker-containers-tutorial/) for helpful resources.

---

## 4. Experiment with the Orchestrator and Peripherals

Explore the [Orchestrator System](https://github.com/robit-man/EGG/tree/main/Orchestrator) to manage and integrate peripherals.

### 4.1. Overview of Components

- **Orchestrator**: Central hub managing interactions between peripherals.
- **ASR Engine**: Automatic Speech Recognition for converting spoken language into text.
- **TTS Engine**: Text-to-Speech synthesis for generating spoken language from text.
- **SLM Engine**: Speech Language Model for processing and routing data between ASR and TTS Engines.

### 4.2. Setting Up the Orchestrator

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

   To ensure that the orchestrator and its components start automatically on system boot, create a systemd service or add scripts to the startup applications.

   **Example using systemd**:

   - **Create Service File**:
     ```bash
     sudo nano /etc/systemd/system/egg-orchestrator.service
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

---

## 5. Experiment with TTS / ASR LLM Services

Utilize the [Agent Interface](https://github.com/robit-man/EGG/tree/main/python_scripts/agent_interface) to automate Python scripts for TTS and ASR services.

### 5.1. Overview

The Agent Interface facilitates automated interactions between the TTS and ASR services, streamlining the workflow for speech recognition and synthesis tasks.

### 5.2. Running Agent Interface Scripts

1. **Navigate to Agent Interface Directory**:
   ```bash
   cd /path/to/agent_interface
   ```
   > **Note**: Replace `/path/to/agent_interface` with the actual path.

2. **Execute Automation Scripts**:
   ```bash
   python3 agent_interface.py
   ```
   - Ensure that all necessary dependencies and environment variables are set before running the scripts.

### 5.3. Configuration

- Refer to the [Agent Interface Documentation](https://github.com/robit-man/EGG/tree/main/python_scripts/agent_interface/README.md) for detailed configuration and usage instructions.

---

## 6. Collaborations

EGG is a collaborative effort involving hardware and software partners to enhance its capabilities.

- **Fractional Robots**
- **Hyperspawn**: [GitHub](https://github.com/Hyperspawn)
- **RokoNetwork**: [RokoNetwork on X](https://x.com/RokoNetwork)
  ![Roko-AI Logo](https://github.com/user-attachments/assets/c0e19c4f-6c3b-461c-9866-937424b12c3e)
- **Fractional Robots Telegram Group**:
  ![Group 2](https://github.com/robit-man/dropbear-neck-assembly/assets/36677806/bd13c6f5-7a3f-4262-9891-4259f17abbe0)
  [Join the Group](https://t.me/fractionalrobots)

---

## 7. Additional Resources

- **EGG Orchestrator System Overview**: For a comprehensive understanding of the EGG Orchestrator system, including detailed descriptions of each component and their interactions, refer to the [EGG Orchestrator Documentation](https://github.com/robit-man/EGG/blob/main/Orchestrator/README.md).
- **NVIDIA Developer Forums**: Engage with the community and seek help at the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/).
- **GitHub Repositories**:
  - [Jetson Containers](https://github.com/dusty-nv/jetson-containers)
  - [EGG Project](https://github.com/robit-man/EGG)
- **Helpful Tutorials**:
  - [JetsonHacks Docker Containers Tutorial](https://jetsonhacks.com/2023/09/04/use-these-jetson-docker-containers-tutorial/)
