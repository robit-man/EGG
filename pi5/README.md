<img src="https://github.com/user-attachments/assets/42caf09c-1e7b-42bd-9255-e3171ebd5006" alt="Sm0l Egg" width="200"/>


# MINI EGG

### ASR > Ollama > TTS Pipeline for Pi5 - EXPERIMENTAL RELEASE!

## Setup and restart with one copy/paste
## (PLEASE REPORT BUGS, THERE ARE SOME ASSUMPTIONS OF HARDWARE BELOW)

```bash
mkdir -p /home/$(whoami)/voice && curl -fsSL https://raw.githubusercontent.com/robit-man/EGG/main/pi5/setup.sh -o /home/$(whoami)/voice/setup.sh && chmod +x /home/$(whoami)/voice/setup.sh && bash /home/$(whoami)/voice/setup.sh
```

The setup script now:
- upgrades packages (unless `EGG_SKIP_APT_UPGRADE=1`)
- syncs runtime scripts from `EGG_repo` to `/home/<user>/voice`
- installs watchdog + router + camera/pipeline services
- creates `~/.config/autostart/mini_egg_watchdog.desktop`
- enables watchdog auto-update from `https://github.com/robit-man/EGG.git`

## Teardown and cleanup with one copy/paste

```bash
curl -sSL https://raw.githubusercontent.com/robit-man/EGG/main/pi5/teardown.sh -o /tmp/egg_teardown.sh && chmod +x /tmp/egg_teardown.sh && bash /tmp/egg_teardown.sh && rm -f /tmp/egg_teardown.sh
```

## Runtime services and local endpoints

- `watchdog.py`: toggles services and keeps them alive
  - live per-service CPU and RAM usage in the service table
  - top-line heaviest/peak resource summary
  - `S` opens CPU throttle settings (governor/min/max/`over_voltage_delta`/auto-apply + apply-now)
  - default watchdog CPU targets are `min=1600000` and `max=1800000` kHz
  - default watchdog undervolt target is `over_voltage_delta=-25000` uV (firmware setting, reboot required after apply)
  - `S` also includes ASR-vs-LLM throttle controls (enable, percent, cycle-ms) to reduce ASR CPU during LLM generation
  - left/right arrows switch lower-pane tabs: `Logs` and `Resources`
  - `Resources` tab renders live side-by-side CPU% and RAM% history graphs plus a top-by-CPU service table
  - `C` copies selected service dashboard/link target, `L` logs all discovered service links
  - click `[copy]` in the service table (mouse-enabled terminals) to copy that row link
  - copied service links auto-include service session keys (`session_key=...`) when auth is enabled
  - watchdog pulls pipeline observability events and shows ASR/LLM/TTS progress + recognized text in log pane
  - LLM streaming events include active model, streamed token chunks, estimated token count, and tokens/sec telemetry
- `router.py`: NKN sidecar + persistent router address + remote tunnel discovery + terminal dashboard
- `camera_router.py`: camera list/snapshot/video (+ `/mjpeg` and `/jpeg`) routes + terminal dashboard
- `pipeline_api.py`: HTTP bridge for LLM prompt/TTS prompt + LLM model dashboard (`/llm/dashboard`) + terminal dashboard
  - includes `/pipeline/state` for live ASR -> LLM -> TTS stage/event observability
- `audio_router.py`: audio auth/device routing + `/llm/prompt` + `/tts/speak` + WebRTC offer route + terminal dashboard
- `output.py`, `model_to_tts.py`, `run_asr_stream.py`, `run_ollama_service.py`, `run_voice_server.py`: direct watchdog-managed background services
  - `model_to_tts.py` chunks responses on punctuation for rapid TTS playback (instead of waiting for full sentences only)
  - voice commands include a tool-command stack: thinking toggle, battery-context toggle, model switching, watchdog tuneables, TTS volume control, battery/SSID queries, and restart confirmation
  - dashboard model changes are hot-reloaded and preempt active inference; next ASR prompt runs on the new model
- `run_asr_stream.py`: Whisper stream -> LLM bridge
- `run_voice_server.py`: Docker voice server wrapper
- `run_ollama_service.py`: Ollama service wrapper
- Audio cues:
  - ASR capture cue: short high blip when transcript is captured and forwarded
  - LLM processing cue: short mid blip when LLM prompt processing begins
  - `audio_router.audio.tts_tail_silence_ms` (default `90`) adds post-TTS silence padding to reduce clipped phrase endings
- LLM runtime battery context:
  - `model_to_tts.py` reads UPS HAT INA219 battery telemetry (I2C bus `1`, addr `0x43`) and appends live battery voltage + estimated percent to the active system message
  - inclusion is toggleable at runtime via voice tool commands (default: enabled)

Local ports:
- `5070` router (`/health`, `/nkn/info`, `/nkn/resolve`, dashboard)
- `8080` camera router (`/auth`, `/list`, `/snapshot/cam0`, `/jpeg/cam0`, `/video/cam0`, `/mjpeg/cam0`)
- `6590` pipeline API (`/auth`, `/list`, `/health`, `/pipeline/state`, `/llm/prompt`, `/llm/dashboard`, `/llm/models`, `/llm/config`, `/llm/pull`, `/llm/pull/status`, `/tts/speak`)
- `8090` audio router (`/auth`, `/list`, `/devices`, `/devices/select`, `/llm/prompt`, `/tts/speak`, `/webrtc/offer`)
- `6545` model bridge (`model_to_tts.py`)
- `6434` voice server (`voice_server.py` via Docker)
- `6353` audio output (`output.py`)
- `11434` ollama (`ollama serve`)

All Flask services bind to `0.0.0.0` so you can use the Pi LAN IP from other devices.

Default service auth:
- `camera_router.py` password defaults to `egg`
- `pipeline_api.py` password defaults to `egg`
- `audio_router.py` password defaults to `egg`
- change in `camera_router_config.json`, `pipeline_api_config.json`, or `audio_router_config.json` (or from each service terminal UI under `Security`)

LLM dashboard quick start:

```bash
# 1) get a session key
SESSION_KEY=$(curl -sS -X POST "http://<PI_LAN_IP>:6590/auth" -H "Content-Type: application/json" -d '{"password":"egg"}' | python3 -c 'import sys,json; print(json.load(sys.stdin).get("session_key",""))')

# 2) open the dashboard in browser
echo "http://<PI_LAN_IP>:6590/llm/dashboard?session_key=${SESSION_KEY}"
```

From the dashboard you can:
- view available Ollama models
- set and save the default `llm_bridge_config.json` model (default: `qwen3:0.6b`)
- toggle `thinking` on/off (default: off)
- view and edit the active LLM `system` message
- toggle tool fallback, toggle per-tool visibility, and toggle ASR leading-`[` gate
- pull new Ollama models and watch pull status/logs

Voice tool toggle:
- saying `turn thinking on` or `turn thinking off` to the LLM bridge flips mode and sends immediate TTS feedback (`Turned Thinking On/Off`).

Voice tool commands (LLM bridge):
- `turn thinking on` / `turn thinking off`
- `turn battery context on` / `turn battery context off`
- `what is the battery state` / `battery voltage` / `battery percent`
- `what is the current ssid` / `what network am i on`
- `restart` (asks for confirmation) then `yes` or `no`
- `switch model ...` (with numbered follow-up confirmation)
- `set cpu min to 1600000`
- `set cpu max to 1800000`
- `set cpu range 1600000 to 1800000`
- `set undervolt to -25000`
- `set over voltage delta to -25000`
- `turn asr throttle on` / `turn asr throttle off`
- `set asr throttle percent to 65`
- `set asr throttle cycle to 320 ms`
- `set tts volume to 70 percent`
- `set tts volume to 7` (maps `1..10` to `10%..100%`)
- `volume 1` through `volume 10`
- spoken number forms are supported (examples: `volume one`, `volume ten percent`)
- `turn asr bracket gate on` / `turn asr bracket gate off`
- watchdog tuneable commands update `.watchdog_runtime/service_state.json`; running watchdog applies them automatically
- battery context defaults and settings live in `pi5/piper/llm_bridge_config.json`:
  - `tool_fallback_enabled` (LLM may emit `<tool>{...}</tool>` fallback calls; tool markup is filtered from TTS playback)
  - `tool_visibility` (per-tool enable/disable map shown in LLM dashboard)
  - `asr_leading_bracket_gate_enabled` (default `true`; drops ASR prompts starting with `[` before LLM inference)
  - `asr_tool_dedupe_enabled` (default `true`; suppresses duplicate loopback ASR tool/control commands)
  - `asr_tool_dedupe_window_seconds` (default `4.0`; duplicate window for suppression)
  - `battery_context_enabled`
  - `battery_i2c_bus` (default `1`)
  - `battery_i2c_addr` (default `67`, i.e. `0x43`)
  - `battery_refresh_seconds` (default `5.0`)

## Installation and Runtime

Purchase the following list of hardware components:

[Raspberry Pi 5 8GB](https://www.sparkfun.com/products/23551) $80

[256Gb Micro SD Memory Card](https://www.amazon.com/SanDisk-Extreme-microSDXC-Memory-Adapter/dp/B09X7C2GBC) $25

[Pi Active Cooling](https://www.sparkfun.com/products/23585) $5

[Pi Battery UPS](https://www.amazon.com/gp/product/B0D39VDMDP) $33

[2x 21700 Batteries](https://www.amazon.com/dp/B0CJ4J6B8Z) $20

[Mini Bluetooth Speaker / Mic](https://www.amazon.com/dp/B0BPNYY61M) $13

### Total Cost: $176


Install [Ollama](https://ollama.com/)
```
curl -fsSL https://ollama.com/install.sh | sh
```
Pull llama3.2:1b
```
ollama pull llama3.2:1b
```
Install [Flite](http://www.festvox.org/flite/) TTS
```
sudo apt install flite
```
Then Get the feedback.py file
```
curl -o ~/feedback.py https://raw.githubusercontent.com/robit-man/EGG/main/pi5/feedback.py
```
Or for LLAVA Support llava_support.py file
```
curl -o ~/feedback.py https://raw.githubusercontent.com/robit-man/EGG/main/pi5/llava_support.py
```
Now we automate the activation across reboots
```
mkdir -p ~/.config/autostart && echo -e "[Desktop Entry]\nType=Application\nName=Feedback Script\nExec=lxterminal -e python3 /home/egg/feedback.py\nStartupNotify=false\nTerminal=false" > ~/.config/autostart/feedback.desktop
```


## Whisper curl

```bash
curl -sSL https://raw.githubusercontent.com/robit-man/EGG/main/pi5/setup_bak.sh -o setup_bak.sh && bash setup_bak.sh || (echo "Using cached setup_bak.sh" && bash setup_bak.sh)
```

## Debugging

Set Full Send Bless Mode for max fan across reboots:

```
sudo nano /etc/systemd/system/fan_pwm.service
```

Then inject:
```
[Unit]
Description=Fan PWM Setup
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'sudo pigpiod && pinctrl FAN_PWM op dl'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Now we reload
```
sudo systemctl daemon-reload
```

And Enable
```
sudo systemctl enable fan_pwm.service
```

And start the service!
```
sudo systemctl start fan_pwm.service
```

Check if its running:
```
sudo systemctl status fan_pwm.service
```

To Reduce Power Consumption when halted:

```
sudo rpi-eeprom-config -e
```

Change:
```
POWER_OFF_ON_HALT=1
```

You will likely run into undercurrent when powering off of battery, therefor we must add the following to cpu_freq settings for throttling power consumption during inference

First open the settings to edit
```
sudo nano /etc/init.d/cpufreq_settings.sh
```

Append the file with the following content
```
#!/bin/sh
### BEGIN INIT INFO
# Provides:          cpufreq_settings
# Required-Start:    $all
# Required-Stop:
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: Set CPU frequency limits
### END INIT INFO

echo "ondemand" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 1800000 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
echo 1600000 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq
```

And apply a conservative undervolt in firmware config:
```
sudo nano /boot/firmware/config.txt
```
```
over_voltage_delta=-25000
```

Make the script executable
```
sudo chmod +x /etc/init.d/cpufreq_settings.sh
```

Register the script to run at boot
```
sudo update-rc.d cpufreq_settings.sh defaults
```

For Camera Info, [Check out this page](https://www.raspberrypi.com/documentation/accessories/camera.html#libcamera-and-libcamera-apps)


For more information and support related to battery monitoring [visit the wiki here](https://www.waveshare.com/wiki/UPS_HAT_(D))

Note that we updated the scripts for the latest python 3.12 expected syntax in [our version](https://github.com/robit-man/EGG/tree/main/pi5/UPS_HAT_D)

If your Bluetooth speaker connects but fails to act as a sound device, here are some steps to troubleshoot:

1. **Install Bluetooth Utilities:** Ensure you have the necessary Bluetooth and audio packages installed. Run:
   ```bash
   sudo apt update
   sudo apt install pulseaudio pulseaudio-module-bluetooth pavucontrol bluez
   ```

2. **Enable Bluetooth Service:** Make sure the Bluetooth service is running:
   ```bash
   sudo systemctl enable bluetooth
   sudo systemctl start bluetooth
   ```

3. **Load Bluetooth Module in PulseAudio:**
   Add or check for this line in the PulseAudio configuration file (`/etc/pulse/default.pa`):
   ```bash
   load-module module-bluetooth-discover
   ```
   Restart PulseAudio:
   ```bash
   pulseaudio -k
   pulseaudio --start
   ```

4. **Connect via Bluetooth Control Tool:**
   Use `bluetoothctl` to connect manually and trust the device:
   ```bash
   bluetoothctl
   power on
   agent on
   default-agent
   scan on  # Find your device
   pair <device_mac>
   trust <device_mac>
   connect <device_mac>
   ```

5. **Select as Output Device:**
   Open the PulseAudio Volume Control tool:
   ```bash
   pavucontrol
   ```
   In the *Playback* or *Output Devices* tab, select your Bluetooth speaker as the audio output.

6. **Reboot and Retry:** Sometimes a full reboot helps after setting these configurations.

## Docker Setup

For the **Raspberry Pi 5 (ARM64 architecture)**, follow these steps to properly install Docker. The Pi 5 is ARM-based, and the Docker packages for ARM64 must be used.

---

### 1. **Update System and Install Prerequisites**
Update your system and install necessary tools:
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg lsb-release
```

---

### 2. **Add Docker’s Official GPG Key**
Create the keyrings directory and add Docker's GPG key:
```bash
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

---

### 3. **Set Up Docker Repository**
For Raspberry Pi OS, which is based on Debian, use the following repository setup for ARM64 architecture:
```bash
echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

> Replace `$(lsb_release -cs)` with your specific Debian release codename if necessary (e.g., `bullseye` for Raspberry Pi OS 11).

---

### 4. **Update and Install Docker**
Update the package list and install Docker:
```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

---

### 5. **Enable and Start Docker**
Ensure Docker starts automatically:
```bash
sudo systemctl enable docker
sudo systemctl start docker
```

---

### 6. **Verify Installation**
Check Docker's version and run a test container:
```bash
docker --version
sudo docker run hello-world
```

---

### Additional Notes for Raspberry Pi 5:
1. **Running Docker Without `sudo`:**
   Add your user to the `docker` group to run Docker commands without `sudo`:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. **ARM-Specific Images:**
   Ensure you use Docker images built for ARM64 architecture. Many images on Docker Hub already support ARM64.

3. **Troubleshooting Release Mismatch:**
   If you encounter issues with the release codename in `$(lsb_release -cs)`, use `bullseye` or `stable` directly:
   ```bash
   echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian stable stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

### Continue inside piper to set up the [piper specific docker container](https://github.com/robit-man/EGG/tree/main/pi5/piper)
