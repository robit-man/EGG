### ASR > Ollama > TTS Pipeline for Pi5

## Installation and Runtime

Purchase the following list of hardware components:
[Raspberry Pi 5 8GB](https://www.sparkfun.com/products/23551) $80
[Pi Active Cooling](https://www.sparkfun.com/products/23585) $5
[Pi Battery UPS](https://www.amazon.com/gp/product/B0D39VDMDP) $33
[2x 21700 Batteries](https://www.amazon.com/dp/B0CJ4J6B8Z) $20
[Mini Bluetooth Speaker / Mic](https://www.amazon.com/dp/B0BPNYY61M) $13

# Total Cost: $151


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
Now we automate the activation across reboots
```
mkdir -p ~/.config/autostart && echo -e "[Desktop Entry]\nType=Application\nName=Feedback Script\nExec=lxterminal -e python3 /home/egg/feedback.py\nStartupNotify=false\nTerminal=false" > ~/.config/autostart/feedback.desktop
```

## Debugging

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
