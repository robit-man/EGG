## 1. install [jetpack 6](https://developer.nvidia.com/embedded/jetpack) on Jetson AGX Orin

Use Jetson SDK from a host linux machine with more than 100 gb of storage available

save and build firmware ALL ON THE SAME DRIVE or you will have install issues

Use the provided USBA-USBC cable provided with the ORIN specifically for firmware connection

Press the two rightmost buttons and hold them down prior to applying power, then once power is applied, release the rightmost button still holding the center button, this sets DFU mode


### 1A. [install nomachine](https://downloads.nomachine.com/download/?id=115&distro=ARM) to access the machine on your network and keep it headless
To install run:
```
sudo dpkg -i nomachine_8.14.2_1_arm64.deb
```
then make sure to disable your hosts nx option to enable audio streaming and microphone forwarding under Server Settings > Devices

## 2. install [jetson-containers](https://github.com/dusty-nv/jetson-containers)

install the container tools
```
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```
automatically pull & run any container
```
jetson-containers run $(autotag l4t-pytorch)
```

## 3. install nilecam81 drivers (JP6)

e-con systems provides a developer portal with driver downloads in your account.

extract the firmware to a given folder

run
```
sudo bash install_binaries.sh 81
```
test
```
# Camera 0
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    videoconvert ! \
    videoflip method=clockwise ! \
    jpegenc ! \
    multifilesink location=/tmp/latest_frame_0.jpg
```
run jetson-containers and load and run python script from [Shared folder found here](https://github.com/robit-man/EGG/blob/main/Shared/frame-inference-test.py)
```
jetson-containers run -v /home/$(whoami)/Shared:/Shared  \
    -e HUGGINGFACE_TOKEN=<YOUR-HF-TOKEN> \
    $(autotag llama-vision) \
      python3 /Shared/frame-inference-test.py
```



## 4. install [riva 2.16.0 ](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html)

Sometimes riva_start.sh fails due to docker daemon not recognizing nvidia as default, therefor we add the following line to the daemon: 

"default-runtime": "nvidia",

```
sudo nano /etc/docker/daemon.json
```
```
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
and
```
jetson-containers run -v /home/$(whoami)/Shared:/Shared  \
    -e HUGGINGFACE_TOKEN=<YOUR HF TOKEN> \
    $(autotag llama-vision) \
      python3 /Shared/frame-inference-test.py
```

## 6. Configure Default Audio Devices (sink and source)

Disable nomachine EnableAudio flag by first:
```
sudo gedit /usr/NX/etc/node.cfg
```
then search for and change EnableAudio to 0:
```
EnableAudio=0
```
Restart nx
```
sudo /etc/NX/nxserver --restart
```

list your sinks:
```
pactl get-default-sink
```
then, list your sources:
```
pactl get-default-source
```
Now add these lines to both prevent automatic switching of defaults, as well as set default audio devices after first:
```
sudo gedit /etc/pulse/default.pa
```
Then, since we are using a respeaker 2.0:
```
### Disable module-switch-on-connect
unload-module module-switch-on-connect

# Set default sink and source
set-default-sink alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo
set-default-source alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input
```
Save and restart pulseaudio
```
pulseaudio -k
pulseaudio --start
```

if devices from nomachine are stubborn create a bash script and save it in home as reset_pulseaudio.sh:
```
#!/bin/bash
while true; do
  pactl set-default-source alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.multichannel-input
  sleep 5
done
```
add this line to startup scripts

```
/home/$(whoami)/reset_pulseaudio.sh
```

## 6. Download files from [GLaDOS_TTS](https://huggingface.co/DavesArmoury/GLaDOS_TTS/tree/main) 

RIVA GLADOS INSTALL
```
sudo docker run --runtime=nvidia -it --rm \
    -v /home/$(whoami)/RIVA/artifacts:/servicemaker-dev \
    -v /home/$(whoami)/RIVA/riva_repo:/data \
    --entrypoint="/bin/bash" \
     nvcr.io/nvidia/riva/riva-speech:2.13.1-servicemaker-l4t-aarch64
```
```
riva-build speech_synthesis \
    /servicemaker-dev/glados.rmir:tlt_encode \
    /servicemaker-dev/glados_fastpitch.riva:tlt_encode \
    /servicemaker-dev/glados_hifigan.riva:tlt_encode \
    --voice_name=GLaDOS \
    --sample_rate 22050
```
```
riva-deploy /servicemaker-dev/glados.rmir:tlt_encode /data/models
```
```
sudo cp -r /riva_repo/models/*. /model_repository/models/
```
