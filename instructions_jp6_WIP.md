install steps egg

## 1. install [jetpack 6](https://developer.nvidia.com/embedded/jetpack) on Jetson AGX Orin

Use Jetson SDK from a host linux machine with more than 100 gb of storage available

save and build firmware ALL ON THE SAME DRIVE or you will have install issues

Use the provided USBA-USBC cable provided with the ORIN specifically for firmware connection

Press the two rightmost buttons and hold them down prior to applying power, then once power is applied, release the rightmost button still holding the center button, this sets DFU mode


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

## 5. Download files from [GLaDOS_TTS](https://huggingface.co/DavesArmoury/GLaDOS_TTS/tree/main) 

RIVA GLADOS INSTALL
```
sudo docker run --runtime=nvidia -it --rm \
    -v /home/roko/RIVA/artifacts:/servicemaker-dev \
    -v /home/roko/RIVA/riva_repo:/data \
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
