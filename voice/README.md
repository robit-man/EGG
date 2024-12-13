Run the following jetson container after installing jetson containers
```bash
jetson-containers run $(autotag piper-tts)
```
If you are using jetpack 6.1, the container will likely need to be built from scratch, however it will prompt you to do so if needed, and is a relatively straightforward automated process.

Download the following model files and place them in the folder with inference.py

[Download Onnx](https://github.com/robit-man/EGG/raw/refs/heads/main/voice/glados_piper_medium.onnx)
<a href="https://raw.githubusercontent.com/robit-man/EGG/refs/heads/main/voice/glados_piper_medium.onnx.json" download>
    <img src="https://img.shields.io/badge/Download-glados_piper_medium.onnx.json-blue" alt="Download glados_piper_medium.onnx.json">
</a>

you must then exit by typing 'exit' and re-run the jetson-containers using the following, and be sure to replace the host shared folder with where you place the 'inference.py' and onnx files so they are exposed to the container.
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag piper-tts)
```
