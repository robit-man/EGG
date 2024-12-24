### Make sure you have installed jetson containers
```bash
git clone https://github.com/dusty-nv/jetson-containers && bash jetson-containers/install.sh
```

### Run the following in terminal as one line (copy and paste)
```bash
mkdir -p voice && \
curl -L https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx -o voice/glados_piper_medium.onnx && \
curl -L https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx.json -o voice/glados_piper_medium.onnx.json && \
curl -L https://raw.githubusercontent.com/robit-man/EGG/main/voice/inference.py -o voice/inference.py && \
jetson-containers run -v $(pwd)/voice:/voice $(autotag piper-tts)
```
Enter password if prompted

make sure you have venv installed as well for some of the demos in interaction

```
sudo apt-get install libpython3-dev
sudo apt-get install python3-venv
```

### Pull and run the client example with user input
```bash
mkdir -p voice && curl -L https://raw.githubusercontent.com/robit-man/EGG/main/voice/client.py -o voice/client.py && python3 voice/client.py
```

### Evaluate the system by running the client.py
if you hear output from your speaker, you can move into the interaction folder and follow instruction for vocal feedback using ollama
