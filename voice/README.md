### Install jetson containers
```bash
git clone https://github.com/dusty-nv/jetson-containers && bash jetson-containers/install.sh
```

### Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
pull the model used in the model_to_tts.py
```bash
ollama pull llama3.2-vision
```

### Run the following in terminal as one line (copy and paste)
```bash
mkdir -p voice && \
[ -f voice/glados_piper_medium.onnx ] || curl -L https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx -o voice/glados_piper_medium.onnx && \
[ -f voice/glados_piper_medium.onnx.json ] || curl -L https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx.json -o voice/glados_piper_medium.onnx.json && \
[ -f voice/inference.py ] || curl -L https://raw.githubusercontent.com/robit-man/EGG/main/voice/inference.py -o voice/inference.py && \
jetson-containers run -v $(pwd)/voice:/voice $(autotag piper-tts) bash -c "cd /voice && python3 inference.py"
```
Enter password if prompted

make sure you have venv installed as well for some of the demos in interaction

```bash
sudo apt-get install libpython3-dev
sudo apt-get install python3-venv
```

### Run the following in terminal to activate TEXT input and model handler inside the [interaction](https://github.com/robit-man/EGG/tree/main/voice/interaction) folder
```bash
mkdir -p /home/$(whoami)/voice && \
if [ -f /home/$(whoami)/voice/input.py ] && [ -f /home/$(whoami)/voice/model_to_tts.py ]; then \
    gnome-terminal -- bash -c 'cd /home/$(whoami)/voice && python3 input.py; exec bash' && \
    gnome-terminal -- bash -c 'cd /home/$(whoami)/voice && python3 model_to_tts.py; exec bash'; \
else \
    git clone --depth=1 --filter=blob:none --sparse https://github.com/robit-man/EGG.git /tmp/EGG && \
    cd /tmp/EGG && git sparse-checkout set voice/interaction && \
    cp -r voice/interaction/* /home/$(whoami)/voice/ && \
    cd /home/$(whoami)/voice && \
    gnome-terminal -- bash -c 'python3 input.py; exec bash' && \
    gnome-terminal -- bash -c 'python3 model_to_tts.py --stream --history; exec bash' && \
    rm -rf /tmp/EGG; \
fi
```

### To run using VOSK for VOCAL input and model handler
```bash
mkdir -p /home/$(whoami)/voice && \
if [ -f /home/$(whoami)/voice/vosk.py ] && [ -f /home/$(whoami)/voice/model_to_tts.py ]; then \
    gnome-terminal -- bash -c 'cd /home/$(whoami)/voice && python3 vosk.py; exec bash' && \
    gnome-terminal -- bash -c 'cd /home/$(whoami)/voice && python3 model_to_tts.py; exec bash'; \
else \
    git clone --depth=1 --filter=blob:none --sparse https://github.com/robit-man/EGG.git /tmp/EGG && \
    cd /tmp/EGG && git sparse-checkout set voice/interaction && \
    cp -r voice/interaction/* /home/$(whoami)/voice/ && \
    cd /home/$(whoami)/voice && \
    gnome-terminal -- bash -c 'python3 vosk.py; exec bash' && \
    gnome-terminal -- bash -c 'python3 model_to_tts.py --stream --history; exec bash' && \
    rm -rf /tmp/EGG; \
fi
```

### Pull and run the client example with user input
```bash
mkdir -p voice && curl -L https://raw.githubusercontent.com/robit-man/EGG/main/voice/client.py -o voice/client.py && python3 voice/client.py
```

### Evaluate the system by running the client.py
if you hear output from your speaker, you can move into the interaction folder and follow instruction for vocal feedback using ollama
