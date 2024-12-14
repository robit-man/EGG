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
jetson-containers run -v "$(pwd)/voice":/app/voice -v "$(pwd)/inference.py":/app/inference.py "$(autotag piper-tts)" python3 /app/inference.py
```
Enter password if prompted

### Pull and run the client example with user input
```bash
mkdir -p voice && curl -L https://raw.githubusercontent.com/robit-man/EGG/main/voice/client.py -o voice/client.py && python3 voice/client.py
```
