## Piper docker setup and automation

Building
```bash
sudo docker build -t piper-tts-rpi5
```

Running first time
```bash
sudo docker run --name piper-tts -it piper-tts-rpi5
```

Running with exposed shared folder
```bash
sudo docker run -v /home/$(whoami)/voice:/opt/voice -it piper-tts-rpi5
```

starting piper inside the container
```bash
piper -h
```

testing
```bash
echo "Testing Speech Synthesis on a Raspberry Pi 5!" | piper --model /opt/voice/glados_piper_medium.onnx
t
 --output_file /opt/voice/tempfile.wav
```

with piper server script and network exposure
```bash
sudo docker run --network host -v /home/$(whoami)/voice:/opt/voice -w /opt/voice -it piper-tts-rpi5 python3 voice_server.py
```
