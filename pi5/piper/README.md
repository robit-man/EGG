## Piper docker setup and automation

Building
```bash
sudo docker build -t piper-tts-rpi5
```

Running
```bash
sudo docker run --name piper-tts -it piper-tts-rpi5
```

starting piper inside the container
```bash
piper -h
```

testing
```bash
echo "To address the elephant in the room: using text-to-speech technology isn’t just practical, it’s a lot of fun too!" | piper --model /opt/piper/voices/aru/medium/en_GB-aru-medium.onnx --output_file /opt/welcome.wav
```
