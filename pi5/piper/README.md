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
echo "Testing Speech Synthesis on a Raspberry Pi 5!" | piper --model /opt/voice/ --output_file /opt/voice/tempfile.wav
```
