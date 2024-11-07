### ASR > Ollama > TTS Pipeline for Pi5
Install Ollama
```
curl -fsSL https://ollama.com/install.sh | sh
```
Then Get the feedback.py file
```
curl -o ~/feedback.py https://raw.githubusercontent.com/robit-man/EGG/main/pi5/feedback.py
```
Now we automate the activation across reboots
```
mkdir -p ~/.config/autostart && echo -e "[Desktop Entry]\nType=Application\nName=Feedback Script\nExec=lxterminal -e python3 /home/egg/feedback.py\nStartupNotify=false\nTerminal=false" > ~/.config/autostart/feedback.desktop
```
