### ASR > Ollama > TTS Pipeline for Pi5
Install [Ollama](https://ollama.com/)
```
curl -fsSL https://ollama.com/install.sh | sh
```
Pull llama3.2:1b
```
ollama pull llama3.2:1b
```
Install [Flite](http://www.festvox.org/flite/) TTS
```
sudo apt install flite
```
Then Get the feedback.py file
```
curl -o ~/feedback.py https://raw.githubusercontent.com/robit-man/EGG/main/pi5/feedback.py
```
Now we automate the activation across reboots
```
mkdir -p ~/.config/autostart && echo -e "[Desktop Entry]\nType=Application\nName=Feedback Script\nExec=lxterminal -e python3 /home/egg/feedback.py\nStartupNotify=false\nTerminal=false" > ~/.config/autostart/feedback.desktop
```
