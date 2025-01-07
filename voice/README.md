# L + Ratio + 50-step-readme + 'muh install scripts' + 'check deps ser' + skill issue

one liner installs or bust! 🙈
```bash
wget -q -O voice.sh https://raw.githubusercontent.com/robit-man/EGG/main/voice/voice.sh && chmod +x voice.sh && mkdir -p ~/.config/autostart && echo -e "[Desktop Entry]\nType=Application\nExec=gnome-terminal -- bash -c '$(pwd)/voice.sh; exec bash'\nHidden=false\nNoDisplay=false\nX-GNOME-Autostart-enabled=true\nName=VoiceScript\nComment=Run voice.sh at startup" > ~/.config/autostart/voice.sh.desktop && gnome-terminal -- bash -c "$(pwd)/voice.sh; exec bash"
```
