![image](https://github.com/user-attachments/assets/65e28b78-92d8-432a-95c7-49c927435b7a)

```bash
wget -q -O voice.sh https://raw.githubusercontent.com/robit-man/EGG/main/voice/voice.sh && chmod +x voice.sh && mkdir -p ~/.config/autostart && [ ! -f ~/.config/autostart/voice.sh.desktop ] && echo -e "[Desktop Entry]\nType=Application\nExec=gnome-terminal -- bash -c '$(pwd)/voice.sh; exec bash'\nHidden=false\nNoDisplay=false\nX-GNOME-Autostart-enabled=true\nName=VoiceScript\nComment=Run voice.sh at startup" > ~/.config/autostart/voice.sh.desktop && gnome-terminal -- bash -c "$(pwd)/voice.sh; exec bash"
```

For [debugging](https://github.com/robit-man/EGG/blob/main/voice/README_DEBUG.md) prior to opening an issue ;)
