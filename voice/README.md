# One liners or you have a skill issue
```bash
wget -q -O voice.sh https://raw.githubusercontent.com/robit-man/EGG/main/voice/voice.sh && chmod +x voice.sh && sudo cp "$(pwd)/voice.sh" /etc/init.d/ && sudo update-rc.d voice.sh defaults && gnome-terminal -- bash -c "$(pwd)/voice.sh; exec bash"
```
