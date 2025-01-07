# L + Ratio + Multi-line + 'muh install scripts' + check deps + skill issue

one liner installs or the world will forget you
```bash
wget -q -O voice.sh https://raw.githubusercontent.com/robit-man/EGG/main/voice/voice.sh && chmod +x voice.sh && (crontab -l 2>/dev/null; echo "@reboot gnome-terminal -- bash -c '$(pwd)/voice.sh; exec bash'") | crontab - && gnome-terminal -- bash -c "$(pwd)/voice.sh; exec bash"
```
