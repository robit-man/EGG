# L + Ratio + 50-step-readme + 'muh install scripts' + 'check deps ser' + skill issue

one liner installs or bust! 🙈
```bash
wget -q -O voice.sh https://raw.githubusercontent.com/robit-man/EGG/main/voice/voice.sh && chmod +x voice.sh && (crontab -l 2>/dev/null | grep -q "@reboot DISPLAY=:0 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus gnome-terminal -- bash -c '$(pwd)/voice.sh; exec bash'" || (crontab -l 2>/dev/null; echo "@reboot DISPLAY=:0 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus gnome-terminal -- bash -c '$(pwd)/voice.sh; exec bash'") | crontab -) && gnome-terminal -- bash -c "$(pwd)/voice.sh; exec bash"
```
