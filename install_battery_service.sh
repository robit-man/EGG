#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run: sudo ./install_battery_service.sh" >&2
  exit 2
fi
target_user=${SUDO_USER:-}
if [[ -z ${target_user} || ${target_user} == root ]]; then
  echo "Run this from the desktop user account: sudo ./install_battery_service.sh" >&2
  exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
target_home=$(getent passwd "${target_user}" | cut -d: -f6)
target_uid=$(id -u "${target_user}")

install -d -m 0755 /usr/local/lib/egg-battery
install -m 0755 "${repo_dir}/battery.py" /usr/local/lib/egg-battery/battery.py
install -m 0644 "${repo_dir}/battery/systemd/egg-battery@.service" /etc/systemd/system/egg-battery@.service
install -d -o "${target_user}" -g "${target_user}" -m 0755 "${target_home}/.config/systemd/user"
install -o "${target_user}" -g "${target_user}" -m 0644 \
  "${repo_dir}/battery/systemd/egg-battery-indicator.service" \
  "${target_home}/.config/systemd/user/egg-battery-indicator.service"

usermod -aG dialout "${target_user}"
systemctl daemon-reload
systemctl enable --now "egg-battery@${target_user}.service"

if [[ -S /run/user/${target_uid}/bus ]]; then
  sudo -u "${target_user}" XDG_RUNTIME_DIR="/run/user/${target_uid}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${target_uid}/bus" \
    systemctl --user daemon-reload
  sudo -u "${target_user}" XDG_RUNTIME_DIR="/run/user/${target_uid}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${target_uid}/bus" \
    systemctl --user enable --now egg-battery-indicator.service
else
  echo "Desktop session bus is not active; the indicator will start at the next login."
fi

echo "Battery service installed. Read machine state with:"
echo "  cat /run/egg-battery/status.json"
