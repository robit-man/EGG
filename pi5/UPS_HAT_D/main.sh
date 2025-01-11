#!/bin/bash
echo "user name = ${USER}"
curPath=$(readlink -f "$(dirname "$0")")
echo $curPath
sudo rm -rf /home/${USER}/.config/autostart/battery.desktop
if [ ! -d "/home/${USER}/.config/autostart" ];then
    sudo mkdir /home/${USER}/.config/autostart
fi
sed -i "s#.*Exec.*#Exec=${curPath}/battery.sh#" `grep Exec -rl battery.desktop`
sed -i "s#.*Icon.*#Icon=${curPath}/images/battery.1.png#" `grep Icon -rl battery.desktop`
sudo cp battery.desktop /home/${USER}/.config/autostart/
sudo rm -rf  battery.sh
sudo touch  battery.sh
sudo chmod 777 battery.sh
echo "sleep 5" >> battery.sh
echo "cd ${curPath}" >> battery.sh
echo "DISPLAY=':0.0' python3 batteryTray.py " >> battery.sh
