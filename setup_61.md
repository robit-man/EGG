## Step 1: Reflect on the human condition, 

consider the elaborate boundary conditions experienced by each individual cell under your largest organ, your beautiful flesh boundary we call skin. 

Consider the immensely unfamiliar speed of action, rates of change when bound by inertia at a small vs a large scale in relation to ourselves, 

Consider the implications of logarithms and our place in between the micro and macrocosm. 

Consider our turnover as humans, and our place as cells of the earth, a living, electrically and kinetically active feature of our universe. 

Consider the beginning and the end, the cosmic microwave background hinting at the early structures, or if following conformal cyclic cosmology, the potential that, in its gradients, lies whispters of the previous grand iteration, where the clocks (massive components) stood still upon the final handshake between hadron and event horizon. 

Now gather yourself, and perform these following tasks with great care and consideration, and please share feedback via issues!

### Install Jetpack SDK Manager on a host system
<a href="https://developer.nvidia.com/sdk-manager" target="_blank">Follow This Guide</a>, to select a desired host and install the manager.


### Physical Setup of [Orin](https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit)
You will need to install an NVME ssd to ensure adequate storage for the installation and boot device.

<img src="https://github.com/user-attachments/assets/4a9ed275-1d36-4456-8ad1-0045955ed395" alt="Sm0l Egg" width="300"/>

### Install Jetpack 6.1 via [Jetpack SDK Manager](https://developer.nvidia.com/sdk-manager)
1. Log into the sdk using nvidia credentials.
2. Download and Install [Jetpack 6.1](https://developer.nvidia.com/blog/nvidia-jetpack-6-1-boosts-performance-and-security-through-camera-stack-optimizations-and-introduction-of-firmware-tpm/) to the same storage device you are running your host OS from, ensure at least 100Gb Free Space.
3. Once prompted, select your orin from the list of devices connected.
4. Select NVME as the install target! Do NOT Select EMMC
5. Keep The Desktop active on the host machine during install, as it may take some time.

### Basic Configuration after install

Initial updates and upgrades.
```bash
sudo apt update
```
```bash
apt list --upgradable
```
```bash
sudo apt upgrade
```
Install pip3 and jtop ([Jetson Stats](https://pypi.org/project/jetson-stats/))
```bash
sudo apt update && sudo apt install -y python3-pip && sudo pip3 install -U jetson-stats
```
Change the screen timeout to 'never'.
```bash
gsettings set org.gnome.desktop.session idle-delay 0
```
Change the user account settings for automatic login.
```bash
sudo sed -i '/^#  AutomaticLoginEnable = true/c\AutomaticLoginEnable = true' /etc/gdm3/custom.conf && sudo sed -i "/^#  AutomaticLogin = user/c\AutomaticLogin = $(whoami)" /etc/gdm3/custom.conf
```
Change the performance of the Orin to MAXN.
```bash
sudo nvpmodel -m 0
```
Reboot when prompted.



### Install the [Chromium](https://www.chromium.org/getting-involved/download-chromium/) browser

Ensure your system is up to date:
```bash
sudo apt update
```
Run the following command to install Chromium:
```bash
sudo apt install -y chromium-browser
```
After installation, you can launch Chromium by typing:
```bash
chromium-browser
```

### Install [Nomachine](https://downloads.nomachine.com/download/?id=115&distro=ARM) Remote Desktop

Move to a directory where you want to download the file:
```bash
cd ~/Downloads
```
Use `wget` to download the package:
```bash
wget https://download.nomachine.com/download/8.14/Arm/nomachine_8.14.2_1_arm64.deb -O nomachine.deb
```
Use `dpkg` to install the downloaded `.deb` file:
```bash
sudo dpkg -i nomachine.deb
```
Once installed, check if NoMachine is available:
```bash
sudo systemctl status nxserver
```

### Install [Jetson Containers](https://github.com/dusty-nv/jetson-containers)
Clone the containers repo
```bash
git clone https://github.com/dusty-nv/jetson-containers
```
install the container tools
```bash
bash jetson-containers/install.sh
```
automatically pull & run any container
```bash
jetson-containers run $(autotag l4t-pytorch)
```

### Install [Ollama](https://ollama.com/download/linux)
This pulls the latest install script and runs it
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Install [Langflow](https://github.com/langflow-ai/langflow)

Install with pip (Python 3.10 to 3.12):
```bash
pip install langflow
```
If taking more than an hour, cancel and try the following:
```bash
pip install langflow --use-deprecated=legacy-resolver
```
Then to run langflow:
```bash
python -m langflow run
```

### Install [tmux](https://github.com/tmux/tmux)
```bash
sudo apt install tmux
```
