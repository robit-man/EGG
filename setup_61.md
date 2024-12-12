## Step 1: Reflect on the human condition, 

consider the elaborate boundary conditions experienced by each individual cell under your largest organ, your beautiful flesh boundary we call skin. 

Consider the immensely unfamiliar speed of action, rates of change when bound by inertia at a small vs a large scale in relation to ourselves, 

Consider the implications of logarithms and our place in between the micro and macrocosm. 

Consider our turnover as humans, and our place as cells of the earth, a living, electrically and kinetically active feature of our universe. 

Consider the beginning and the end, the cosmic microwave background hinting at the early structures, or if following conformal cyclic cosmology, the potential that, in its gradients, lies whispters of the previous grand iteration, where the clocks (massive components) stood still upon the final handshake between hadron and event horizon. 

Now gather yourself, and perform these following tasks with great care and consideration, and please share feedback via issues!

### Install Jetpack SDK Manager on a host system
<a href="https://developer.nvidia.com/sdk-manager" target="_blank">Follow This Guide</a>, to select a desired host and install the manager.


### Physical Setup of Orin
You will need to install an NVME ssd to ensure adequate storage for the installation and boot device.

<img src="https://github.com/user-attachments/assets/4a9ed275-1d36-4456-8ad1-0045955ed395" alt="Sm0l Egg" width="300"/>

### Install Jetpack 6.1 via Jetpack SDK Manager
1. Log into the sdk using nvidia credentials.
2. Download and Install jetpack 6.1 to the same storage device you are running your host OS from, ensure at least 100Gb Free Space.
3. Once prompted, select your orin from the list of devices connected.
4. Select NVME as the install target! Do NOT Select EMMC
5. Keep The Desktop active on the host machine during install, as it may take some time.

### Basic Configuration after install
Change the screen timeout to 'never'
```bash
gsettings set org.gnome.desktop.session idle-delay 0
```
Change the user account settings for automatic login
```bash
sudo sed -i '/^#  AutomaticLoginEnable = true/c\AutomaticLoginEnable = true' /etc/gdm3/custom.conf && sudo sed -i "/^#  AutomaticLogin = user/c\AutomaticLogin = $(whoami)" /etc/gdm3/custom.conf
```
Change the performance of the Orin to MAXN
```bash
sudo nvpmodel -m 0
```
Reboot When Prompted

### Install the Chromium browser

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

### Install Nomachine Remote Desktop

Move to a directory where you want to download the file:
```bash
cd ~/Downloads
```
Use `curl` to download the package:
```bash
curl -O https://download.nomachine.com/download/8.14/Arm/nomachine_8.14.2_1_arm64.deb
```
Use `dpkg` to install the downloaded `.deb` file:
```bash
sudo dpkg -i nomachine_8.14.2_1_arm64.deb
```
Once installed, check if NoMachine is available:
```bash
sudo systemctl status nxserver
```

### Install Langflow

Install with pip (Python 3.10 to 3.12):
```bash
pip install langflow
```

