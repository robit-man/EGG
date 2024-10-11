# Bill Of Materials
![Frame 4](https://github.com/user-attachments/assets/78cee061-4959-4f37-923d-290cc060ac84)

## Parts List
- [JETSON AGX ORIN](https://www.amazon.com/NVIDIA-Jetson-Orin-64GB-Developer/dp/B0BYGB3WV4/) $1,999
- [24V LI-ION BATTERY](https://www.amazon.com/SSCYHT-Rechargeable-Replacement-Wheelchair-24v12-5ah/dp/B0DDGRSYZ6) $147 (Must be 140mm x 120mm x 68mm)
- [Audio Amplifier](https://www.amazon.com/Bluetooth-Amplifier-DAMGOO-Technique-Password/dp/B07XG33WPN) $21.98
- [NileCAM81 4k Camera](https://www.e-consystems.com/camera-modules/ar0821-4k-hdr-gmsl2-camera-module.asp) (Contact to buy 6 at once) $389 / Camera
- [NileCAM81_CUOAGX Carrier Board](https://www.e-consystems.com/nvidia-cameras/jetson-agx-orin-cameras/ar0821-4k-hdr-gmsl2-camera.asp) $300 / Connector Board
- [6 Inch Farka Z Female to Female](https://www.amazon.com/gp/product/B094XXCY3M)
- [120W DC 24 to DC 12 Volt Buck](https://www.amazon.com/dp/B097SWGRRJ)
- [DisplayPort 90 Degree Converter](https://www.amazon.com/dp/B0CL4R649J)
- [Dummy Display Port](https://www.amazon.com/Furjosta-DisplayPort-Headless-Emulator-1920x1080/dp/B0BTP19LPG)
- [3.5mm Audio Cable](https://www.amazon.com/dp/B08MDPW6R2)
- [ReSpeaker Voice Recog Smart Microphone / Pre Amplifier](https://www.amazon.com/seeed-studio-ReSpeaker-USB-Array/dp/B07ZGZSBS4)
- [90 Degree usb-c adapter](https://www.amazon.com/dp/B0BBW8JKJX)
- [90 Degree usb-a adapter](https://www.amazon.com/Adapter-Converter-Extender-Compatible-Charger/dp/B0BP6T8SDG)
- [90 Degree 5.5x2.1mm Female to 5.5x2.5mm Male DC Barrel Jack](https://www.amazon.com/dp/B07YWPGGCG)
- [90 Degree DC Power Cable](https://www.amazon.com/dp/B0B6HSJRVH) (Get 2 packs)
- [USB-A Splitter Male to 2 Female](https://www.amazon.com/gp/product/B098L7WJ4C/)
- [BU353N GPS Reciever](https://www.amazon.com/GlobalSat-BU-353N-GNSS-Receiver-Black/dp/B0BLF7DSRY)
- [Toggle Power Button](https://www.amazon.com/DMWD-Waterproof-Anti-Vandal-Terminals-Aluminium/dp/B0BQXYX1PP)
- [7 Inch HDMI Monitor / Touchscreen](https://www.amazon.com/7inch-HDMI-LCD-Display-Capacitive/dp/B0894Q5VH3)
- [1kg PLA](https://www.amazon.com/stores/page/2E20608D-8170-49B9-B3F6-E6E303A31716) or other material of your choice
- Various [M4 bolts](https://www.amazon.com/FullerKreg-M4-0-7-Alloy-Socket-Finish/dp/B07B2R7LZF), Check thickness of stacked parts in your CAD program of choice
- [10mm m3 bolts](https://www.amazon.com/Socket-Screws-Bolts-Thread-100pcs/dp/B07CMSBQ11) for camera cooling fan mounting
- [Double Sided Mounting Tape](https://www.amazon.com/Gorilla-Heavy-Double-Sided-Mounting/dp/B082TQ3KB5)
## Construction Procedure

| Step Description | Image |
|------------------|-------|
| Assemble the following initial structure with 4x 20mm or 25mm m4 bolts, leave the center hole open for the battery corner protector later on | <img src="https://github.com/user-attachments/assets/f7a44754-64d0-4c89-b323-158b3dad556b" alt="step-1" style="max-width:600px;"> |
| Drop in the battery, and fit the battery hat on top and between the battery bracket | <img src="https://github.com/user-attachments/assets/7c6771ae-3190-4195-ac3c-1cdb65b2d383" alt="step-2" style="max-width:600px;"> |
| Fasten the camera mounts to the battery hat using 14mm m4 bolts, and the battery corner protectors to the bottom bracket via the single screw hole in the center, using 35mm m4 bolts on either side | <img src="https://github.com/user-attachments/assets/1f64c5d5-6aaa-404a-81bb-95e0ea1ef430" alt="step-3" style="max-width:600px;"> |
| Fasten the AGX ORIN seat to the top of the camera mounts | <img src="https://github.com/user-attachments/assets/fc89bd90-c9d1-4c74-ac3c-6071cf0f9263" alt="step-4" style="max-width:600px;"> |
| Place the AGX ORIN atop the seat, and then bolt the agx top straps to the battery straps using 20mm m4 bolts | <img src="https://github.com/user-attachments/assets/94aa6438-6361-471f-b4a8-3fa30ac54003" alt="step-5" style="max-width:600px;"> |
| Install the camera protector faces to the camera mounts using 20mm m4 bolts | <img src="https://github.com/user-attachments/assets/1f425bec-9d82-4500-8afb-d50ec117433f" alt="step-6" style="max-width:600px;"> |


## Power Electronics

| Step Description | Image |
|------------------|-------|
| The cameras should be mounted to the brackets prior to installing on the main frame | <img src="https://github.com/user-attachments/assets/ddc783cf-b517-4d6e-922e-5c6f892deb92" alt="photo_2024-09-06_10-32-12" style="max-width:600px;"> |
| The Farka connectors can be seen here, which should be first fastened to the carrier board mounted to the ORIN, and then fed to each camera carefully, note the orientation of the cameras | <img src="https://github.com/user-attachments/assets/057f44cc-a358-4a38-a09b-cb2ac8c96663" alt="photo_2024-09-06_10-33-18" style="max-width:600px;"> |
| The buck converter is mounted via double sided tape to the opposing side of the battery’s output cables, such that they wrap around and route through a toggle button and then onto the buck | <img src="https://github.com/user-attachments/assets/fc3f7f3f-ce47-4417-b3e9-8ab4e7ab61d1" alt="egg-pink" style="max-width:600px;"> |
| The audio amplifier is also fastened to the black PCIE Shroud on the AGX ORIN using the same double sided tape | <img src="https://github.com/user-attachments/assets/f5f6ce48-3c79-40b7-8e9d-bad0c5e0fe2f" alt="power_and_amp" style="max-width:600px;"> |
| The fan mounts can be seen as reversed for two of the 6 camera mounts | <img src="https://github.com/user-attachments/assets/2983d273-11b9-4e04-8711-87b94cdc8786" alt="fan-mount-closeup" style="max-width:600px;"> |
