1. Hardware setup
A Raspberry Pi 4 will be used with one camera module. Later, a secord camera will be added to for better visibility of the track. If Raspberry Pi 5 will replace 4 if more cpu power is needed.
Setup a Raspberry Pi 4 with camera module connected. Install latest Raspberry OS  (with desktop). For convinience, plug an ethernet cable, monitor, keyboard, mouse and turn on Raspberry Pi Connect.

2. Software setup
Once Raspberry is started, update packages 
```bash
sudo apt-get update && sudo apt-get full-upgrade -y
```

Install required packages
```bash
sudo apt install python3-opencv python3-picamera2 -y
```

3. Test camera connection
Run command to verify that camera is properly connected and it can be used by Pi
```bash
rpicam-hello
```
A short live camera preview will open up

4. Test camera by python
Run python script test1_camera.py
A live camera preview will open up. Press Q to quit.
Use resolution 1280×720 or 640×480. Higher resolution increases CPU massively.
Until now, verified: Pi -> Camera -> Python -> OpenCV -> Live Frames

