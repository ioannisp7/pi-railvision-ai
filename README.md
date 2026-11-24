# pi-railvision-ai
DIY model railway automation system using Raspberry Pi, computer vision, and AI-based train detection.

Feature requests and ideas are welcome via Issues/Discussions. Implementation is done solely by the project owner.

This project explores using camera input and machine learning techniques to detect trains, monitor track occupancy, and support automation features for a model railway setup.

## Features (planned)
- Detect trains using camera-based computer vision
- Track train movement across the layout
- Track occupancy monitoring
- Experiment with AI-based recognition on Raspberry Pi
- Build a foundation for model railway automation

## Status
Early development / experimental stage.

## Hardware
- Raspberry Pi
- Camera module
- Model railway layout

## Software
- Python
- OpenCV / computer vision libraries
- Machine learning models (TBD)

## Setup
Requires Raspberry Pi OS with desktop and a Pi camera module.

```bash
sudo apt install python3-picamera2 python3-opencv python3-pil python3-pil.imagetk python3-tk -y
git clone --recurse-submodules https://github.com/ioannisp7/pi-railvision-ai.git
```

If the repository was cloned without submodules:

```bash
git submodule update --init
```

## Running
Run from inside `src/`, because the detection module path is relative to the working directory:

```bash
cd src
python3 pi_railvision_ai.py
```

## Notes
This is a personal hobby project for learning and experimentation.
