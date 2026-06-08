1. Visualize detection zones
Run python script 02_detection_zones.py
A live camera preview with overlayed image:
- a rectangle drawn on screen
- named detection zone
- occupancy text
At this point, zones are set manually with coordinates. Later will have interactive zone calibration and automatic track alignment
Until now, verified: Camera -> Frame -> Manual virtual zone

2. Background subtraction
In this test the system attempts to determine if something is inside the zone by comparing learned background vs current frame.
Setup test:
- Set camera view
- Camera movement during run will break the test
- Start with NO objects inside zone
- Keep lighting stable
- Avoid moving hand/objects in frame

Run python script 03_background_subtraction.py
Two windows will open, one window with camera image with overlayed green rectangle waiting B key to be pressed. when the key is pressed the image will be kept as background and all future images will be compared against it. A second window opens with foreground mask, black = background, white = detected changes

Test 1: Move an object inside zone and stop it there. The occupancy text should change to OCCUPIED and remains.
Test 2: Move an object outside zone. The occupancy text should change to FREE.
Test 3: Have an object enter and exit zone without stopping and verify it gets detected.

Script pipeline:
STEP 1 — Take photo
STEP 2 — Convert to grayscale because detecting changes is easier
STEP 3 — Save empty layout wheb B key is presses
STEP 4 — Compare current image to empty image
STEP 5 — Convert small changes to black (tiny noise ignored)
STEP 6 — Look ONLY inside zone
STEP 7 — Count white pixels
STEP 8 — Decide occupancy

Until now, verified: camera-based virtual occupancy detection

3. Image noise cleanup
Setup test and run python script 04_remove_noise.py

4. Code refactoring
Setup test and run python script 05_refactored.py
In script, added the capability to easy enable/disable image cleanup. Perform tests with both settings by changing
line 28 to True/False

5. Add multiple zones
Setup test and run python script 06_multiple_zones.py
Zones are defined in a list of dictionaries and any number of zones can be defined as items in the list. Each zone is a dictionary and contains the required attributes.

6. Setup test and run python script 07_polygon_zones.py
Zones now can be in any quadrilateral shape
