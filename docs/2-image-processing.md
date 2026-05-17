1. Visualize detection zones
Run python script test2_detection_zones.py
A live camera preview with overlayed image:
- a rectangle drawn on screen
- named detection zone
- occupancy text
At this point, zones are set manually with coordinates. Later will have interactive zone calibration and automatic track alignment
Until now, verified: Camera -> Frame -> Manual virtual zone

2. Background subtraction
In this test the system attempts to determine if something is inside the zone by comparing learned background vs current frame.
Setup test:
- Set camera view at a rails part
- Camera movement during run will break the test
- Start with NO train inside zone
- Keep lighting stable
- Avoid moving hand/objects in frame

Run python script test3_background_subtraction.py
Two windows will open one window with camera image with overlayed green rectangle waiting B key to be pressed. when the key is pressed the image will be kept as background and all future images will be complared against it. A second window opens with foreground mask, black = background, white = detected changes

Test 1: Move a train inside zone and stop it there. The occupancy text should change to OCCUPIED and remains.
Test 2: Move a train outside zone. The occupancy text should change to FREE.
Test 3: Have a train enter and exit zone without stopping a nd verify it get detected.

Script pipeline:
STEP 1 — Take photo
STEP 2 — Convert to grayscale because detecting changes is easier
STEP 3 — Save empty layout wheb B key is presses
STEP 4 — Compare current image to empty image
STEP 5 — Convert small changes to black (tiny noise ignored)
STEP 6 — Look ONLY inside railway zone
STEP 7 — Count white pixels
STEP 8 — Decide occupancy

Until now, verified: camera-based virtual occupancy detection

3. Image noise cleanup
Setup the same test as 2. and run python script test4_remove_noise.py

4. Code refactoring
Setup the same test as 2. and 3. and run python script test5_refactored.py
In script added the capability to easy enable/disable image cleanup. Perform tests with both settings by changing
line 28 to True/False

5. Add multiple zones
Setup the same test as before and run python script test6_multiple_zones.py.
Zones are defined in a list of dictionaries and any number of zones can be defined as items in the list. Each zone as a dictionary have the required attributes.

6. Setup the same test as before and run python script test7_polygon_zones.py.
Zones now can be in any quadrilateral shape
