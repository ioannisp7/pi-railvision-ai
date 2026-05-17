The architecture of the project have 5 layers of development:

Layer 1 — Cameras
Cameras continuously send frames to the Pi. They observe:
- track blocks
- stations
- intersections

Layer 2 — Image Processing
Software analyzes video frames to determine:
- “something moved”
- “a train occupies this section”
- “train moving left in the turnout”
- “this is Train A”

Layer 3 — Track State Model
- This is the “digital twin” of the layout.
- Block_A = occupied
- Block_B = free
- Train_1 = entering turnout T3

Layer 4 — Automation Logic
- switch turnout?
- stop train?
- reserve route?
- prevent collision?
- dispatch next train?

Layer 5 — Physical Control
Commands go to:
- DCC-EX
- turnout decoders
- relays
- GPIO
- signal LEDs
- servo motors