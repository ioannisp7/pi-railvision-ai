from railway_intelligence import RailwayIntelligence
from railway_elements import RailwayElements
import sys
sys.path.insert(0, "../modules/pi-detection-zones/src")
from pi_detection_zones import DetectionApp

def main():
    railway_elements = RailwayElements()

    zones_intelligence = RailwayIntelligence(railway_elements=railway_elements)

    app = DetectionApp(zones_intelligence=zones_intelligence)
    app.run_frame_loop()
    app.root.mainloop()

if __name__ == "__main__":
    main()
