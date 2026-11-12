class RailwayIntelligence:

    def __init__(self):
        self.zone_states = {}
        self.message_callback = None

    def set_message_callback(self, callback):
        self.message_callback = callback

    def report_message(self, message):
        if self.message_callback is not None:
            self.message_callback(message)

    def synchronize_zones(self, zone_names):
        self.zone_states = {}

        for zone_name in zone_names:
            self.zone_states[zone_name] = False

    def process_occupancy_event(self, event):
        self.zone_states[event.zone_name] = event.occupied
