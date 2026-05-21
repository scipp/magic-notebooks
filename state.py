# state.py

class AppState:
    def __init__(self):
        self.clear()

    def clear(self):
        self.data_event = None
        self.data_event_hist = None
        self.data_cave_monitor = None
        self.data_event_vanadium = None
        self.data_event_hist_vanadium = None
        self.data_cave_monitor_vanadium = None
        self.data_event_normalized_per_monitor = None
        self.data_event_normalized_per_vanadium = None
        self.data_peaks = None
        
        

STATE = AppState()
