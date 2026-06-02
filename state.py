# state.py


class AppState:
    def __init__(self):
        self.clear()

    def clear(self):
        self.magic_data = None
        self.detector_a_event = None
        self.detector_a_event_hist = None
        self.detector_a_event_normalized = None
        self.detector_a_event_hist_normalized = None
        self.detector_b_event = None
        self.detector_b_event_hist = None
        self.detector_b_event_normalized = None
        self.detector_b_event_hist_normalized = None
        self.monitor_cave = None
        self.da_peaks = None


STATE = AppState()
STATE_VANADIUM = AppState()
