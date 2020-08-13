#Dummy model to standardize the player classes
class DummyModel():
    def __init__(self):
        # Shared conv layers for feature extraction
        # Note input type for Train.py
        self.obsIsImage = False