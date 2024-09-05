import numpy as np

class display_encode():
    def __init__(self, display_encoded_a, display_encoded_gamma):
        self.display_encoded_a = display_encoded_a
        self.display_encoded_gamma = display_encoded_gamma
    def L2C(self, Luminance):
        C = (Luminance / self.display_encoded_a) ** (1 / self.display_encoded_gamma)
        return C
    def C2L(self, Color):
        L = self.display_encoded_a * Color ** self.display_encoded_gamma
        return L