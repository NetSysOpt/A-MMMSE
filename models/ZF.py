import sys
import numpy as np


class ZF():
    def __init__(self, T, R, I, d, sigma, total_power, alpha):
        """
        R: Number of receiving antennas.
        d: Number of data streams.
        I: Number of users.
        """
        self.T = T
        self.R = R
        self.d = d
        self.I = I
        self.sigma = sigma
        self.P = total_power
        self.alpha = alpha

    def name(self):
        return "zf"

    def forward(self, H):
        V = np.zeros(shape=(self.T, self.d, self.I)) + 1j*np.zeros(shape=(self.T, self.d, self.I))
        for i in range(self.I):
            V[:, :, i] = np.conjugate(H[:, :, i]).T @ np.linalg.inv(H[:, :, i] @ np.conjugate(H[:, :, i]).T)
        return V