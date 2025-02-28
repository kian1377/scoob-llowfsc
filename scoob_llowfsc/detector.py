from .math_module import xp, xcipy, ensure_np_array

import numpy as np
import astropy.units as u
import copy

class DETECTOR():

    def __init__(
            self,
            exp_time=0.001, 
            throughput=0.1,
            gain=1, 
            read_noise=5, 
            blacklevel=5,
            dark_current=0.5, # e per second 
            qe=0.5, 
            Nbits=16,
        ):
        self.exp_time = exp_time
        self.qe = qe
        self.throughput = throughput
        self.read_noise = read_noise
        self.blacklevel = blacklevel
        self.dark_current = dark_current
        self.Nbits = Nbits
        self.sat_thresh = 2**self.Nbits - 1
        self.gain = gain

    def add_noise(self, flux_image):
        flux = self.throughput * flux_image
        ph_counts = flux * self.exp_time
        e_counts = self.qe * ph_counts
        dark_counts = self.dark_current * self.exp_time * xp.ones_like(flux_image)
        read_noise = self.read_noise * xp.random.randn(flux_image.shape[0], flux_image.shape[1])
        # print(read_noise.shape)

        det_counts = self.gain * xp.random.poisson(e_counts + dark_counts)

        det_counts = det_counts + self.blacklevel * xp.ones_like(flux_image) + read_noise
        det_counts[det_counts>self.sat_thresh] = self.sat_thresh
        det_counts = xp.round(det_counts)
        return det_counts




