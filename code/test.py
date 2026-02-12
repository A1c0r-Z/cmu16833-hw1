'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 9000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        self._occupancy_map = occupancy_map

        self._ray_cast_step_size = 1

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        x, y, theta = x_t1
        num_beams = len(z_t1_arr)
        
        indices = np.arange(0, num_beams, self._subsampling)
        
        z_k_observed = z_t1_arr[indices]
        
        valid_mask = z_k_observed > 0
        if not np.any(valid_mask):
            return 1e-10
        
        z_k_observed = z_k_observed[valid_mask]
        valid_indices = indices[valid_mask]
        
        beam_angles_rad = np.deg2rad(valid_indices - 90)
        beam_angles_world = theta + beam_angles_rad
        
        num_valid = len(z_k_observed)
        z_k_star = np.zeros(num_valid)
        for i in range(num_valid):
            z_k_star[i] = self.ray_cast(x, y, beam_angles_world[i])
        
        p_hit = self.p_hit(z_k_observed, z_k_star)
        p_short = self.p_short(z_k_observed, z_k_star)
        p_max = self.p_max(z_k_observed)
        p_rand = self.p_rand(z_k_observed)
        
        p_z_k = (self._z_hit * p_hit + 
                 self._z_short * p_short + 
                 self._z_max * p_max + 
                 self._z_rand * p_rand)
        
        p_z_k = np.maximum(p_z_k, 1e-10)
        
        log_prob_zt1 = np.sum(np.log(p_z_k))
        
        return np.exp(log_prob_zt1)

    def ray_cast(self, x, y, theta):
        step_size = self._ray_cast_step_size
        
        dx = np.cos(theta) * step_size
        dy = np.sin(theta) * step_size
        
        curr_x, curr_y = x, y
        
        for distance in np.arange(0, self._max_range, step_size):
            curr_x += dx
            curr_y += dy
            
            grid_x = int(curr_x / 10)
            grid_y = int(curr_y / 10)
            
            if (grid_x < 0 or grid_x >= self._occupancy_map.shape[1] or 
                grid_y < 0 or grid_y >= self._occupancy_map.shape[0]):
                return self._max_range
            
            if self._occupancy_map[grid_y, grid_x] > self._min_probability:
                return distance
        
        return self._max_range

    def p_hit(self, z_k, z_k_star):
        prob = np.zeros_like(z_k, dtype=float)
        
        valid = z_k <= self._max_range
        
        if not np.any(valid):
            return prob
        
        prob[valid] = norm.pdf(z_k[valid], loc=z_k_star[valid], scale=self._sigma_hit)
        
        # normalization = (norm.cdf(self._max_range, loc=z_k_star[valid], scale=self._sigma_hit) - 
                        # norm.cdf(0, loc=z_k_star[valid], scale=self._sigma_hit))
        normalization = np.ones_like(prob)
        valid_norm = normalization > 0
        prob[valid][valid_norm] = prob[valid][valid_norm] / normalization[valid_norm]
        
        return prob

    def p_short(self, z_k, z_k_star):
        prob = np.zeros_like(z_k, dtype=float)
    
        valid = (z_k <= z_k_star) & (z_k <= self._max_range)
        
        if not np.any(valid):
            return prob

        # eta = 1.0 / (1 - np.exp(-self._lambda_short * z_k_star[valid]))
        eta = 1
        prob[valid] = eta * self._lambda_short * np.exp(-self._lambda_short * z_k[valid])
        
        return prob

    def p_max(self, z_k):
        prob = np.where(np.abs(z_k - self._max_range) < 0.5, 1.0, 0.0)
        return prob

    def p_rand(self, z_k):
        prob = np.where((z_k >= 0) & (z_k <= self._max_range), 
                       1.0 / self._max_range, 
                       0.0)
        return prob