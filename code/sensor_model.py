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
        # Tuned parameters matching the reference logic
        self._z_hit = 1.0       
        self._z_short = 0.12
        self._z_max = 0.05
        self._z_rand = 600

        self._sigma_hit = 100.0
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 8183.0

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 3

        self._occupancy_map = occupancy_map
        self._resolution = 10.0
        
        self._offset = 25

        self._ray_cast_step_size = 5.0

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        x_robot, y_robot, theta_robot = x_t1
        
        indices = np.arange(0, len(z_t1_arr), self._subsampling)
        z_k_observed = np.array(z_t1_arr)[indices]
        
        laser_x = x_robot + self._offset * np.cos(theta_robot)
        laser_y = y_robot + self._offset * np.sin(theta_robot)

        beam_angles_deg = indices - 90.0
        beam_angles_rad = np.radians(beam_angles_deg)
        beam_angles_world = theta_robot + beam_angles_rad
        
        z_k_star = np.zeros_like(z_k_observed)
        
        for i, angle in enumerate(beam_angles_world):
            z_k_star[i] = self.ray_cast(laser_x, laser_y, angle)
        
        p_hit = self.p_hit(z_k_observed, z_k_star)
        p_short = self.p_short(z_k_observed, z_k_star)
        p_max = self.p_max(z_k_observed)
        p_rand = self.p_rand(z_k_observed)
        
        p_z_k = (self._z_hit * p_hit + 
                 self._z_short * p_short + 
                 self._z_max * p_max + 
                 self._z_rand * p_rand)
        
        p_z_k = p_z_k[p_z_k > 0]
        
        if len(p_z_k) == 0:
            return 1e-10

        log_prob_zt1 = np.sum(np.log(p_z_k))
        
        return np.exp(log_prob_zt1)

    def ray_cast(self, x, y, theta):
        """
        Ray cast from (x,y) in direction theta to find distance to nearest obstacle.
        """
        start_x, start_y = x, y
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        dist_range = np.arange(0, self._max_range, self._ray_cast_step_size)
        
        for dist in dist_range:
            curr_x = start_x + dist * cos_theta
            curr_y = start_y + dist * sin_theta
            
            grid_x = int(curr_x / self._resolution)
            grid_y = int(curr_y / self._resolution)
            
            if (grid_x < 0 or grid_x >= self._occupancy_map.shape[1] or 
                grid_y < 0 or grid_y >= self._occupancy_map.shape[0]):
                return self._max_range
            
            prob_val = self._occupancy_map[grid_y, grid_x]
            if prob_val >= self._min_probability or prob_val == -1:
                return np.sqrt((curr_x - start_x)**2 + (curr_y - start_y)**2)
        
        return self._max_range

    def p_hit(self, z_k, z_k_star):
        mask = (z_k >= 0) & (z_k <= self._max_range)
        prob = np.zeros_like(z_k)
        
        if not np.any(mask):
            return prob

        eta = norm.cdf(self._max_range, loc=z_k_star, scale=self._sigma_hit) - \
              norm.cdf(0, loc=z_k_star, scale=self._sigma_hit)
        
        eta[eta == 0] = 1e-10
        
        prob[mask] = norm.pdf(z_k[mask], loc=z_k_star[mask], scale=self._sigma_hit) / eta[mask]
        return prob

    def p_short(self, z_k, z_k_star):
        prob = np.zeros_like(z_k)
        
        mask = (z_k >= 0) & (z_k <= z_k_star)
        
        if not np.any(mask):
            return prob

        safe_z_star = z_k_star.copy()
        safe_z_star[safe_z_star == 0] = 1e-10
        
        eta = 1.0 / (1.0 - np.exp(-self._lambda_short * safe_z_star))
        
        prob[mask] = eta[mask] * self._lambda_short * np.exp(-self._lambda_short * z_k[mask])
        return prob

    def p_max(self, z_k):
        return (np.abs(z_k - self._max_range) < 1e-3).astype(float)

    def p_rand(self, z_k):
        mask = (z_k >= 0) & (z_k < self._max_range)
        prob = np.zeros_like(z_k)
        prob[mask] = 1.0 / self._max_range
        return prob