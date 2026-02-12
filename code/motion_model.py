'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.001
        self._alpha2 = 0.001
        self._alpha3 = 0.001
        self._alpha4 = 0.001

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        delta_trans = np.sqrt(np.sum((u_t1[:2] - u_t0[:2]) ** 2))
        delta_rot1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        delta_rot2 = u_t1[2] - u_t0[2] - delta_rot1
        
        delta_rot1 = self.normalize_angle(delta_rot1)
        delta_rot2 = self.normalize_angle(delta_rot2)

        rng = np.random.default_rng()
        # add noise
        delta_trans = rng.normal(delta_trans, np.sqrt(self._alpha3 * delta_trans ** 2 + self._alpha4 * (delta_rot1 ** 2 + delta_rot2 ** 2)))
        delta_rot1 = rng.normal(delta_rot1, np.sqrt(self._alpha1 * delta_rot1 ** 2 + self._alpha2 * delta_trans ** 2))
        delta_rot2 = rng.normal(delta_rot2, np.sqrt(self._alpha1 * delta_rot2 ** 2 + self._alpha2 * delta_trans ** 2))
        
        x_t = x_t0[0] + delta_trans * np.cos(x_t0[2] + delta_rot1)
        y_t = x_t0[1] + delta_trans * np.sin(x_t0[2] + delta_rot1)
        theta_t = x_t0[2] + delta_rot1 + delta_rot2
        theta_t = self.normalize_angle(theta_t)
        
        return np.array([x_t, y_t, theta_t])