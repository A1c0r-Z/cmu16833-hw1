'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        self.eps = 1e-10
    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        M = X_bar.shape[0]
        
        weights = X_bar[:, 3]
        w_sum = np.sum(weights)
        
        if w_sum == 0:
            w_sum = 1.0
            weights = np.full(M, 1.0 / M)
        
        X_bar_resampled = np.zeros_like(X_bar)
        
        step = w_sum / M
        
        r = np.random.uniform(0, step)
        
        c = weights[0] 
        i = 0
        
        for m in range(M):
            t = r + m * step
            
            while t > c:
                i += 1
                if i >= M:
                    i = M - 1
                    break
                
                c += weights[i]
            
            X_bar_resampled[m, :] = X_bar[i, :].copy()
        
        X_bar_resampled[:, 3] = 1.0 / M
        
        return X_bar_resampled