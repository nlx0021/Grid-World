import numpy as np
from sympy import Matrix

def compute_visit_prob(P,
                       prob_policy,
                       init_dist,
                       gamma):

    S_size = P.shape[1]
    A_size = prob_policy.shape[1]
    
    P_pi = np.zeros_like(P[0])          # Size: [S_size, S_size]
    for s in range(S_size):
        P_pi[s, :] = np.dot(prob_policy[s, :], P[:, s, :]).reshape((-1,))     
    
    D = np.linalg.inv(np.eye(S_size) - gamma * P_pi)
    
    d = np.real(np.dot(init_dist, D)) * (1 - gamma)
    
    d = d / d.sum()
    
    return d