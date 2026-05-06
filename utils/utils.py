import os
import imageio
import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

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


def vis_heat_map(return_dict,
                 save_dir,
                 data_list=["policy", "A", "grad"],
                 vis_iter_list=[0]):
    
    if "solution_policy" in return_dict.keys():
        num_subfig = len(data_list) + 1
    else:
        num_subfig = len(data_list)
    for vis_iter in vis_iter_list:
        fig = plt.figure(figsize=(16,7))
        for idx, data_type in enumerate(data_list):
            
            mat = return_dict[data_type + "_list"][vis_iter]
            # mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-15)
            ax = fig.add_subplot(int("1%1d%1d" % (num_subfig, idx+1)))
            _mappable = ax.imshow(mat, cmap="afmhot")
            ax.set_title(data_type)
            fig.colorbar(_mappable)
        
        if "solution_policy" in return_dict.keys():
            ax = fig.add_subplot(int("1%1d%1d" % (num_subfig, idx+2)))
            policy = return_dict["solution_policy"]
            policy_mat = np.zeros_like(mat, dtype=np.uint8)
            for s in range(policy_mat.shape[0]):
                policy_mat[s, policy[s]] = 1
            _mappable = ax.imshow(policy_mat, cmap="afmhot")
            ax.set_title("solution")
            fig.colorbar(_mappable)            
        
        fig.suptitle("iter: %d" % vis_iter)
        fig.savefig(os.path.join(save_dir, "%6d.png" % vis_iter))
        plt.clf()
        
    # plt.show() 
    # Write the video.
    ims = []
    for fname in [_ for _ in sorted(os.listdir(save_dir)) if _.endswith(".png")]:
        ims.append(Image.open(os.path.join(save_dir, fname)))
    
    with imageio.get_writer(os.path.join(save_dir, "output.mp4"), fps=5) as video:
        for im in ims:
            frame = im.convert("RGB")
            frame = np.array(frame, dtype=np.uint8)
            video.append_data(frame)
    
    for fname in [_ for _ in sorted(os.listdir(save_dir)) if _.endswith(".png")]:
        os.remove(os.path.join(save_dir, fname))
        

def add_noise(x: np.ndarray,
              noise=.1):
    if noise is None: return x
    noise = np.random.normal(scale=noise, size=x.shape)
    return x + noise
        
        
def phi_exp_factory(p, q=1):
    assert p % 2 == 1 & q % 2 == 1
    return lambda x: np.exp(np.power(np.abs(np.array(x)), p/q) * ((np.array(x) > 0).astype(np.float32) * 2 - 1))

def phi_exp_inv_factory(p, q=1):
    assert p % 2 == 1 & q % 2 == 1
    def phi_exp_inv(x: np.ndarray):
        x = np.array(x)
        sign = (x > 0).astype(np.float32) * 2 - 1
        x = np.abs(x)
        x = np.exp(sign * np.power(x, q/p))
        return x
    return phi_exp_inv

def phi_exp_combined_factory(p_1, q_1=1, p_2=1, q_2=1):
    assert p_1 % 2 == 1 & q_1 % 2 == 1
    assert p_2 % 2 == 1 & q_2 % 2 == 1
    phi_exp = phi_exp_factory(p_1, q_1)
    phi_exp_inv = phi_exp_inv_factory(p_2, q_2)
    
    def phi_exp_combined(x: np.ndarray):
        flag = np.float32(np.abs(np.array(x)) > 1)
        return phi_exp(x) * flag + phi_exp_inv(x) * (1-flag)
    
    return phi_exp_combined

def phi_poly_factory(p=1):
    def phi_poly(x: np.ndarray):
        x = np.array(x)
        return np.power(1+p*x, p)
    return phi_poly

def solve_zero_point_by_binary_searching(f):
    # The function f is assumed to be monotone decreasing. We want to find the zero point of f.
    left, right = -1e10, 1e10
    ct = 0
    while right - left > 1e-10:
        mid = (left + right) / 2
        if f(mid) > 0:
            left = mid
        else:
            right = mid
        ct += 1
        if ct > 1000:
            print("Binary searching failed: cannot find the zero point within 1000 iterations.")
            break
    return (left + right) / 2

def tsallis_entropy_funcs_factory(q):
    assert q > 1
    
    def psi(x):
        return (np.power(x, q) - 1) / (q-1)
    
    def psi_prime(x):
        return np.power(x, q-1) * q / (q-1)
    
    def psi_prime_inv(x):
        return np.power(x * (q-1)  / q, 1/(q-1))
    
    return psi, psi_prime, psi_prime_inv