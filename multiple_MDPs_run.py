import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks
from tqdm import tqdm

from core.MDP import MDP
from utils.random_board import generate_random_board, generate_one_goal_board
from utils.utils import *
from models.Grid_world import Grid_world
from models.Random_model import RandomMDP

EPSILON = 1e-21

def multiple_MDPs_run(model_list: list,
        max_iter=10000,
        param=[{"alg": "Softmax_NPG"}, .1],
        metric="rho",
        noise=None,
        need_kappa=False,
        seed=21):
    '''
        param_list:
            [[mode1, step1], [mode2, step2] ... ]
    '''
    
    log_diff_dict = {key: [] for (model, key) in model_list}
    delta_dict = {key: 0 for (model, key) in model_list}
    for (model, key) in tqdm(model_list):
        model.solve_mdp(mode={"alg": "policy_iteration"},
                        epsilon=EPSILON)
        
        V_star = model.mdp.V.copy()
        delta = model.mdp.compute_delta()
        kappa_dict = {}

        mode, step_size = param
        model.mdp.init_policy_and_V(random_init=True, seed=seed)
        return_dict = model.solve_mdp(mode=mode,
                                        max_iter=max_iter,
                                        step_size=step_size,
                                        verbose=True,
                                        need_return=True,
                                        noise=noise,
                                        seed=seed)
        V_list = return_dict["V_list"]
        if metric == "rho":
            log_diff_list = np.array([np.log((V_star - V).mean() + EPSILON) for V in V_list])
        elif metric == "infty":
            log_diff_list = np.array([np.log((V_star - V).max() + EPSILON) for V in V_list])
        elif metric == "random-rho":
            metric_rho = np.random.uniform(0, 1, size=(model.mdp.S_size,))
            metric_rho = metric_rho / metric_rho.sum()
            log_diff_list = np.array([np.log(np.dot(V_star - V, metric_rho) + EPSILON) for V in V_list])
        
        log_diff_dict[key].append(log_diff_list)
        delta_dict[key] = delta

    # Plot the results for each model in a single figure.
    plt.figure(figsize=(10, 6))
    for (model, key) in model_list:
        log_diff_list = log_diff_dict[key][0]
        delta = delta_dict[key]
        plt.plot(log_diff_list, label=f"{key} (delta={delta:.4f})")
    plt.xlabel("Iteration")
    plt.ylabel(f"Log {metric}")
    # Clip the y-axis to [-21, 0] for better visualization.
    plt.ylim(-21, 0)
    plt.title("Convergence of MDP Solvers")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    seed = 42
    model_list = [(RandomMDP(S_size=s, A_size=10, seed=seed, gamma=0.9), "$|\mathcal{S}|=%d$" % s) for s in [5,10,25,50,100]]
    
    multiple_MDPs_run(model_list=model_list,
        max_iter=10000,
        # param=[{"alg": "softmax_NPG"}, 1],
        param=[{"alg": "policy_descent"}, 1],
        metric="rho",
        noise=None,
        need_kappa=False,
        seed=seed)