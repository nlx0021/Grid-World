import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.MDP import MDP
from utils.random_board import generate_random_board, generate_one_goal_board
from utils.utils import *
from models.Random_model import RandomMDP
from models.Grid_world import Grid_world

EPSILON = 1e-13

def run(model: RandomMDP,
        max_iter=10000,
        param_list=[[{"alg": "Softmax_NPG"}, .1]],
        exp_mode="multi-run",
        metric="rho",
        noise=None,
        seed=21,
        clip=1e-13,
        path="./Figures/Exp_Exp.pdf"):
    '''
        param_list:
            [[mode1, step1], [mode2, step2] ... ]
    '''
    assert exp_mode in ["multi-run", "local-rate", "policy-converge", "single-run"]
    if exp_mode != "multi-run":
        assert len(param_list) == 1, "Only support one algorithm."
        assert noise is None, "Please do not add noise."
    
    log_diff_dict = {str(param): [] for param in param_list}
    model.solve_mdp(mode={"alg": "policy_iteration"},
                    epsilon=EPSILON)
    
    V_star = model.mdp.V.copy()
    delta = model.mdp.compute_delta()

    for (mode, step_size, *extra_args) in tqdm(param_list):
        step_size_increasing = False
        if extra_args != []:
            step_size_increasing = extra_args[0]
        model.mdp.init_policy_and_V(random_init=True, seed=seed)
        return_dict = model.solve_mdp(mode=mode,
                                        max_iter=max_iter,
                                        step_size=step_size,
                                        verbose=True,
                                        need_return=True,
                                        noise=noise,
                                        seed=seed,
                                        step_size_increasing=step_size_increasing)
        V_list = return_dict["V_list"]
        if exp_mode == "policy-converge":
            policy_list = return_dict["policy_list"]
        if metric == "rho":
            log_diff_list = np.array([np.log(max((V_star - V).mean(), EPSILON)) for V in V_list])
        elif metric == "infty":
            log_diff_list = np.array([np.log(max((V_star - V).max(), EPSILON)) for V in V_list])
        elif metric == "random-rho":
            metric_rho = np.random.uniform(0, 1, size=(model.mdp.S_size,))
            metric_rho = metric_rho / metric_rho.sum()
            log_diff_list = np.array([np.log(np.dot(V_star - V, metric_rho) + EPSILON) for V in V_list])
        
        log_diff_list = log_diff_list[log_diff_list > clip]
        log_diff_dict[str([mode, step_size])] = (log_diff_list)
        # max_iter = min(max_iter, len(log_diff_list))
        
    if exp_mode == "multi-run":
    # Plot the curve.
        fig = plt.figure(figsize=(5,4))
        ax = plt.axes()
        for (mode, step_size, *_) in param_list:   
            diff_lists = log_diff_dict[str([mode, step_size])]
            label = mode.get("label", mode["alg"])
            # ax.plot(np.arange(max_iter), diff_lists[:max_iter], '-', label=str(label))
            ax.plot(np.arange(len(diff_lists)), diff_lists, '-', label=str(label))

        ax.set_xlabel("iters")
        ax.set_ylabel("log value error")
        ax.legend()
        ax.grid(True)
        ax.grid(alpha=0.3)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
        # plt.show()
        plt.savefig(path)
        
    elif exp_mode == "single-run":
        fig = plt.figure(figsize=(5,4))
        ax = plt.axes()
        for (mode, step_size) in param_list:   
            diff_lists = log_diff_dict[str([mode, step_size])]
            label = mode.get("label", mode["alg"])
            ax.plot(np.arange(max_iter), diff_lists[:max_iter], '-', label=str(label))

        ax.set_xlabel("iters")
        ax.set_ylabel("log value error")
        ax.grid(True)
        ax.grid(alpha=0.3)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
        # plt.show()
        plt.title(label=str(label))
        plt.savefig(path)
    
    elif exp_mode == "local-rate":
        assert mode.get("phi", None) is not None, "Please provide the phi function."
        phi = mode["phi"]
        diff_lists = log_diff_dict[str([mode, step_size])]
        fig = plt.figure(figsize=(5,4))
        ax = plt.axes()
        ax.plot(np.arange(max_iter), diff_lists[:max_iter], '-', label=str("Convergence curve"))
        # ax.plot(diff_lists, '-', label=str("Convergence curve"))
    
        theory_rate = np.log(phi(-step_size * delta) / phi(0))
        # Compute the interception.
        # max_iter = len(diff_lists)
        a = max_iter // 100 * 99
        v_a = diff_lists[a]
        interception = v_a - theory_rate * a
        
        theory_lists = np.arange(max_iter) * theory_rate + interception
        ax.plot(theory_lists, '--', color="red", label=str("Local convergence rate"))
        
        ax.set_xlabel("iters")
        ax.set_ylabel("log value error")
        ax.legend()
        ax.grid(True)
        ax.grid(alpha=0.3)        
        # plt.title(mode["label"])
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
        # plt.show()
        plt.savefig(path)
        
    elif exp_mode == "policy-converge":
        last_policy = policy_list[-1]
        _temp = []
        for s in tqdm(range(model.mdp.S_size)):
            Divergence_list = np.array([np.linalg.norm(last_policy[s] - policy[s], ord=1) for policy in policy_list])
            _temp.append(Divergence_list)
        
        Divergence_result = np.mean(np.array(_temp), axis=0)
        
        # fig = plt.figure(figsize=(5,4))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,8))
        # ax = plt.axes()
        ax1.plot(Divergence_result, '-', label=str("$|| \pi^k - \pi^\mathrm{last} ||_1$"))
        ax1.legend()
        ax1.grid(True)
        ax1.grid(alpha=0.3)        
        # plt.show()
        # print(last_policy)
        
        ax2 = model.visualize_prob_policy(ax2, verbose=True)
        fig.savefig(path, pad_inches=0.2, bbox_inches="tight")
        
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='all')
    args = parser.parse_args()
    
    exp_name = args.type

    S_size = 50
    A_size = 10
    gamma = .98

    seed = 42
    random.seed(seed)
    model = RandomMDP(S_size=S_size,
                        A_size=A_size,
                        gamma=gamma,
                        seed=seed)     
    
    H, W = 5, 5 
    board = generate_random_board(H, W, p_1=.1, p_2=.1)
        
    grid_world = Grid_world(board,
                            gamma=gamma,
                            win_reward=1,
                            punish_reward=-.5)      
    
    
    # run(
    #         model=model,
    #         max_iter=3000,
    #         param_list=[
    #             # [{"alg": "phi", "label": "softmax NPG", "phi": phi_exp_inv_factory(1, 1)}, 0.5],
    #             # [{"alg": "policy_iteration", "label": "PI"}, None]
    #             [{"alg": "policy_descent", "label": "PPG with constant step size"}, 1],
    #             [{"alg": "projected_Q_descent", "label": "PPG with increasing step size"}, 1, True],
    #             # [{"alg": "projected_Q_descent", "label": "PQA"}, 1],
    #             # [{"alg": "softmax", "label": "softmax PG"}, 3]
    #         ],
    #         metric="rho",
    #         exp_mode="multi-run",
    #         noise=None,
    #         seed=seed,
    #         path="PPG-1.pdf",
    #         clip=-35
    # )
    
    
    S_size = 50
    A_size = 15
    gamma = .98

    seed = 42
    random.seed(seed)
    model = RandomMDP(S_size=S_size,
                        A_size=A_size,
                        gamma=gamma,
                        seed=seed)       
    
    run(
            model=model,
            max_iter=3000,
            param_list=[
                # [{"alg": "phi", "label": "softmax NPG", "phi": phi_exp_inv_factory(1, 1)}, 0.5],
                # [{"alg": "policy_iteration", "label": "PI"}, None]
                [{"alg": "policy_descent", "label": "PPG with $\eta_k=2$"}, 2],
                [{"alg": "policy_descent", "label": "PPG with $\eta_k=0.05 / (\\tilde\mu \gamma^{2k+1})$"}, .05, True],
                # [{"alg": "projected_Q_descent", "label": "PQA"}, 1],
                # [{"alg": "softmax", "label": "softmax PG"}, 3]
            ],
            metric="rho",
            exp_mode="multi-run",
            noise=None,
            seed=seed,
            path="PPG-1.pdf",
            clip=-35
    )