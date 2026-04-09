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

def run(model: RandomMDP,
        max_iter=10000,
        param_list=[[{"alg": "Softmax_NPG"}, .1]],
        exp_mode="multi-run",
        metric="rho",
        noise=None,
        need_kappa=False,
        seed=21):
    '''
        param_list:
            [[mode1, step1], [mode2, step2] ... ]
    '''
    assert exp_mode in ["multi-run", "local-rate", "policy-converge"]
    if exp_mode != "multi-run":
        assert len(param_list) == 1, "Only support one algorithm."
        assert noise is None, "Please do not add noise."
    
    log_diff_dict = {str(param): [] for param in param_list}
    model.solve_mdp(mode={"alg": "policy_iteration"},
                    epsilon=EPSILON)
    
    V_star = model.mdp.V.copy()
    delta = model.mdp.compute_delta()
    kappa_dict = {}

    for (mode, step_size) in tqdm(param_list):
        model.mdp.init_policy_and_V(random_init=True, seed=seed)
        return_dict = model.solve_mdp(mode=mode,
                                        max_iter=max_iter,
                                        step_size=step_size,
                                        verbose=True,
                                        need_return=True,
                                        noise=noise,
                                        seed=seed)
        V_list = return_dict["V_list"]
        if exp_mode == "policy-converge":
            policy_list = return_dict["policy_list"]
        if metric == "rho":
            log_diff_list = np.array([np.log((V_star - V).mean() + EPSILON) for V in V_list])
        elif metric == "infty":
            log_diff_list = np.array([np.log((V_star - V).max() + EPSILON) for V in V_list])
        elif metric == "random-rho":
            metric_rho = np.random.uniform(0, 1, size=(model.mdp.S_size,))
            metric_rho = metric_rho / metric_rho.sum()
            log_diff_list = np.array([np.log(np.dot(V_star - V, metric_rho) + EPSILON) for V in V_list])
        
        if need_kappa:
            kappa = return_dict.get("kappa", None)
            assert kappa is not None, "kappa only supports for Softmax_PG."
            kappa_dict[str([mode, step_size])] = kappa
        
        log_diff_dict[str([mode, step_size])] = (log_diff_list)
        max_iter = min(max_iter, len(log_diff_list))
    
    if need_kappa:
        print("Kappa dict: ", kappa_dict)
        # max_iter = max(max_iter, len(log_diff_list))
        
    if exp_mode == "multi-run":
    # Plot the curve.
        fig = plt.figure(figsize=(10,6))
        ax = plt.axes()
        for (mode, step_size) in param_list:   
            diff_lists = log_diff_dict[str([mode, step_size])]
            label = mode.get("label", mode["alg"])
            if need_kappa:
                label = label + " (kappa=%.4f)" % kappa_dict[str([mode, step_size])]
            ax.plot(np.arange(max_iter), diff_lists[:max_iter], '-', label=str(label))
        # Clip the y-axis to better show the difference.
        ax.set_ylim(-14, 3)
        ax.set_xlabel("iters")
        ax.set_ylabel("log value error")
        # Make Legend larger and clearer.
        ax.legend(fontsize=12)
        ax.grid(True)
        ax.grid(alpha=0.3)
        plt.show()
        # plt.savefig("./outputs/Convergence Curve.png")   # Hard Coding FIXME
    
    elif exp_mode == "local-rate":
        assert mode.get("phi", None) is not None, "Please provide the phi function."
        phi = mode["phi"]
        diff_lists = log_diff_dict[str([mode, step_size])]
        this_errors = log_diff_list[1:]
        last_errors = log_diff_list[:-1]
        log_inter_diff = this_errors - last_errors
        fig = plt.figure(figsize=(5,4))
        ax = plt.axes()
        ax.plot(log_inter_diff, '-', label=str("Convergence curve"))
    
        theory_rate = np.log(phi(-step_size * delta) / phi(0))
        ax.axhline(theory_rate, color="red")
        
        ax.legend()
        ax.grid(True)
        ax.grid(alpha=0.3)        
        plt.show()
        
    elif exp_mode == "policy-converge":
        # # Another optimal policy.
        # return_dict = model.solve_mdp(mode={"alg": "policy_descent"}, step_size=10, verbose=True,
        #                 epsilon=EPSILON, need_return=True)
        # last_policy = return_dict["policy_list"][-1]
        # Otherwise
        last_policy = policy_list[-1]
        # Search for not monotonic state.
        _temp = []
        for s in tqdm(range(model.mdp.S_size)):
            Divergence_list = np.array([np.linalg.norm(last_policy[s] - policy[s]) ** 2 for policy in policy_list])
            # diff = Divergence_list[1:] - Divergence_list[:-1]
            # if np.sum(diff > 1e-5) > 0:
            #     print("发现非单调: 状态 %d" % s)
            #     import pdb; pdb.set_trace()
            #     break
            _temp.append(Divergence_list)
        
        Divergence_result = np.mean(np.array(_temp), axis=0)
        
        fig = plt.figure(figsize=(5,4))
        ax = plt.axes()
        ax.plot(Divergence_result, '-', label=str("Divergence curve"))
        ax.legend()
        ax.grid(True)
        ax.grid(alpha=0.3)        
        plt.show()
        print(last_policy)
    
    
if __name__ == '__main__':
    
    S_size = 20
    A_size = 10
    gamma = .9
    
    seed = np.random.randint(65536)
    seed = 21
    model = RandomMDP(S_size=S_size,
                      A_size=A_size,
                      gamma=gamma,
                      seed=seed)   
    # np.random.seed(seed)
    # H, W = 10, 10
    # board = generate_one_goal_board(H, W, random=False)
    # board = generate_random_board(H, W, p_1=.2, p_2=.1)
        
    # model = Grid_world(board,
    #                    gamma,
    #                    win_reward=1,
    #                    punish_reward=-1)  
    
    run(
        model=model,
        max_iter=10000,
        param_list=[
            # [{"alg": "escort", "p": 4}, 1],
            # [{"alg": "phi", "label": "Poly(2)", "phi": phi_poly_factory(2)}, 0.01],
            # [{"alg": "phi", "label": "Poly(4)", "phi": phi_poly_factory(4), "step_include_d": True}, 0.01],
            # [{"alg": "phi", "label": "Poly(8)", "phi": phi_poly_factory(8), "step_include_d": True}, 0.01],
            # [{"alg": "phi", "label": "Exp(1,1)", "phi": phi_exp_inv_factory(1, 1)}, 1],
            # [{"alg": "policy_descent"}, 10],
            # [{"alg": "softmax_adaptive", "label": "reshaped SPG"}, 1],
            [{"alg": "softmax", "label": "$\eta=0.01$"}, 0.1],
            [{"alg": "softmax", "label": "$\eta=1$"}, 1], 
            [{"alg": "softmax", "label": "$\eta=100$"}, 100], 
            [{"alg": "softmax", "label": "$\eta=1000$"}, 1000], 
        ],
        metric="rho",
        exp_mode="multi-run",
        noise=None,
        seed=21,
        need_kappa=True
    )
    
    # model.TD_policy_evaluation(epsilon=0, max_iter=100000, fix_steps_size=1.2, step_size_scale=True, max_length=2)