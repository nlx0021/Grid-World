import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import xticks, yticks
from tqdm import tqdm

from core.MDP import MDP
from utils.random_board import generate_random_board, generate_one_goal_board
from utils.utils import compute_visit_prob, vis_heat_map
from models.Grid_world import Grid_world
from models.Random_model import RandomMDP


def run(H_range=[5,15],
        W_range=[5,15],
        gamma=.9, win_reward=1,
        step_size=1,
        mode="policy_iteration"):
    
    hh = np.arange(H_range[0], H_range[1]+1)
    ww = np.arange(W_range[0], W_range[1]+1)
    HH, WW = np.meshgrid(hh, ww)
    
    step_rec = np.zeros_like(HH)
    upper_rec = np.zeros_like(HH)
    
    for H_idx, H in tqdm(enumerate(range(H_range[0], H_range[1]+1))):
        for W_idx, W in enumerate(range(W_range[0], W_range[1]+1)):
            
            board = generate_one_goal_board(H, W)
            
            one_grid_world = Grid_world(board,
                                        gamma,
                                        win_reward,
                                        punish_reward=-10)
            
            one_grid_world.mdp.init_policy_and_V(random_init=True)
            
            return_dict =  one_grid_world.solve_mdp(mode=mode,
                                                    init=False, verbose=True, need_return=True, step_size=step_size)
            
            V_list = return_dict["V_list"]
            
            step = len(V_list) - 1
            
            step_rec[H_idx, W_idx] = step
            
            Delta_theory = gamma ** (H+W-3) * (1-gamma)
            if mode == "policy_iteration":
                upper = np.log(2/(Delta_theory * (1-gamma))) / (1-gamma)
            elif mode == "projected_Q_descent":
                upper = 2 / (step_size*Delta_theory) * (1 / (step_size*(1-gamma)) + 1 / (1-gamma)**2) * (1/Delta_theory + step_size) - 1
            elif mode == "policy_descent":
                upper = 0
            
            upper_rec[H_idx, W_idx] = upper
            
            
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    
    ax3.plot_surface(HH,WW,step_rec,cmap='rainbow') 
    # if mode == "policy_iteration":
    #     ax3.plot_surface(HH,WW,upper_rec,cmap='rainbow')
    
    plt.show()
            
          
            
def vis_process(grid_world: Grid_world):
    
    V_old = -100 * np.ones((grid_world.H * grid_world.W))  
    grid_world.mdp.init_policy_and_V(random_init=True)  
    grid_world.visualize_policy()
    while True:
        
        return_dict = grid_world.solve_mdp(
            mode="policy_iteration",
            max_iter=1,
            need_return=True
        )
        
        # grid_world.visualize_policy()
        V_list = return_dict["V_list"]

        V = V_list[1]
        if np.linalg.norm(V_old - V, ord=np.inf) < 1e-23:
            
            break
        
        V_old = V
        print(np.linalg.norm(V))



def fix_delta(L=40,
              gamma=.9, win_reward=1,
              step_size=1,
              mode="policy_iteration"):
    
    S_size_list = []
    step_list = []
    
    for H in tqdm(range(2, L-1)):
        
        W = L - H
        S_size = H * W
        
        board = generate_one_goal_board(H, W)
        
        one_grid_world = Grid_world(board,
                                    gamma,
                                    win_reward,
                                    punish_reward=-10)
        
        one_grid_world.mdp.init_policy_and_V(random_init=True)
        
        return_dict = one_grid_world.solve_mdp(mode=mode,
                                               init=False, verbose=True, need_return=True, step_size=step_size)
        
        print("Delta: %f" % one_grid_world.mdp.compute_delta())
        
        V_list = return_dict["V_list"]
        
        step = len(V_list) - 1  
        
        print(step)      
    
        S_size_list.append(S_size)
        step_list.append(step)
        
    _temp = list(zip(S_size_list, step_list))
    _temp = sorted(_temp, key=lambda x: x[0])
    
    S_size_list = [_[0] for _ in _temp]
    step_list = [_[1] for _ in _temp]
    
    plt.plot(S_size_list, step_list)
    plt.show()



def exp_1(S_size=100,
          A_size=20,
          gamma=.99,
          step_size=1,
          exp_num=10,
          max_iter=10000,
          mode_list=["projected_Q_descent",
                     "softmax",
                     "softmax_adaptive",
                     "softmax_temp"]):
    
    log_diff_dict = {mode: [] for mode in mode_list}
    for exp in tqdm(range(exp_num)):
        
        seed = np.random.randint(65536)
        model = RandomMDP(S_size=S_size,
                          A_size=A_size,
                          gamma=gamma,
                          seed=seed)
        
        model.solve_mdp(mode="policy_iteration",
                        epsilon=1e-20)
        
        V_star = model.mdp.V.copy()

        for mode in mode_list:
            
            model.mdp.init_policy_and_V(random_init=True)
            return_dict = model.solve_mdp(mode=mode,
                                          max_iter=max_iter,
                                          step_size=step_size,
                                          verbose=True,
                                          need_return=True)
            V_list = return_dict["V_list"]
            
            log_diff_list = [np.log((V_star - V).max()) for V in V_list]
            
            log_diff_dict[mode].append(log_diff_list)
        
    # Plot the curve.
    ax = plt.axes()
    for mode in mode_list:
        
        diff_lists = log_diff_dict[mode]
        
        # 1. Clip the value list.
        x_min = min([len(diff_list) for diff_list in diff_lists])
        clipped_diff_lists = [diff_list[:x_min] for diff_list in diff_lists]
        
        # 2. Compute the mean and dev.
        diff_mean_list = [
            np.mean(values) for values in zip(*clipped_diff_lists)
        ]
        diff_dev_list = [
            np.sqrt(np.var(values)) for values in zip(*clipped_diff_lists)
        ]

        x = np.arange(len(diff_mean_list))
        ax.plot(x, diff_mean_list, '-', label=mode)
        
        y_high, y_low = np.array(diff_mean_list) + 1.98 * np.array(diff_dev_list),   \
                        np.array(diff_mean_list) - 1.98 * np.array(diff_dev_list)

        ax.fill_between(x, y_low, y_high, alpha=.5)

    ax.set_xlabel("iters")
    ax.set_ylabel("log value error")
    ax.legend()
    plt.show()
    
    
def exp_2_for_softmax(step_size_list=[.01, .1, 1, 5, 10, 20, 50, 100],
                      S_size=100,
                      A_size=20,
                      gamma=.99,
                      max_iter=10000,
                      mode="softmax",
                      save_return=False):
    
    seed = np.random.randint(65536)
    model = RandomMDP(S_size=S_size,
                      A_size=A_size,
                      gamma=gamma,
                      seed=seed)    
    
    model.solve_mdp(mode="policy_iteration",
                    epsilon=1e-20)
    
    V_star = model.mdp.V.copy()    
    
    log_diff_dict = {step_size: [] for step_size in step_size_list}
    
    for step_size in tqdm(step_size_list):
        
        model.mdp.init_policy_and_V(random_init=True)
        
        return_dict = model.solve_mdp(mode=mode,
                                      max_iter=max_iter,
                                      step_size=step_size,
                                      verbose=True,
                                      need_return=True)
        V_list = return_dict["V_list"]
        
        log_diff_list = [np.log((V_star - V).max()) for V in V_list]
        log_diff_dict[step_size] = log_diff_list
        
        if save_return:
            save_path = "./log_data/return_dict_stepsize%.2f.npy" % step_size
            np.save(save_path, return_dict)
        
    ax = plt.axes()
    for step_size in step_size_list:
        
        ax.plot(log_diff_dict[step_size], '-', label=str("step_size: %f" % step_size))
        
    ax.set_xlabel("iters")
    ax.set_ylabel("log value error")
    ax.legend()
    plt.show()  
    
    
def exp_3_local_rate(step_size=.1,
                     S_size=100,
                     A_size=20,
                     gamma=.99,
                     max_iter=10000,
                     mode="softmax_NPG"):  
    
    seed = np.random.randint(65536)
    model = RandomMDP(S_size=S_size,
                      A_size=A_size,
                      gamma=gamma,
                      seed=seed)
    
    # H, W = 10, 5
    # board = generate_one_goal_board(H, W)
        
    # model = Grid_world(board,
    #                    gamma,
    #                    win_reward=1,
    #                    punish_reward=0)        
    
    model.solve_mdp(mode="policy_iteration",
                    epsilon=1e-20)
    
    V_star = model.mdp.V.copy()  
    delta = model.mdp.compute_delta()  

    model.mdp.init_policy_and_V(random_init=True)
    
    return_dict = model.solve_mdp(mode=mode,
                                  max_iter=max_iter,
                                  step_size=step_size,
                                  verbose=True,
                                  need_return=True)
    V_list = return_dict["V_list"]
    
    log_diff_list = np.array([np.log((V_star - V).max()) for V in V_list])
    this_errors = log_diff_list[1:]
    last_errors = log_diff_list[:-1]
    log_inter_diff = this_errors - last_errors
    
    ax = plt.axes()
    ax.plot(log_inter_diff, '-', label=str("Convergence curve"))
    
    if mode == "softmax_NPG":
        theory_rate = np.log(np.exp(-step_size * delta))
        ax.axhline(theory_rate, color="red")
    
    ax.legend()
    plt.show()
    
    
def improvement_lower_bound(seed=21,
                            eta_range=(1e-2, 100),
                            mode="policy_descent"):
    
    np.random.seed(seed)
    
    model = RandomMDP(seed=seed)
    
    S_size = model.S_size
    A_size = model.A_size
    gamma = model.gamma
    P = model.P
    
    # Randomly choose a states and generate policy and Advantage functions.
    random_s = np.random.randint(S_size)
    
    if mode == "softmax" or mode == "softmax_adaptive":
        random_param = np.random.rand(S_size, A_size) * .1
        random_pi_k = np.exp(random_param) / np.sum(np.exp(random_param), axis=1, keepdims=True)
    else:
        random_pi_k = np.random.rand(S_size, A_size) + 1e-5
        random_pi_k = random_pi_k / np.sum(random_pi_k, axis=1, keepdims=True)                  # Normalize.
    random_pi_k_s = random_pi_k[random_s]
    random_A_k_s = np.random.normal(size=(A_size,)) * .3 / (1-gamma)
    random_A_k_s[-1] = -np.dot(random_A_k_s[:-1], random_pi_k_s[:-1]) / random_pi_k_s[-1]       # E[A] = 0.
    
    random_mu = np.random.rand(S_size) + 1e-5
    random_mu = random_mu / random_mu.sum()
    
    d_k_s = compute_visit_prob(P, random_pi_k, random_mu, gamma)[random_s]
    
    # Update.
    def proj_to_simplex(policy):
        
        _policy = policy.copy()
        _policy_sorted = np.sort(_policy)
        
        for i in range(A_size-1, 0, -1):
            t_i = (np.sum(_policy_sorted[i:]) - 1) / (A_size - i)
            if t_i >= _policy_sorted[i-1]:
                t = t_i
                break
        
        else:
            t = (np.sum(_policy_sorted) - 1) / A_size
        
        policy = np.clip(_policy - t, a_min=0, a_max=1)
        policy = policy / np.sum(policy)

        return policy
    
    
    # Plot the curve.
    real_list = []
    bound_list = []
    eta_list = []
    
    for eta in np.linspace(eta_range[0], eta_range[1], num=1000):
    
        if mode == "policy_descent":
            random_pi_next_s = proj_to_simplex(
                random_pi_k_s + eta / (1-gamma) * random_A_k_s * d_k_s
            )
        elif mode == "softmax":
            random_param_next_s = random_param[random_s] + eta / (1-gamma) * random_A_k_s * d_k_s * random_pi_k_s
            random_pi_next_s = np.exp(random_param_next_s) / np.exp(random_param_next_s).sum()
        elif mode == "softmax_adaptive":
            random_param_next_s = random_param[random_s] + eta / (1-gamma) * random_A_k_s * d_k_s
            random_pi_next_s = np.exp(random_param_next_s) / np.exp(random_param_next_s).sum()            
            
        # Real value.
        real = np.dot(random_pi_next_s, random_A_k_s)
        
        real_list.append(real)
        eta_list.append(eta)
        
    ax = plt.axes()
    ax.plot(eta_list, real_list, '-', label='real')
    if mode == "policy_descent":
        ax.axhline(random_A_k_s.max())
    else:
        a_tilde = np.argmax(random_A_k_s * random_pi_k_s)
        ax.axhline(random_A_k_s[a_tilde] * random_pi_k_s[a_tilde] / (random_pi_k_s[a_tilde]))
    
    ax.set_xlabel("eta")
    ax.set_ylabel("improvement")
    ax.legend()
    plt.show()    
    
    

if __name__ == '__main__':
    
    # run(mode="projected_Q_descent",
    #     step_size=100)
    
    # run(mode="policy_descent",
    #     step_size=100000)    
    
    # fix_delta(L=50,
    #           step_size=10,
    #           mode="policy_iteration")
    
    # exp_1(step_size=30,
    #       S_size=20,
    #       A_size=5,
    #       max_iter=50000,
    #       exp_num=1)
    
    # exp_2_for_softmax(S_size=20,
    #                   A_size=5,
    #                   gamma=.95,
    #                   max_iter=10000,
    #                   mode="softmax_NPG",
    #                   save_return=True)
    
    exp_3_local_rate(step_size=.1,
                     S_size=20,
                     A_size=5,
                     gamma=.95,
                     max_iter=30000,
                     mode="softmax_NPG")    
    
    # return_dict = np.load("./log_data/return_dict_stepsize1000.00.npy",
    #                       allow_pickle=True).item()
    # vis_iter_list = list(range(0, 1000, 10)) + list(range(2000, 20000, 100))
    # vis_heat_map(return_dict=return_dict,
    #              save_dir="./",
    #              vis_iter_list=vis_iter_list)
    
    # for seed in range(20):
    #     improvement_lower_bound(seed=seed,
    #                             eta_range=(.001, 30),
    #                             mode="softmax_adaptive")
    
    # random.seed(21)
    # H = 3; W = 25
    # # board = generate_random_board(10, 10, .2, .05)
    # board = generate_one_goal_board(H, W)
    
    # gamma = .9
    # win_reward = 1,
    # punish_reward = -10
    
    # one_grid_world = Grid_world(board,
    #                             gamma,
    #                             win_reward,
    #                             punish_reward)   
    
    # vis_process(one_grid_world) 
    # V_list = one_grid_world.solve_mdp(mode="policy_iteration",
    #                                   need_return=True)
    # [print(np.linalg.norm(V, ord=1)) for V in V_list]